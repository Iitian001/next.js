//! Garbage collection for the persistent backend.
//!
//! GC identifies and tears down tasks that have no reverse references as defined by the reference
//! counts in `parent_count` and `transient_ref_count`.  Tasks are marked `deleted` and then have
//! their outgoing edges teared down recursively.
//!
//! A collected task also has its cell data released immediately rather than waiting for the
//! post-snapshot eviction sweep to erase the whole entry, so the bulk of the memory is reclaimed
//! *before* the snapshot instead of after it. The entry itself stays resident until the tombstone
//! commits, because a concurrent `CleanupOldEdges` scrub may still open it.
//!
//!  The pass runs under the
//! coordinator's GC phase (see
//! [`SnapshotCoordinator::begin_gc`](crate::backend::snapshot_coordinator)) — which excludes normal
//! operations

use std::{fmt::Display, ops::ControlFlow, sync::atomic::Ordering};

use turbo_tasks::{TaskId, TurboTasks, scope_unbounded::scope_unbounded_with};

use crate::backend::{
    TurboTasksBackend,
    operation::{
        AggregationUpdateQueue, CleanupOldEdgesOperation, ExecuteContext, ExecuteContextImpl,
        TaskGuard, capture_all_outgoing_edges,
    },
    storage::{SpecificTaskDataCategory, TaskDataCategory},
    storage_schema::TaskStorageAccessors,
};

/// One unit of GC work.
enum GcJob {
    /// Scan one shard of the resident map (by index) and enqueue its candidates as
    /// [`GcJob::Collect`].
    ScanShard(usize),
    /// Collect a single task
    Collect(TaskId),
}

/// Observability counters for one [`TurboTasksBackend::gc_collect`] pass.
#[derive(Default)]
pub(crate) struct GcStats {
    /// Tasks collected (marked soft-deleted).
    pub collected: usize,
    /// Edges torn down across all collected tasks (children + forward-dependency reverse edges).
    pub edges_deleted: usize,
}

impl Display for GcStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "collected: {collected}, edges_deleted: {edges_deleted}",
            collected = self.collected,
            edges_deleted = self.edges_deleted
        )
    }
}

impl GcStats {
    fn merge(mut self, other: Self) -> Self {
        self.collected += other.collected;
        self.edges_deleted += other.edges_deleted;
        self
    }
}

impl TurboTasksBackend {
    /// An execute context for the garbage collector that does not take an operation guard. Only
    /// valid during GC.
    fn execute_context_gc<'a>(
        &'a self,
        turbo_tasks: &'a TurboTasks<TurboTasksBackend>,
    ) -> impl ExecuteContext<'a> {
        ExecuteContextImpl::new_for_gc(self, turbo_tasks)
    }

    /// Collect all garbage from the task-cache
    ///
    /// Returns [`GcStats`] for the pass.
    pub(crate) fn gc_collect(&self, turbo_tasks: &TurboTasks<TurboTasksBackend>) -> GcStats {
        // TODO(perf): recycle the task ids of collected tasks.

        scope_unbounded_with(
            // Start by scanning all shards
            (0..self.storage.shard_count()).map(GcJob::ScanShard),
            GcStats::default,
            |spawner, job, stats| {
                let task_id = match job {
                    GcJob::ScanShard(index) => {
                        self.storage
                            .gc_scan_shard(index, |task_id| spawner.spawn(GcJob::Collect(task_id)));
                        return ControlFlow::Continue(());
                    }
                    GcJob::Collect(task_id) => task_id,
                };
                let mut ctx = self.execute_context_gc(turbo_tasks);
                // `All` restores Data so the edge capture below can read the Data-category dep
                // sets.
                let mut task = ctx.task(task_id, TaskDataCategory::All);
                // Collectibility was checked when this job was queued but it is possible a
                // concurrent tear down could add an aggregation edge temporarily.  Defensively
                // recheck while holding the lock.
                if !task.is_gc_collectible() {
                    return ControlFlow::Continue(());
                }

                let old_edges = capture_all_outgoing_edges(&task);
                // Clear `immutable` defensively so `resurrect_deleted` can mark the task dirty if
                // it needs to
                task.set_immutable(false);
                // Drop the whole cell payload. This recovers most of the RAM while persistence
                // writes the tombstone.
                let _ = task.take_cell_data();
                task.set_deleted(true);
                if task.new_task() {
                    task.discard_modifications_for_gc_new_task();
                } else {
                    // Persisted ensure it is marked modified so the next snapshot tombstones it.
                    let _ = task.track_modification(SpecificTaskDataCategory::Meta, "gc_deleted");
                }
                stats.collected += 1;

                drop(task);

                stats.edges_deleted += old_edges.len();
                CleanupOldEdgesOperation::run(
                    task_id,
                    old_edges,
                    AggregationUpdateQueue::new(),
                    &mut ctx,
                );
                // recursively spawn all newly collectible tasks
                for candidate in ctx.take_gc_collectible() {
                    spawner.spawn(GcJob::Collect(candidate));
                }
                ControlFlow::Continue(())
            },
            GcStats::merge,
        )
    }

    pub(super) fn gc_pin(&self, task: TaskId, turbo_tasks: &TurboTasks<TurboTasksBackend>) {
        // Once stopping, GC bookkeeping is irrelevant
        if self.stopping.load(Ordering::Acquire) {
            return;
        }
        let mut ctx = self.execute_context(turbo_tasks);
        let existed = ctx.resident_task(task);
        debug_assert!(
            existed.is_some(),
            "pin_task_for_gc: task {task} has no resident entry (pinned an already-collected \
             task?)"
        );
        if let Some(mut guard) = existed {
            guard.update_and_get_transient_ref_count(1);
        }
    }

    pub(super) fn gc_unpin(&self, task: TaskId, turbo_tasks: &TurboTasks<TurboTasksBackend>) {
        // See `gc_pin`: no-op once stopping, so handles finalized during shutdown (after the map is
        // dropped) don't underflow the count.
        if self.stopping.load(Ordering::Acquire) {
            return;
        }
        let mut ctx = self.execute_context(turbo_tasks);
        let existed = ctx.resident_task(task);
        debug_assert!(
            existed.is_some(),
            "unpin_task_for_gc: task {task} has no resident entry (unpinned an already-collected \
             task?)"
        );
        if let Some(mut guard) = existed {
            guard.update_and_get_transient_ref_count(-1);
        }
    }

    /// Runs a full GC pass under the GC phase and returns the number of tasks collected.
    #[doc(hidden)]
    pub fn gc_for_testing(&self, turbo_tasks: &TurboTasks<TurboTasksBackend>) -> usize {
        let _serialize = self.snapshot_in_progress.lock();
        let _gc_phase = self.snapshot_coord.begin_gc();
        self.gc_collect(turbo_tasks).collected
    }
}
