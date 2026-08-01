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
//! operations — and is driven as a fully parallel, unbounded job pool
//! (see [`TurboTasksBackend::gc_collect`]).
//!
//! The GC-specific logic — job types, the pool driver, per-job teardown, and the pin/unpin
//! bookkeeping — as an `impl TurboTasksBackend`. Its entry points are called from `mod.rs`.

use std::{
    fmt::Display,
    ops::ControlFlow,
    sync::atomic::Ordering,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use parking_lot::Mutex;
use rustc_hash::{FxHashMap, FxHashSet};
use turbo_tasks::{TaskId, TurboTasks, scope_unbounded::scope_unbounded_with};

use crate::backend::{
    GC_ROOT_TTL, TurboTasksBackend,
    operation::{
        AggregationUpdateQueue, CleanupOldEdgesOperation, ExecuteContext, ExecuteContextImpl,
        GcCandidate, TaskGuard, capture_all_outgoing_edges,
    },
    storage::{SpecificTaskDataCategory, TaskDataCategory},
    storage_schema::TaskStorageAccessors,
};

/// One unit of GC work. Parallelism is *across* jobs (the unbounded pool in
/// [`TurboTasksBackend::gc_collect`] runs many on different workers); each job is internally
/// sequential. Both variants produce more work, which flows straight back into the same pool.
enum GcJob {
    /// Scan one shard of the resident map (by index) and enqueue its candidates as
    /// [`GcJob::Collect`].
    ///
    /// Seeding the pool with these rather than scanning the whole map up front keeps the scan off
    /// the critical path: the first shard's candidates begin tearing down while the last shard is
    /// still being read.
    ScanShard(usize),
    /// Tear down a single task: scrub its edges, drop its children's `parent_count`, and mark it
    /// soft-deleted. Can discover more work — a child the cleanup drives to `parent_count == 0`
    /// that is itself collectible.
    Collect(TaskId),
}

/// Observability counters for one [`TurboTasksBackend::gc_collect`] pass.
///
/// `collected`/`edges_deleted` are accumulated per drainer and folded at the join (see
/// [`GcStats::merge`]) rather than through shared atomics: at one increment per collected task
/// across every worker, shared counters were several percent of collect time in profiles.
#[derive(Default)]
pub(crate) struct GcStats {
    /// Size of the reconciled GC roots map at the end of the pass. Unlike the other two counters
    /// this is not accumulated per drainer — it is assigned once from the final map, after the
    /// scan-discovered and cascade-discovered roots are folded in and the collected ones removed.
    pub gc_roots: usize,
    /// Tasks collected (marked soft-deleted).
    pub collected: usize,
    /// Edges torn down across all collected tasks (children + forward-dependency reverse edges).
    pub edges_deleted: usize,
    /// Cross-session roots that aged out past the TTL and seeded collection this pass (a subset of
    /// the seeds — the resident scan supplies the rest). Recorded on the `gc` span so the e2e
    /// dogfood can confirm a deleted route's subtree is reclaimed on a *later* session rather than
    /// in-session.
    ///
    /// Like `gc_roots`, assigned once after the pool drains (from the seed set) rather than
    /// accumulated per drainer, so [`GcStats::merge`] leaves it alone.
    pub aged_out_roots: usize,
}

impl Display for GcStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "gc_roots = {gc_roots}, collected: {collected}, edges_deleted: {edges_deleted}, \
             aged_out_roots = {aged_out_roots}",
            gc_roots = self.gc_roots,
            collected = self.collected,
            edges_deleted = self.edges_deleted,
            aged_out_roots = self.aged_out_roots
        )
    }
}

impl GcStats {
    /// Combines two drainers' counts. Addition, so associative and commutative as
    /// [`scope_unbounded_with`] requires.
    fn merge(mut self, other: Self) -> Self {
        self.collected += other.collected;
        self.edges_deleted += other.edges_deleted;
        self.gc_roots += other.gc_roots;
        self
    }
}

impl TurboTasksBackend {
    /// An execute context for the garbage collector that does not take an operation guard. Only
    /// valid while the caller holds the coordinator's GC phase (which provides exclusion); see
    /// [`ExecuteContextImpl::new_for_gc`].
    fn execute_context_gc<'a>(
        &'a self,
        turbo_tasks: &'a TurboTasks<TurboTasksBackend>,
    ) -> impl ExecuteContext<'a> {
        ExecuteContextImpl::new_for_gc(self, turbo_tasks)
    }

    /// Runs a garbage-collection pass under the coordinator's GC phase. Scans the resident map for
    /// collectible tasks (no persistent parent, quiescent, no aggregation edges), re-validates each
    /// under the exclusion, and tears down the ones that are still collectible: scrubbing their
    /// reverse-dependency edges, decrementing their children's `parent_count` (cascading to any
    /// child that reaches 0), marking them deleted, and buffering an on-disk tombstone for the next
    /// persistence commit.
    ///
    /// The pass is fully parallel and unbounded via [`scope_unbounded_with`]: work is a pool of
    /// [`GcJob`]s — scan a shard, or collect a task — and both kinds spawn more, so there are no
    /// synchronization barriers anywhere in the pass.
    ///
    /// Why this is safe to run concurrently under the GC phase (which excludes normal operations
    /// but not the GC jobs from each other):
    /// - Each job builds its own [`ExecuteContext`] (`execute_context_gc`); the concurrent-lock
    ///   detector is per-context, so jobs on different threads holding different task guards do not
    ///   false-positive.
    /// - The storage map is a sharded dashmap: different tasks hit different shards; same-task
    ///   access is serialized by the shard lock. A `ScanShard` holds only a **read** lock on its
    ///   own shard, and a collect elsewhere in the map does not contend with it; a collect landing
    ///   on the shard being scanned simply waits for that shard's (short) read lock.
    /// - Interleaving the scan with the cascade cannot miss or double-collect. Nothing is missed:
    ///   every collectible task is reached either by the scan of the shard it lives in or by the
    ///   cascade that orphans it. Nothing is collected twice: a queued candidate can be collected
    ///   by a sibling job before its own `Collect` runs, so `Collect` re-validates
    ///   `is_gc_collectible` under its write guard and skips if the task is already `deleted` (or
    ///   has since regained an anchor).
    /// - The cascade decrement (`update_and_get_parent_count(-1)`) is a read-modify-write under the
    ///   child's entry write lock, so if two collected parents decrement the same child
    ///   concurrently, exactly one observes the count hit 0 and spawns its collect — no
    ///   double-collect, no lost decrement.
    /// - A collectible task has `parent_count == 0`, so no *surviving* task lists it as a child: a
    ///   task becomes a collect target only after its last persistent parent was itself collected
    ///   (which removed the edge). So a `Collect` never races a decrement of the same task and
    ///   never `ctx.task`-resurrects a task another job just removed.
    ///
    /// Returns [`GcStats`] for the pass plus the reconciled GC roots map (task -> last-anchored-ms)
    /// to persist in the same snapshot commit. The on-disk tombstones are not produced here —
    /// collected tasks are left resident with their `deleted` flag set, and the next snapshot
    /// derives the tombstones from that flag (see `snapshot_and_persist`).
    ///
    /// The roots map is loaded from disk at the start of the pass and lives only for its duration —
    /// it's per-pass state, not backend state, so nothing is held resident between passes (and a
    /// backend that never GCs never reads it).
    pub(crate) fn gc_collect(
        &self,
        turbo_tasks: &TurboTasks<TurboTasksBackend>,
    ) -> (GcStats, Vec<(TaskId, u64)>) {
        // A single wall-clock reading for the whole pass: every root touched here (scan-time
        // resident roots, refreshed roots, and cascade-discovered roots) is stamped with the *same*
        // `now`, so within one pass all timestamps are consistent (and it's one syscall, not
        // three).
        let now = Self::now_ms();

        // Load the previous session's roots map (task id -> last-anchored millis; empty on a fresh
        // or non-persistent database). Maintained locally through this pass and returned
        // for persistence.
        let mut roots = self
            .backing_storage
            .roots()
            .unwrap_or_else(|err| {
                // A corrupt/unreadable roots key shouldn't abort GC — treat it as empty and let the
                // scan below re-discover the resident roots. Cross-session orphans of a genuinely
                // unreadable set are collected once their (re-discovered) roots age out.
                tracing::warn!("failed to read GC roots, treating as empty: {err:?}");
                Vec::new()
            })
            .into_iter()
            .collect::<FxHashMap<TaskId, u64>>();

        // Seed the pool with the resident-map scan, split one job per shard. Each shard job applies
        // the cheap `gc_maybe_collectible` pre-filter under a shard read lock and feeds its hits
        // back as `Collect` jobs. The scan only sees resident tasks; disk-only garbage is collected
        // after it is next restored.
        //
        // TODO(perf): recycle the task ids of collected tasks. `persisted_task_id_factory`
        // (`IdFactoryWithReuse`) can hand out freed ids, and the persisted `next_free_task_id`
        // high-water mark only grows today, so the id space grows unboundedly across churn even
        // though the task set stays flat. Reuse must happen only AFTER the `save_snapshot` that
        // tombstoned the id has committed (a crash before commit leaves the task on disk — reusing
        // its id would alias it), and must be guarded against resurrection: between removal and
        // commit a `get_or_create_task` for the same type could re-mint the id, and the id must not
        // be handed out while any live `OperationVc`/`DetachedVc` still references it. Feed the
        // recycled ids into `persisted_task_id_factory` so the high-water mark can stop growing.

        // The scan only sees resident tasks; disk-only garbage is collected after it is next
        // restored.
        let seeds: Vec<GcJob> = (0..self.storage.shard_count())
            .map(GcJob::ScanShard)
            .collect();

        // Reconcile the carried-forward roots map and decide which roots have aged out (un-anchored
        // past the TTL) and should be collected. This runs *before* the pool, but it does not need
        // the shard scan's `resident_roots`: the retain classifies each carried-forward entry with
        // the same `gc_is_root` predicate the scan uses, so a resident still-root is refreshed here
        // regardless. Roots the scan newly discovers are folded into the map after the pool drains,
        // which is what keeps the scan off the critical path.
        let aged_out = self.gc_roots_refresh_and_age_out(&mut roots, Vec::new(), now);

        // Aged-out roots are candidates for collection, but — unlike the pre-filtered resident
        // candidates the scan produces — they are not guaranteed collectible: one may have been
        // re-anchored, may still hold aggregation edges, or (once restored) may prove to have a
        // live child. Re-validate each under a guard (restoring a non-resident root's Meta)
        // and only seed the genuinely collectible ones; a non-collectible aged-out root is
        // simply left un-collected this pass. Its entry **stays in the roots map** (the
        // refresh above keeps every entry, and only an actual collect removes one below),
        // so it is re-seeded next pass rather than being silently forgotten.
        //
        // The set also drives observability: the collect-site trace below tags seeds that came from
        // an aged-out cross-session root (as opposed to the resident scan), so the e2e dogfood can
        // confirm a deleted route's subtree is reclaimed on a *later* session rather than
        // in-session.
        let mut aged_out_seeds = FxHashSet::default();
        {
            let mut ctx = self.execute_context_gc(turbo_tasks);
            // Bulk-fetch the aged-out roots' Meta in one batched restore rather than a `task` call
            // (and per-task disk restore) each. `is_gc_collectible` only reads Meta fields.
            ctx.for_each_task_meta(aged_out, "gc aged-out root revalidation", |task, _ctx| {
                if task.is_gc_collectible() {
                    aged_out_seeds.insert(task.id());
                }
            });
        }

        // Newly-orphaned tasks discovered as the cascade decrements children to `parent_count == 0`
        // but that are NOT collected this pass (e.g. still anchored by a pin) — they are new
        // durable roots and must enter the map with a fresh timestamp, or they'd never be
        // tracked/aged.
        let discovered_roots = Mutex::new(Vec::<TaskId>::new());

        // Tasks this pass actually marked deleted. `gc_roots_refresh_and_age_out` deliberately
        // leaves every entry in `roots` (it only *seeds* aged-out roots), so this is what tells us
        // which entries may now be dropped from the persisted map. Recorded at the `set_deleted`
        // site below, i.e. only for tasks that really were collected — a seed that was re-validated
        // non-collectible, or never reached, stays in the map and ages again next pass.
        let collected_ids = Mutex::new(Vec::<TaskId>::new());

        // Roots discovered by the shard scans, folded into the map after the pool drains.
        let scanned_roots = Mutex::new(Vec::<TaskId>::new());

        // Each job builds its own GC `ExecuteContext`; see the doc above for the concurrency
        // argument. Counts accumulate into a per-drainer `GcStats` and are folded at the join, so
        // the hot path touches no shared state. The aged-out seeds ride in as `Collect` jobs
        // alongside the per-shard scans.
        let seeds = seeds
            .into_iter()
            .chain(aged_out_seeds.iter().copied().map(GcJob::Collect));
        let mut stats = scope_unbounded_with(
            seeds,
            GcStats::default,
            |spawner, job, stats| {
                let task_id = match job {
                    GcJob::ScanShard(index) => {
                        let roots = self
                            .storage
                            .gc_scan_shard(index, |task_id| spawner.spawn(GcJob::Collect(task_id)));
                        if !roots.is_empty() {
                            scanned_roots.lock().extend(roots);
                        }
                        return ControlFlow::Continue(());
                    }
                    GcJob::Collect(task_id) => task_id,
                };
                let mut ctx = self.execute_context_gc(turbo_tasks);
                // `All` restores Data so the edge capture below can read the Data-category dep
                // sets. The target is resident either way: a scan candidate was
                // read out of the resident map, and a cascade child was just
                // decremented through its resident entry.
                let mut task = ctx.task(task_id, TaskDataCategory::All);
                // Collectibility was checked when this job was queued, but jobs run concurrently: a
                // sibling job's cascade can since have collected this task (it is then `deleted`)
                // or flipped it non-collectible (regained `activeness`/`in_progress`, or gained an
                // aggregation edge). Re-check under the guard we now hold and skip rather than
                // delete — collecting twice would double-run the edge teardown. A task that is
                // still genuinely garbage is re-selected by a later pass.
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
                // Soft-delete rather than remove: a later `ctx.task` on a removed task would
                // resurrect it from disk as a zombie, so it stays resident until a later step
                // hard-deletes it.
                task.set_deleted(true);
                if task.new_task() {
                    // Never persisted: there is nothing on disk to tombstone, so drop it out of the
                    // next snapshot's scan entirely (clearing its modified bits + shard count).
                    // Eviction still removes the resident, soft-`deleted` task.
                    task.discard_modifications_for_gc_new_task();
                } else {
                    // Persisted: force the task into the next snapshot's scan so `process`
                    // tombstones its on-disk copy. `deleted` is transient, so setting it tracks
                    // nothing — explicitly track a meta modification. This matters even in the
                    // collectible states that otherwise leave meta clean (e.g. a pinned parentless
                    // task then unpinned, or one restored parentless from a prior session).
                    let _ = task.track_modification(SpecificTaskDataCategory::Meta, "gc_deleted");
                }
                stats.collected += 1;
                // Record the collect so the roots map can drop this entry (if it had one) after the
                // pass. Touched once per collected task, not per child/dep, so the lock is not a
                // contention hot spot — same argument as `discovered_roots`.
                collected_ids.lock().push(task_id);
                // A span (not a free-standing event) so the collect shows up as a node in the trace
                // tree under the `gc` span — the trace server / `next internal trace` MCP surface
                // spans, and free events attached to no span are not queryable there.
                // `cross_session_root` tags collects seeded from an aged-out root (the
                // deleted-route path) vs the resident scan.
                let _collect_span = tracing::info_span!(
                    "gc collect task",
                    task = %task_id,
                    cross_session_root = aged_out_seeds.contains(&task_id),
                )
                .entered();

                drop(task);

                stats.edges_deleted += old_edges.len();
                CleanupOldEdgesOperation::run(
                    task_id,
                    old_edges,
                    AggregationUpdateQueue::new(),
                    &mut ctx,
                );

                // `CleanupOldEdges` classified every task this cleanup drove past a GC-predicate
                // clause — one that lost its last persistent parent, or its last aggregation edge.
                // Each was classified under the guard the cascade already held, so there is
                // nothing to re-check and no need to re-open the task here.
                for candidate in ctx.take_gc_candidates() {
                    match candidate {
                        // Collecting a task twice is harmless: `Collect` re-validates under its own
                        // write guard and skips one a sibling job already soft-deleted.
                        GcCandidate::Garbage(id) => spawner.spawn(GcJob::Collect(id)),
                        // Parent-less but still anchored (a pin / transient parent) or still
                        // holding aggregation edges — it just became a durable root, so it enters
                        // the roots map with a fresh timestamp and starts aging instead of being
                        // collected. (The shard scan may not have caught this newly-orphaned child:
                        // its shard may already have been scanned before this cascade ran.)
                        GcCandidate::Root(id) => discovered_roots.lock().push(id),
                    }
                }

                ControlFlow::Continue(())
            },
            GcStats::merge,
        );

        // Fold in the roots the shard scans found. Deferred to here (rather than passed into
        // `gc_roots_refresh_and_age_out`) so the scan never blocks the cascade: `or_insert` leaves
        // an already-tracked root's refreshed timestamp alone and only adds roots new to the map
        // this session.
        //
        // Then fold in roots discovered during the cascade (orphaned-but-not-collected children),
        // same rule: new to the map → `now`; already tracked → keep the existing timestamp, since a
        // cascade re-orphaning doesn't reset an already-running clock.
        for id in scanned_roots.into_inner() {
            roots.entry(id).or_insert(now);
        }
        for id in discovered_roots.into_inner() {
            roots.entry(id).or_insert(now);
        }

        // Now drop the entries for tasks this pass actually collected.
        // `gc_roots_refresh_and_age_out` kept every entry (it only seeds aged-out roots),
        // so this is the *only* place a root leaves the map — which is what keeps the
        // persisted set from losing a root that was seeded but not collected. A collected
        // task is gone from the graph, so its entry would otherwise be a permanent stale
        // key in the roots set.
        //
        // Runs after the fold-ins so the two can't fight over an id. They are in fact disjoint (a
        // cascade child is recorded as *either* spawned-and-collected *or* a discovered root, and a
        // collected task fails `gc_is_root` so the scan won't report it), but ordering it this way
        // means a collect always wins, which is the safe direction: the task no longer exists.
        for id in collected_ids.into_inner() {
            roots.remove(&id);
        }

        stats.gc_roots = roots.len();
        stats.aged_out_roots = aged_out_seeds.len();
        (stats, roots.into_iter().collect())
    }

    /// The GC root TTL for this pass. Precedence: the per-backend test override
    /// (`set_gc_root_ttl_for_testing`, race-free across parallel tests) → the
    /// `TURBO_ENGINE_GC_ROOT_TTL_MS` env (for the e2e dogfood, where a process-global is fine) →
    /// [`GC_ROOT_TTL`]. Read each call — GC runs rarely, so this is negligible.
    fn gc_root_ttl(&self) -> Duration {
        let override_ms = self.gc_root_ttl_override_ms.load(Ordering::Relaxed);
        if override_ms != u64::MAX {
            return Duration::from_millis(override_ms);
        }
        match std::env::var("TURBO_ENGINE_GC_ROOT_TTL_MS") {
            Ok(v) => match v.parse::<u64>() {
                Ok(ms) => Duration::from_millis(ms),
                Err(_) => GC_ROOT_TTL,
            },
            Err(_) => GC_ROOT_TTL,
        }
    }

    /// Wall-clock now as millis since the Unix epoch (saturating to 0 before the epoch, which can't
    /// happen in practice). This is the backend layer, not a task-execution context, so
    /// `SystemTime` is fine here (same as `snapshot_and_persist`).
    fn now_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }

    /// Reconcile the GC roots map against this pass's freshly-scanned resident roots, and return
    /// the ids of roots that have aged out (gone un-anchored past the TTL) to seed collection.
    ///
    /// Order matters: the carried-forward map is reconciled by the `retain` **first**, then the
    /// freshly-scanned `resident_roots` are `or_insert`ed. So a resident root already tracked is
    /// refreshed once by the retain (the scan and retain share the `gc_is_root` predicate), and the
    /// insert only *adds* roots new to the map this session — no entry is stamped twice.
    ///
    /// Each carried-forward entry is classified **without restoring** it (a non-inserting
    /// `with_task`), which is both correct and the point — the roots we most want to age out are
    /// the non-resident ones, and we must not pull them back into memory just to look:
    /// - **resident and still a root** ([`TaskStorage::gc_is_root`]: `parent_count == 0 &&
    ///   transient_ref_count > 0`) → refresh `last_anchored_ms = now`. Using the *full*
    ///   `gc_is_root` predicate — not just `transient_ref_count > 0` — is what stops a resident
    ///   task that regained a persistent parent from being refreshed forever as a stale "root" (it
    ///   fails `gc_is_root`, so it ages out and is dropped).
    /// - **resident but no longer a root**, or **not resident** → un-anchored: a non-resident task
    ///   holds no `transient_ref_count` (transient state is in-memory only) and was not re-anchored
    ///   this session (re-anchoring restores it), so it is correctly treated as orphaned. Keep its
    ///   timestamp; if `now - last_anchored_ms > TTL`, drop from the map and return it as a
    ///   collection seed (the aged-out path in `gc_collect` restores + re-validates it before
    ///   collecting).
    ///
    /// Using `gc_is_root` here — the same predicate the scan admits roots with — keeps membership
    /// and refresh from drifting. Runs under the GC phase (exclusion), so the counts don't race
    /// pins.
    fn gc_roots_refresh_and_age_out(
        &self,
        map: &mut FxHashMap<TaskId, u64>,
        resident_roots: Vec<TaskId>,
        now: u64,
    ) -> Vec<TaskId> {
        let ttl_ms = self.gc_root_ttl().as_millis() as u64;

        // Reconcile the carried-forward map (prior-session entries) first. Every entry is
        // re-classified by the *same* `gc_is_root` predicate the scan uses, so a resident root that
        // is still a root is refreshed here — we don't need `resident_roots` to touch it.
        let mut aged_out = Vec::new();
        map.retain(|&id, last_anchored| {
            // Non-inserting: a non-resident root reads as `None` → un-anchored → ages (which is
            // what we want; we don't restore disk-only orphans just to check them). A
            // resident task is a still-live root only if it passes the full
            // `gc_is_root` (parent_count 0 AND anchored), so a re-parented resident
            // task fails here and ages out of the map.
            let still_root = self
                .storage
                .with_task(id, |t| t.gc_is_root())
                .unwrap_or(false);
            if still_root {
                *last_anchored = now;
                return true;
            }
            // Un-anchored: has it aged past the TTL? `now < last_anchored` (clock skew across
            // sessions) is treated as not-yet-aged (saturating), never negative.
            if now.saturating_sub(*last_anchored) > ttl_ms {
                aged_out.push(id);
                false
            } else {
                true
            }
        });

        // Now fold in this pass's resident roots. `or_insert` (not `insert`) so that a root already
        // carried forward keeps the timestamp the retain just refreshed — this only *adds* roots
        // new to the map this session, stamped `now`. (A resident root already in the map was
        // handled by the retain above, since the scan and the retain share the `gc_is_root`
        // predicate; doing this after the retain avoids touching those entries twice.)
        for id in resident_roots {
            map.entry(id).or_insert(now);
        }

        aged_out
    }

    pub(super) fn gc_pin(&self, task: TaskId, turbo_tasks: &TurboTasks<TurboTasksBackend>) {
        // Once stopping, GC bookkeeping is irrelevant (the map is torn down in `stop()`), so
        // pin/unpin become no-ops — also safe against handles finalized during shutdown (a
        // `DetachedVc` dropped during Node teardown unpins *after* `stop()` dropped the map).
        if self.stopping.load(Ordering::Acquire) {
            return;
        }
        // An operation-guarded context so pin runs strictly before or after a collection, never
        // concurrently with it. Deadlock-free: no pin caller already holds a guard, and the GC pass
        // never pins.
        let mut ctx = self.execute_context(turbo_tasks);
        // A pin is an in-session reference from outside the tracked graph (`prevent_gc`, or a
        // `DetachedVc` holding the task's `OperationVc` across NAPI), counted like a transient
        // parent's edge: `transient_ref_count` keeps the task uncollectible and unevictable while
        // > 0. Counting (not a bool) balances each pin against its own unpin.
        //
        // `resident_task` is non-inserting: a pin targets a live reference, so the task must be
        // resident. A missing entry means a pin of an already-collected task (a "zombie
        // `OperationVc`") — surfaced via debug_assert rather than papered over with a blank entry.
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
        self.gc_collect(turbo_tasks).0.collected
    }
}
