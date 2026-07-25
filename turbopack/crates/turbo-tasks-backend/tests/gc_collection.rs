#![feature(arbitrary_self_types)]
#![feature(arbitrary_self_types_pointers)]
#![allow(clippy::needless_return)] // tokio macro-generated code doesn't respect this

mod util;

use anyhow::Result;
use turbo_tasks::{
    GcRoot, ResolvedVc, State, TaskId, TurboTasks, Vc, prevent_gc,
    unmark_top_level_task_may_leak_eventually_consistent_state,
};

use crate::util::create_tt;

/// The `TaskId` backing a resolved `Vc` (its `TaskOutput` node).
fn task_id_of<T>(vc: Vc<T>) -> TaskId {
    Vc::into_raw(vc)
        .try_get_task_id()
        .expect("a resolved Vc should be backed by a task")
}

#[turbo_tasks::value(transparent)]
struct Selector(State<bool>);

#[turbo_tasks::function(operation, root)]
fn create_selector(initial: bool) -> Vc<Selector> {
    Selector(State::new(initial)).cell()
}

#[turbo_tasks::function]
fn leaf(n: u32) -> Vc<u32> {
    Vc::cell(n)
}

#[turbo_tasks::function]
async fn branch_a() -> Result<Vc<u32>> {
    Ok(Vc::cell(1 + *leaf(10).await?))
}

#[turbo_tasks::function]
async fn branch_b() -> Result<Vc<u32>> {
    Ok(Vc::cell(2 + *leaf(20).await?))
}

/// A task that pins itself against GC while executing. Once pinned it must survive collection even
/// after it is disconnected.
#[turbo_tasks::function]
async fn pinned_branch() -> Result<Vc<u32>> {
    prevent_gc();
    Ok(Vc::cell(99))
}

/// Reads exactly one branch depending on the selector; flipping it re-executes and disconnects the
/// previously-read branch (and its subtree), which should drop that branch's `parent_count` to 0.
#[turbo_tasks::function(operation, root)]
async fn select(selector: ResolvedVc<Selector>) -> Result<Vc<u32>> {
    let use_b = *selector.await?.get();
    let value = if use_b {
        *branch_b().await?
    } else {
        *branch_a().await?
    };
    Ok(Vc::cell(value))
}

/// Like `select`, but reads `pinned_branch` instead of `branch_a` when the selector is false.
#[turbo_tasks::function(operation, root)]
async fn select_pinned(selector: ResolvedVc<Selector>) -> Result<Vc<u32>> {
    let use_b = *selector.await?.get();
    let value = if use_b {
        *branch_b().await?
    } else {
        *pinned_branch().await?
    };
    Ok(Vc::cell(value))
}

/// A plain leaf read at the top level of a `run_once`. The current task is `None` there, so the
/// leaf's task is created with no persistent parent — but the transient `Once` task that reads it
/// connects it as a child, anchoring it via `transient_ref_count`. That transient anchor (not a
/// persisted "root" flag) is what keeps such a top-level-read task alive.
#[turbo_tasks::function]
async fn gc_root_leaf() -> Result<Vc<u32>> {
    Ok(Vc::cell(77))
}

/// One leaf per index — distinct tasks, so a wide parent accumulates that many distinct children.
#[turbo_tasks::function]
fn wide_leaf(index: u32) -> Vc<u32> {
    Vc::cell(index)
}

/// Reads `WIDE_FANOUT` distinct children — deliberately above `connect_children`'s 10_000
/// parallelization threshold, so the child-side `parent_count` bump runs through the chunked,
/// parallel `process_new_children` path (each chunk on its own worker context) rather than the
/// serial one.
const WIDE_FANOUT: u32 = 12_000;

#[turbo_tasks::function(operation, root)]
async fn wide_parent() -> Result<Vc<u32>> {
    let mut sum = 0u32;
    for index in 0..WIDE_FANOUT {
        sum = sum.wrapping_add(*wide_leaf(index).await?);
    }
    Ok(Vc::cell(sum))
}

/// A selector-gated root over the wide fanout, so flipping disconnects the whole `wide_parent`
/// subtree cleanly. `wide_parent` has enough children to be promoted to an aggregating node, so
/// disconnecting it exercises both GC discovery buffers: `wide_parent` loses its last persistent
/// parent (count-zeroed) and the aggregation-graph rebalance that frees the leaves runs during the
/// same cascade.
#[turbo_tasks::function(operation, root)]
async fn select_wide(selector: ResolvedVc<Selector>) -> Result<Vc<u32>> {
    let use_wide = !*selector.await?.get();
    let value = if use_wide {
        *wide_parent().connect().await?
    } else {
        0u32
    };
    Ok(Vc::cell(value))
}

/// Drives `select_wide` connected, then flips the selector to disconnect the whole `wide_parent`
/// subtree without invalidating it (so the leaves stay clean and simply lose their parent).
async fn build_and_disconnect_wide(tt: Arc<TurboTasks<TurboTasksBackend>>) {
    turbo_tasks::run_once(tt, async move {
        unmark_top_level_task_may_leak_eventually_consistent_state();
        let selector_op = create_selector(false);
        let selector_vc = selector_op.resolve().strongly_consistent().await?;
        let selector = selector_op.read_strongly_consistent().await?;

        let output = select_wide(selector_vc);
        output.read_strongly_consistent().await?;

        selector.set(true);
        output.read_strongly_consistent().await?;
        anyhow::Ok(())
    })
    .await
    .unwrap();
}

/// A GC pass collects a disconnected subtree via the parent_count cascade: disconnecting branch_a
/// drops its count to 0 (branch_a is collected), which decrements its child leaf(10) to 0 (leaf(10)
/// is then collected too). The live branch (branch_b + leaf(20)) is untouched and the graph still
/// computes; flipping back recomputes branch_a fresh, proving no dangling references.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn gc_collects_disconnected_subtree() {
    let (tt, _persistence_dir) = create_tt("gc_collects_disconnected_subtree");
    let tt2 = tt.clone();

    let result = turbo_tasks::run_once(tt.clone(), async move {
        unmark_top_level_task_may_leak_eventually_consistent_state();

        let selector_op = create_selector(false);
        let selector_vc = selector_op.resolve().strongly_consistent().await?;
        let selector = selector_op.read_strongly_consistent().await?;

        let output = select(selector_vc);
        assert_eq!(*output.read_strongly_consistent().await?, 11);

        // Flip: select drops branch_a; branch_a (parent_count 0) becomes a candidate.
        selector.set(true);
        assert_eq!(*output.read_strongly_consistent().await?, 22);

        anyhow::Ok(())
    })
    .await;
    result.unwrap();

    // GC runs in a fresh `run_once` after the first completes: a `run_once` root keeps every task
    // it touched active (active_counter > 0) until it returns, so a task disconnected *within*
    // that run is not yet collectible. Once the run ends the active counts are released and the
    // disconnected branch_a (parent_count 0) becomes collectible; the cascade then drops
    // leaf(10) to 0 too.
    let collected = tt2.backend().gc_for_testing(&tt2);
    assert_eq!(
        collected, 2,
        "branch_a and its cascaded child leaf(10) should both be collected"
    );
    assert_eq!(
        tt2.backend().gc_for_testing(&tt2),
        0,
        "a second GC pass must collect nothing"
    );

    // Flipping back must recompute branch_a fresh, since it was collected.
    let tt3 = tt.clone();
    let result = turbo_tasks::run_once(tt.clone(), async move {
        let selector_op = create_selector(true);
        let selector_vc = selector_op.resolve().strongly_consistent().await?;
        let selector = selector_op.read_strongly_consistent().await?;
        let output = select(selector_vc);
        assert_eq!(*output.read_strongly_consistent().await?, 22);
        selector.set(false);
        assert_eq!(*output.read_strongly_consistent().await?, 11);
        let _ = &tt3;
        anyhow::Ok(())
    })
    .await;
    result.unwrap();

    tt.stop_and_wait().await;
}

/// A task that pins itself via `prevent_gc()` must survive collection even after it is disconnected
/// from the live graph, because the pin makes it a GC root.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn gc_does_not_collect_pinned_task() {
    let (tt, _persistence_dir) = create_tt("gc_does_not_collect_pinned_task");
    let tt2 = tt.clone();

    let result = turbo_tasks::run_once(tt.clone(), async move {
        unmark_top_level_task_may_leak_eventually_consistent_state();

        let selector_op = create_selector(false);
        let selector_vc = selector_op.resolve().strongly_consistent().await?;
        let selector = selector_op.read_strongly_consistent().await?;

        let output = select_pinned(selector_vc);
        assert_eq!(*output.read_strongly_consistent().await?, 99);

        // Flip: select_pinned re-executes, reads branch_b, and disconnects pinned_branch.
        selector.set(true);
        assert_eq!(*output.read_strongly_consistent().await?, 22);

        anyhow::Ok(())
    })
    .await;
    result.unwrap();

    // GC runs after the run has released activeness, so pinned_branch is disconnected
    // (parent_count 0) and otherwise collectible.
    let collected = tt2.backend().gc_for_testing(&tt2);
    assert_eq!(
        collected, 0,
        "a pinned task must not be collected even when disconnected"
    );

    // A snapshot + evict must not lose the (transient) pin. A pinned task is not forced fully
    // resident — its Meta/Data may be partially evicted — but the session-only
    // `transient_ref_count` is retained as residue (the map entry is kept), so the task stays
    // uncollectible and a subsequent GC still collects nothing.
    tt2.backend().snapshot_and_evict_for_testing(&tt2);
    assert_eq!(
        tt2.backend().gc_for_testing(&tt2),
        0,
        "pinned task must survive eviction and not be collected"
    );

    tt.stop_and_wait().await;
}

/// A [`GcRoot`] guard pins a task for its lifetime and unpins it on drop. This is the anchor a
/// permanent root uses (e.g. the `ProjectContainer` operation held by a NAPI `ProjectInstance`):
/// while the guard lives the task is uncollectible even at `parent_count == 0`; dropping the guard
/// releases the pin so it becomes collectible. The guard owns exactly one pin and is not `Clone`,
/// so it can't double-unpin (which would underflow `transient_ref_count`).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn gc_root_guard_pins_until_dropped() {
    let (tt, _persistence_dir) = create_tt("gc_root_guard_pins_until_dropped");
    let tt2 = tt.clone();

    // A parentless leaf read inside a `run`, then guarded by a `GcRoot` created *outside* the run
    // (as `project_new` does for the container — `run` returns the task id; the guard is built
    // after). `run` (unlike `run_once`) creates no lingering transient root anchoring it, so
    // the guard is its only anchor.
    let leaf_id = tt
        .run(async move {
            unmark_top_level_task_may_leak_eventually_consistent_state();
            let leaf_vc = leaf(55);
            let _ = *leaf_vc.await?;
            Ok(task_id_of(leaf_vc.resolve().await?))
        })
        .await
        .unwrap();
    let guard = GcRoot::pin(tt2.clone(), leaf_id);

    assert_eq!(guard.task_id(), leaf_id);
    assert_eq!(
        tt.backend().transient_ref_count_for_testing(leaf_id),
        1,
        "the GcRoot guard pins the task (transient_ref_count 1)"
    );
    assert_eq!(
        tt.backend().gc_for_testing(&tt),
        0,
        "a task pinned by a live GcRoot must not be collected"
    );

    // Drop the guard: it unpins, so the now-unanchored parentless leaf becomes collectible.
    drop(guard);
    assert_eq!(
        tt.backend().transient_ref_count_for_testing(leaf_id),
        0,
        "dropping the GcRoot released the pin"
    );
    assert_eq!(
        tt.backend().gc_for_testing(&tt),
        1,
        "the task is collectible once its GcRoot is dropped"
    );

    tt.stop_and_wait().await;
}

/// A parentless task read at the top level of a `run_once` is no longer force-kept as a persisted
/// topology "root" (we removed that blanket flagging). It is instead kept alive only by a real
/// anchor: here, the never-disposed transient `Once` task of the `run_once` that read it keeps it
/// reachable/active, so it is not collected while that anchor exists. (This is the same "a
/// `run_once` root keeps everything it touched active until — and here, because it is never
/// disposed, beyond — it returns" behavior that `gc_re_rooting_stays_flat` accounts for by
/// measuring the *persistent* resident set. The leak fix for disposable ops is covered by
/// `gc_collects_disconnected_subtree` and `dispose_root_task_releases_anchored_subgraph`.)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn parentless_top_level_task_kept_by_transient_root_not_a_topology_flag() {
    let (tt, _persistence_dir) =
        create_tt("parentless_top_level_task_kept_by_transient_root_not_a_topology_flag");
    let tt2 = tt.clone();
    let tt3 = tt.clone();

    let root_id = turbo_tasks::run_once(tt.clone(), async move {
        unmark_top_level_task_may_leak_eventually_consistent_state();

        assert_eq!(*gc_root_leaf().await?, 77);

        let root_id = task_id_of(gc_root_leaf().resolve().await?);
        // No persistent parent connected it, so parent_count is 0 — it is not a persistent-graph
        // child of anything.
        assert_eq!(
            tt3.backend().parent_count_for_testing(root_id),
            0,
            "a parentless top-level task has no persistent parent edge"
        );

        anyhow::Ok(root_id)
    })
    .await
    .unwrap();

    // Its `run_once`'s `Once` task is never disposed and keeps it anchored, so GC does not collect
    // it — but note this is an in-session transient anchor, NOT the removed persisted topology
    // flag.
    assert_eq!(
        tt2.backend().parent_count_for_testing(root_id),
        0,
        "still parent_count 0 after the run"
    );
    let collected = tt2.backend().gc_for_testing(&tt2);
    assert_eq!(
        collected, 0,
        "kept alive by its (undisposed) transient Once root, not by a topology flag"
    );

    tt.stop_and_wait().await;
}

/// Disposing a root task (as `RootTask::Drop` / `root_task_dispose` does when JS stops listening to
/// a subscription) must release the anchor its child edges placed on the persistent tasks it read,
/// so the subscription's subgraph becomes collectible. A transient root task bumps each persistent
/// child's `transient_ref_count`; `dispose_root_task` tears the (clean) root's edges down via
/// `CleanupOldEdges`, which sheds that count (and rebalances the aggregation graph). Also checks
/// the contract `RootTask::Drop` relies on: disposal is idempotent and safe after the backend
/// stopped.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dispose_root_task_releases_anchored_subgraph() {
    let (tt, _persistence_dir) = create_tt("dispose_root_task_releases_anchored_subgraph");

    // Spawn a real root task (as `subscribe` does) whose body reads a persistent leaf, connecting
    // it as a child of the transient root — bumping the leaf's transient_ref_count. The root sends
    // the leaf's id out via a oneshot so we can inspect it WITHOUT a separate `run_once` probe (a
    // probe would be its own transient `Once` task that also connects the leaf — Once tasks are
    // never disposed, so it would pollute the leaf's transient_ref_count and never drop to 0).
    let (tx, rx) = tokio::sync::oneshot::channel();
    let tx = Arc::new(std::sync::Mutex::new(Some(tx)));
    let root_id = tt.spawn_root_task(move || {
        let tx = tx.lock().unwrap().take();
        Box::pin(async move {
            // The root body runs as a top-level task; unmark so the eventually-consistent leaf read
            // is allowed (as `subscribe`'s HMR handler does).
            unmark_top_level_task_may_leak_eventually_consistent_state();
            let leaf_vc = leaf(88);
            let value = *leaf_vc.await?;
            if let Some(tx) = tx {
                let _ = tx.send(task_id_of(leaf_vc.resolve().await?));
            }
            anyhow::Ok(Vc::<u32>::cell(value))
        })
    });

    // The root connects the leaf as its only anchor: no persistent parent (parent_count 0), one
    // transient child edge (transient_ref_count 1).
    let leaf_id = rx.await.unwrap();
    for _ in 0..100 {
        if tt.backend().transient_ref_count_for_testing(leaf_id) > 0 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert_eq!(
        tt.backend().parent_count_for_testing(leaf_id),
        0,
        "leaf read only by the root task has no persistent parent"
    );
    assert_eq!(
        tt.backend().transient_ref_count_for_testing(leaf_id),
        1,
        "the transient root task anchors the leaf via exactly one child edge"
    );

    // While the root task is live, the leaf is anchored and must not be collected.
    assert_eq!(
        tt.backend().gc_for_testing(&tt),
        0,
        "leaf anchored by the live root task must not be collected"
    );

    // Dispose the root task (as `RootTask::Drop` now does when JS skipped `root_task_dispose`). Its
    // `CleanupOldEdges` sheds the leaf's transient_ref_count.
    tt.dispose_root_task(root_id);
    // Idempotent: disposing again (the shape of explicit JS dispose + a later `Drop`) must not
    // panic and must not underflow the child's count (the edges were already removed).
    tt.dispose_root_task(root_id);

    assert_eq!(
        tt.backend().transient_ref_count_for_testing(leaf_id),
        0,
        "disposing the root task released its transient_ref_count anchor on the leaf"
    );

    // Collect in a fresh run (a `run_once` keeps touched tasks active until it returns, so GC runs
    // after it). The leaf is now unanchored, quiescent, and collectible.
    turbo_tasks::run_once(tt.clone(), async move {
        unmark_top_level_task_may_leak_eventually_consistent_state();
        anyhow::Ok(())
    })
    .await
    .unwrap();
    assert_eq!(
        tt.backend().gc_for_testing(&tt),
        1,
        "the leaf becomes collectible once the root task that anchored it is disposed"
    );

    // Disposal must be safe after the backend has stopped, as a `RootTask` finalized during Node
    // worker teardown would be (the whole task map is dropped by `stop`).
    tt.stop_and_wait().await;
    tt.dispose_root_task(root_id);
}

/// Unpinning a task after the backend has started stopping must not panic. This mirrors the real
/// shutdown ordering: a `DetachedVc` handed to JS across NAPI is finalized (dropped, which unpins)
/// during Node worker teardown, which can run *after* `stop()` has dropped the whole task map.
/// pin/unpin are gated on the `stopping` flag (set before the map is dropped) so a late unpin is a
/// no-op rather than resurrecting a blank entry and underflowing `transient_ref_count`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unpin_after_stop_does_not_panic() {
    let (tt, _persistence_dir) = create_tt("unpin_after_stop_does_not_panic");

    // Pin a real task inside a session (as `prevent_gc` / `DetachedVc::new` would).
    let tt2 = tt.clone();
    let leaf_id = turbo_tasks::run_once(tt.clone(), async move {
        unmark_top_level_task_may_leak_eventually_consistent_state();
        let id = task_id_of(leaf(7).resolve().await?);
        tt2.pin_task_for_gc(id);
        anyhow::Ok(id)
    })
    .await
    .unwrap();

    // Stop the backend — this drops the in-memory task map, so the pinned task is no longer
    // resident (exactly as at `next build` shutdown).
    tt.stop_and_wait().await;

    // Unpin after teardown, as a `DetachedVc`'s `Drop` finalized during Node worker cleanup would.
    // The task is gone from the map, so this must be a harmless no-op — not an underflow panic.
    tt.unpin_task_for_gc(leaf_id);
}
