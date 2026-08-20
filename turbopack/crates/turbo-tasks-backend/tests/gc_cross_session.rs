#![feature(arbitrary_self_types)]
#![allow(clippy::needless_return)] // tokio macro-generated code doesn't respect this

//! Cross-session collection of GC roots.
//!
//! A task read at the top level of a session has no persistent parent, so the resident-scan GC —
//! which only reclaims tasks that lost a parent — can never reclaim it. Those tasks are tracked as
//! durable *roots* in a persisted map instead, and a root that stops being requested has to age out
//! of that map for its subtree to be reclaimed. That is the only path by which a persisted root
//! ever becomes garbage, and it spans sessions by construction.
//!
//! These tests drive real restarts (stop the backend, reopen the same database) and assert only on
//! observable outcomes — what is still resident, and whether the graph still computes. The roots
//! map itself is an implementation detail deliberately left unasserted.

mod util;

use std::sync::atomic::{AtomicU32, Ordering};

use anyhow::Result;
use turbo_tasks::{
    GcRoot, ResolvedVc, State, Vc, unmark_top_level_task_may_leak_eventually_consistent_state,
};

use crate::util::{create_persistence_dir, reopen_tt};

/// Counts executions of [`orphan_leaf`], keyed by its argument. A collected task has to re-execute
/// when it is next requested, so a bump here is the observable signal that it was reclaimed —
/// whereas a task that merely got evicted restores from disk without executing.
static LEAF_EXECUTIONS: [AtomicU32; 3] = [AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0)];

fn leaf_executions(n: u32) -> u32 {
    LEAF_EXECUTIONS[n as usize].load(Ordering::Relaxed)
}

/// A leaf keyed by `n`, so each root gets a distinct child whose fate can be observed on its own.
#[turbo_tasks::function]
fn orphan_leaf(n: u32) -> Vc<u32> {
    LEAF_EXECUTIONS[n as usize].fetch_add(1, Ordering::Relaxed);
    Vc::cell(n)
}

/// A `(operation, root)` op reading `orphan_leaf(n)`: a two-task "root -> subtree". Read at the top
/// level of a session it has no persistent parent, so it is a durable GC root and its leaf gets
/// `parent_count 1`.
#[turbo_tasks::function(operation, root)]
async fn root_with_child(n: u32) -> Result<Vc<u32>> {
    Ok(Vc::cell(*orphan_leaf(n).await? + 1))
}

// --- Diamond fixture for the cross-session disk-only forward-dep scrub test ---
//
// A "diamond": a reader `A` (`diamond_reader`) holds a **forward cell-dependency** on a target `B`
// (`diamond_target`) that is *not* its child — `B` is called by the root and its resolved `Vc` is
// passed into `A`, so reading it records a dep edge without a child edge. Both `A` and `B` are
// children of the diamond root. This is the shape that, cross-session under GC, can require
// scrubbing a **disk-only** forward-dep target: when the orphaned root is collected, `A` and `B`
// cascade together, and `A`'s `CleanupOldEdges` may open `B` to remove the stale reverse edge while
// `B` has not yet been restored from disk this session.
//
// The target must be **mutable** to record dependents at all (an immutable constant records none),
// so it reads a long-lived `State` whose value never changes.

#[turbo_tasks::value(transparent)]
struct Constant(State<u32>);

#[turbo_tasks::function(operation, root)]
fn create_constant() -> Vc<Constant> {
    Constant(State::new(0)).cell()
}

/// The forward-dependency *target* `B`. Reading `constant`'s `State` makes it mutable so a reader
/// records a real `cell_dependents` reverse edge on it.
#[turbo_tasks::function]
async fn diamond_target(constant: ResolvedVc<Constant>, index: u32) -> Result<Vc<u32>> {
    let base = *constant.await?.get();
    Ok(Vc::cell(base.wrapping_add(index).wrapping_mul(7)))
}

/// The diamond *reader* `A`: reads the cell of a `diamond_target` (`B`) passed in as an
/// already-resolved `Vc`, so `A` forward-deps on `B` without parenting it.
#[turbo_tasks::function]
async fn diamond_reader(target: ResolvedVc<u32>) -> Result<Vc<u32>> {
    Ok(Vc::cell(1 + *target.await?))
}

const DIAMOND_FANOUT: u32 = 64;

/// The diamond root: for each index, calls `diamond_target(index)` (`B`, the root's child) and
/// `diamond_reader(B)` (`A`, the root's child that forward-deps on `B`). `FANOUT` distinct A/B
/// pairs give many chances for the racing interleaving where an `A` scrubs a not-yet-restored `B`.
#[turbo_tasks::function(operation, root)]
async fn diamond_root(constant: ResolvedVc<Constant>) -> Result<Vc<u32>> {
    let mut sum = 0u32;
    for index in 0..DIAMOND_FANOUT {
        let target = diamond_target(*constant, index).to_resolved().await?;
        sum = sum.wrapping_add(*target.await?);
        sum = sum.wrapping_add(*diamond_reader(*target).await?);
    }
    Ok(Vc::cell(sum))
}

/// The behaviour that justifies the whole persisted-roots mechanism: **a root is kept alive by
/// being used, and reclaimed by not being used** — across process restarts.
///
/// Two sibling roots are persisted in session 1. From then on only one of them is ever requested
/// again. The reused root and its subtree must survive every session; the abandoned one must
/// eventually be collected, taking its subtree with it. Neither outcome is reachable by the
/// resident-scan GC alone: both roots have `parent_count == 0` forever, so nothing about the
/// in-memory graph distinguishes them. Only the cross-session roots map does.
///
/// The TTL is forced to 0 so the age-out lands inside the test rather than days later. Demotion and
/// age-out are still two distinct steps — the pass that first notices a root is gone only starts
/// its clock — which is why each session runs two passes. With a production TTL the second step
/// would instead wait out the deadline, giving a root that was live last session a full stale
/// session of grace; that grace is the point of the design (a session that happens not to touch a
/// route must not immediately discard it) and forcing the TTL to 0 is what compresses it to one
/// session here.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reused_root_survives_sessions_that_abandon_its_sibling() {
    let dir = create_persistence_dir("reused_root_survives_sessions_that_abandon_its_sibling");

    // Session 1: build both subtrees and anchor each root with a pin, the way the embedder holds a
    // live handle to a route. The pin is what makes the task a *durable root* (`transient_ref_count
    // > 0`) and gets it recorded in the persisted roots map on shutdown.
    let kept_root = {
        let tt = reopen_tt(&dir);
        let ids = turbo_tasks::run_once(tt.clone(), async move {
            unmark_top_level_task_may_leak_eventually_consistent_state();
            let kept = root_with_child(1);
            let dropped = root_with_child(2);
            assert_eq!(*kept.read_strongly_consistent().await?, 2);
            assert_eq!(*dropped.read_strongly_consistent().await?, 3);
            anyhow::Ok((kept.task_id(), dropped.task_id()))
        })
        .await
        .unwrap();

        // Both are anchored in session 1, so both are persisted as roots.
        let kept_pin = GcRoot::pin(tt.clone(), ids.0);
        let dropped_pin = GcRoot::pin(tt.clone(), ids.1);
        tt.backend().snapshot_and_evict_for_testing(&tt);
        drop(kept_pin);
        drop(dropped_pin);

        tt.stop_and_wait().await;
        ids.0
    };

    // Sessions 2 and 3: only root 1 is ever requested again. Root 2 is never mentioned.
    //
    // Session 2's first pass demotes root 2 (starts its clock); a later pass ages it out. Session 3
    // is a second restart, proving the collection stuck and left the surviving root unharmed.
    let mut total_collected = 0usize;
    for session in 2..=3 {
        let tt = reopen_tt(&dir);
        tt.backend().set_gc_root_ttl_for_testing(0);
        let tt2 = tt.clone();
        turbo_tasks::run_once(tt.clone(), async move {
            unmark_top_level_task_may_leak_eventually_consistent_state();
            // Re-request root 1 only, and restore it so it has a resident entry to pin. Root 2 is
            // never mentioned in this session.
            assert_eq!(
                *root_with_child(1).read_strongly_consistent().await?,
                2,
                "the reused root must still compute in session {session}"
            );
            let _ = &tt2;
            anyhow::Ok(())
        })
        .await
        .unwrap();

        // Re-anchor only the reused root. Root 2 has no pin this session, so it is un-anchored:
        // the first pass demotes it, a later one ages it out.
        let kept_pin = GcRoot::pin(tt.clone(), kept_root);
        // Two passes: with the TTL at 0 the demote and the age-out can both land in this session,
        // but the demote always costs a pass of its own.
        total_collected += tt.backend().gc_for_testing(&tt);
        total_collected += tt.backend().gc_for_testing(&tt);
        drop(kept_pin);

        tt.stop_and_wait().await;
    }

    // The abandoned root and its leaf are two tasks; nothing else in this graph is collectible
    // (root 1 is re-anchored every session, and its leaf has a parent). Requiring at least the pair
    // is what distinguishes "the sibling was reclaimed" from "GC did nothing at all".
    assert!(
        total_collected >= 2,
        "the abandoned root and its leaf should have been collected across sessions 2-3 (got \
         {total_collected})"
    );

    // Session 4: the surviving root must still be cached, and the collected one must rebuild.
    {
        let tt = reopen_tt(&dir);
        let kept_before = leaf_executions(1);
        let dropped_before = leaf_executions(2);
        let result = turbo_tasks::run_once(tt.clone(), async move {
            unmark_top_level_task_may_leak_eventually_consistent_state();

            // Both must still produce the right value: the survivor from cache, the collected one
            // rebuilt from scratch. A dangling edge left by the cross-session collection — or a
            // resurrected half-deleted task — would surface as a wrong value or a panic here.
            assert_eq!(*root_with_child(1).read_strongly_consistent().await?, 2);
            assert_eq!(*root_with_child(2).read_strongly_consistent().await?, 3);

            // The reused root's subtree was never collected, so its leaf is served from the
            // persisted cache without executing again.
            assert_eq!(
                leaf_executions(1),
                kept_before,
                "the reused root's subtree must be served from cache, not recomputed"
            );
            // The abandoned root's subtree was collected, so requesting it again has to re-execute.
            assert_eq!(
                leaf_executions(2),
                dropped_before + 1,
                "the abandoned root's subtree must have been collected and rebuilt"
            );
            anyhow::Ok(())
        })
        .await;
        tt.stop_and_wait().await;
        result.unwrap();
    }
}

/// Regression for the cross-session cascade panic found by the gc-04 dogfood: collecting an
/// orphaned root whose subtree contains a **forward cell-dependency** to a target that is
/// **disk-only** (not restored this session) must scrub that stale reverse edge by restoring the
/// live target — not trip a "target not resident, would resurrect a collected task" guard.
///
/// Setup (see the diamond fixture): each `diamond_reader` `A` holds a forward cell-dep on a
/// `diamond_target` `B` that is *not* its child; both are children of `diamond_root`. In session 1
/// we build and persist the whole diamond. In session 2 we reopen (nothing re-requests the root, so
/// with TTL 0 it ages out) and run GC: the root is collected and its `A`/`B` children cascade
/// concurrently. When an `A`'s `CleanupOldEdges` opens its target `B` to remove the stale
/// `cell_dependents` edge, `B` may not yet have been restored from disk — the disk-only case the
/// old `gc_target_resident` debug_assert wrongly rejected. The pass completing without a panic is
/// the assertion; session 3 confirms a clean recompute (no dangling reverse edge survived).
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn gc_collect_scrubs_disk_only_forward_dep_target() {
    let dir = create_persistence_dir("gc_collect_scrubs_disk_only_forward_dep_target");

    // Session 1: build the diamond and persist it. The root is anchored with a pin and snapshotted
    // so it lands in the persisted roots map — that is what lets a *later* session notice it went
    // un-anchored and age it out. Without this the root is never a tracked root at all, so nothing
    // is ever collected and the cascade under test never runs.
    {
        let tt = reopen_tt(&dir);
        let root_id = turbo_tasks::run_once(tt.clone(), async move {
            unmark_top_level_task_may_leak_eventually_consistent_state();
            let constant_op = create_constant();
            let constant_vc = constant_op.resolve().strongly_consistent().await?;
            // Read the root so the whole diamond is built and persisted.
            let root_op = diamond_root(constant_vc);
            root_op.read_strongly_consistent().await?;
            anyhow::Ok(root_op.task_id())
        })
        .await
        .unwrap();

        let root_pin = GcRoot::pin(tt.clone(), root_id);
        // Record the root in the persisted roots map while it is still anchored. A GC pass is what
        // admits a live root to that map, so run one explicitly rather than relying on the
        // snapshot: the snapshot alone leaves nothing for session 2 to age out.
        tt.backend().gc_for_testing(&tt);
        tt.backend().snapshot_and_evict_for_testing(&tt);
        drop(root_pin);

        tt.stop_and_wait().await;
    }

    // Session 2: reopen, TTL 0, run GC. The diamond root is never re-requested, so it ages out;
    // collecting it cascades to every A/B pair. B targets are disk-only until the cascade restores
    // them — an A that scrubs its target before that must restore it, not panic.
    {
        let tt = reopen_tt(&dir);
        tt.backend().set_gc_root_ttl_for_testing(0);
        let tt2 = tt.clone();
        turbo_tasks::run_once(tt.clone(), async move {
            unmark_top_level_task_may_leak_eventually_consistent_state();
            // No panic in these passes is the assertion (the old guard fired inside
            // `CleanupOldEdges`). Demotion and age-out take a pass each: a root that was live last
            // session gets one stale pass of grace before the zero TTL lets it go.
            //
            // Age-out needs the elapsed time to be *strictly* greater than the TTL, and the clock
            // behind it (`gc.rs::now_ms`) has millisecond resolution — so even at TTL 0 the two
            // passes have to land in different milliseconds. Two back-to-back passes can share one,
            // which collects nothing; sleeping between attempts is what makes this deterministic.
            // The point of this test is the *cascade* — an `A` scrubbing a disk-only `B` — so it
            // needs the collect to actually happen.
            // How many passes the collect takes is deliberately not asserted. The root has to be
            // demoted to `FirstStale(t)` and then age out, and either step may already have
            // happened: session 1's snapshot ran a GC pass of its own, so the root can arrive here
            // already stale and be collected by the very first pass. When it arrives `MostRecent`
            // instead, only one pass per session may demote it
            // (`gc.rs::first_gc_pass_of_session`), and age-out then needs elapsed > TTL
            // *strictly* — with `gc.rs::now_ms` at millisecond resolution, a pass sharing a
            // millisecond with the demotion collects nothing. Sleeping between passes covers that;
            // asserting a fixed pass count would just encode whichever case this environment
            // happens to produce.
            let mut collected = 0usize;
            for _ in 0..20 {
                collected += tt2.backend().gc_for_testing(&tt2);
                if collected > DIAMOND_FANOUT as usize {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
            // The diamond is a root plus DIAMOND_FANOUT A/B pairs, so a real cascade collects far
            // more than a token task or two; requiring the whole shape distinguishes "the subtree
            // was reclaimed" from "GC nibbled at an edge of it".
            assert!(
                collected > DIAMOND_FANOUT as usize,
                "the orphaned diamond root subtree should be collected (got {collected})"
            );
            anyhow::Ok(())
        })
        .await
        .unwrap();
        tt.stop_and_wait().await;
    }

    // Session 3: a clean recompute must still work — no dangling reverse edge left on any target.
    {
        let tt = reopen_tt(&dir);
        let result = turbo_tasks::run_once(tt.clone(), async move {
            unmark_top_level_task_may_leak_eventually_consistent_state();
            let constant_op = create_constant();
            let constant_vc = constant_op.resolve().strongly_consistent().await?;
            diamond_root(constant_vc).read_strongly_consistent().await?;
            anyhow::Ok(())
        })
        .await;
        tt.stop_and_wait().await;
        result.unwrap();
    }
}
