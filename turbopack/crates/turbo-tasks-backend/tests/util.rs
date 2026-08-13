//! Shared setup for tests that need a real on-disk persistent backend and direct access to
//! [`TurboTasksBackend`] test hooks (`snapshot_and_evict_for_testing`, `parent_count_for_testing`,
//! …).
//!
//! These tests can't use `turbo_tasks_testing::register!`: that harness hands back an
//! `Arc<dyn TurboTasksApi>`, which erases the concrete backend type and makes `tt.backend()`
//! unreachable. It also owns the session loop (re-running the body and comparing results), whereas
//! these tests need to drive snapshots — and sometimes a DB reopen — on their own schedule.
//!
//! This module is compiled into each test binary separately, so any helper a given binary doesn't
//! call reads as dead code there.
#![allow(dead_code)]

use std::{path::Path, sync::Arc};

use turbo_tasks::TurboTasks;
use turbo_tasks_backend::{
    BackendOptions, BackingStorageOptions, EvictionMode, GitVersionInfo, TurboTasksBackend,
};

/// Opens a backend rooted at `path`.
///
/// Reusing the same `path` (after the previous backend has been stopped) reopens the persisted
/// database, which is how a test can assert that state survives a restart.
fn open_tt_at(path: &Path, num_workers: usize) -> Arc<TurboTasks<TurboTasksBackend>> {
    TurboTasks::new(TurboTasksBackend::new(
        BackendOptions {
            num_workers: Some(num_workers),
            small_preallocation: true,
            // Avoid racing with the background snapshot loop; these tests drive
            // snapshot_and_evict_for_testing manually.
            storage_mode: Some(turbo_tasks_backend::StorageMode::ReadWriteOnShutdown),
            eviction_mode: EvictionMode::Full,
            ..Default::default()
        },
        turbo_tasks_backend::turbo_backing_storage(
            path,
            &GitVersionInfo {
                describe: "test-unversioned",
                dirty: false,
            },
            BackingStorageOptions {
                is_short_session: true,
                skip_compaction: true,
                ..Default::default()
            },
        )
        .unwrap()
        .0,
    ))
}

/// A persistence directory that outlives the backends opened on it, so a test can stop one backend
/// and open another on the same path to simulate a restart. Pair with [`reopen_tt`].
pub fn create_persistence_dir(name: &str) -> tempfile::TempDir {
    tempfile::Builder::new()
        .prefix(&format!("{name}-"))
        .tempdir()
        .unwrap()
}

/// Opens a backend on an existing persistence directory — a new *session* over the same database.
/// The previous backend must already be stopped (`stop_and_wait`) so its shutdown snapshot has been
/// flushed.
pub fn reopen_tt(dir: &tempfile::TempDir) -> Arc<TurboTasks<TurboTasksBackend>> {
    open_tt_at(dir.path(), 2)
}

/// A fresh persistent backend in its own temp directory, with `num_workers` workers.
pub fn create_tt_with_workers(
    name: &str,
    num_workers: usize,
) -> (Arc<TurboTasks<TurboTasksBackend>>, tempfile::TempDir) {
    let dir = create_persistence_dir(name);
    let tt = open_tt_at(dir.path(), num_workers);
    (tt, dir)
}

/// A fresh persistent backend in its own temp directory, with the default worker count.
pub fn create_tt(name: &str) -> (Arc<TurboTasks<TurboTasksBackend>>, tempfile::TempDir) {
    create_tt_with_workers(name, 2)
}
