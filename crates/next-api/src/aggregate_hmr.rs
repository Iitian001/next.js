use std::collections::BTreeSet;

use anyhow::Result;
use rustc_hash::FxHashSet;
use turbo_rcstr::RcStr;
use turbo_tasks::{
    FxIndexMap, FxIndexSet, NonLocalValue, ReadRef, ResolvedVc, TraitRef, TryJoinIterExt, Vc,
    debug::ValueDebugFormat, trace::TraceRawVcs,
};
use turbo_tasks_fs::FileSystemPath;
use turbo_tasks_hash::{Xxh3Hash64Hasher, encode_base64};
use turbopack_browser::ecmascript::list::content::EcmascriptDevChunkListContent;
use turbopack_core::{
    update_instruction::UpdateInstruction,
    version::{PartialUpdate, Update, Version, VersionState, VersionedContent},
};
use turbopack_ecmascript::chunk_list::{
    merged_update::EcmascriptMergedUpdate,
    update::{ChunkListUpdate, ChunkUpdate, EcmascriptUpdateInstruction},
};
use turbopack_nodejs::ecmascript::node::entry::chunk_list_content::EcmascriptBuildNodeChunkListContent;

use crate::versioned_content_map::VersionedContentMap;

#[derive(TraceRawVcs, PartialEq, Eq, ValueDebugFormat, NonLocalValue)]
pub struct HmrChunkWithContent {
    pub path: RcStr,
    pub content: ResolvedVc<Box<dyn VersionedContent>>,
}

#[turbo_tasks::value(transparent, serialization = "skip")]
pub struct HmrChunksWithContent(Vec<HmrChunkWithContent>);

/// Whether `content` is a chunk list, i.e. an entry point of the chunk graph that
/// an HMR subscription can be anchored on.
///
/// Note this must enumerate every chunk list content type. A new chunking context
/// that introduces one has to be added here, otherwise its chunks silently drop
/// out of the HMR subscription.
pub fn is_entry_chunk_list_content(content: ResolvedVc<Box<dyn VersionedContent>>) -> bool {
    ResolvedVc::try_downcast_type::<EcmascriptBuildNodeChunkListContent>(content).is_some()
        || ResolvedVc::try_downcast_type::<EcmascriptDevChunkListContent>(content).is_some()
}

/// Per-chunk versions keyed by path
#[turbo_tasks::value(serialization = "skip", shared)]
pub struct AggregateHmrVersion {
    #[turbo_tasks(trace_ignore)]
    pub versions: FxIndexMap<RcStr, TraitRef<Box<dyn Version>>>,
}

#[turbo_tasks::value_impl]
impl Version for AggregateHmrVersion {
    #[turbo_tasks::function]
    async fn id(&self) -> Result<Vc<RcStr>> {
        let entries = self
            .versions
            .iter()
            .map(|(path, version)| {
                let path = path.clone();
                let version = TraitRef::cell(version.clone());
                async move {
                    let id = version.id().owned().await?;
                    anyhow::Ok((path, id))
                }
            })
            .try_join()
            .await?;

        let mut hasher = Xxh3Hash64Hasher::new();
        hasher.write_value(entries.len());
        for (path, id) in entries {
            hasher.write_value(path.as_str());
            hasher.write_value(id.as_str());
        }
        Ok(Vc::cell(encode_base64(hasher.finish()).into()))
    }
}

impl AggregateHmrVersion {
    pub async fn from_map(
        map: Vc<VersionedContentMap>,
        root: FileSystemPath,
    ) -> Result<Vc<Box<dyn Version>>> {
        // An empty `versions` map behaves the same as `NotFoundVersion` would in
        // `diff_chunks_against`, so no special case is needed here.
        let chunks = map.hmr_chunks_in_path(root).await?;
        Ok(Vc::upcast(Self::from_chunks(&chunks).await?))
    }

    pub async fn from_chunks(chunks: &[HmrChunkWithContent]) -> Result<Vc<Self>> {
        let versions = chunks
            .iter()
            .map(|HmrChunkWithContent { path, content }| {
                let path = path.clone();
                let content = *content;
                async move {
                    let version = content.version().into_trait_ref().await?;
                    anyhow::Ok((path, version))
                }
            })
            .try_join()
            .await?
            .into_iter()
            .collect();
        Ok(Self { versions }.cell())
    }
}

/// Aggregates per-entry HMR instructions into a single combined `ChunkListUpdate`.
#[derive(Default)]
pub struct ChunkListUpdateBuilder {
    chunks: FxIndexMap<RcStr, ChunkUpdate>,
    merged: FxIndexSet<EcmascriptMergedUpdate>,
    affected_entries: BTreeSet<RcStr>,
}

impl ChunkListUpdateBuilder {
    /// Adds an entry's instruction and records that entry when the instruction
    /// contains runtime work. Recording happens before merged instructions are
    /// deduplicated so a shared update retains every owning entry.
    pub fn add_entry_instruction(&mut self, path: &str, instruction: &UpdateInstruction) {
        if Self::instruction_has_changes(instruction) {
            self.affected_entries.insert(path.into());
        }
        self.add_instruction(instruction);
    }

    /// Records an entry whose chunk list disappeared from the aggregate.
    pub fn add_affected_entry(&mut self, path: &str) {
        self.affected_entries.insert(path.into());
    }

    /// Whether an update instruction carries runtime work. Seed/version-only
    /// shapes (empty chunk maps, merged updates without entries or chunks)
    /// carry no code and must not mark the entry. Unknown instruction types
    /// conservatively count as changes: a future protocol addition degrades
    /// to a spurious scope reload (safe) instead of a silently dropped update.
    fn instruction_has_changes(instruction: &UpdateInstruction) -> bool {
        let Some(instruction) = instruction.downcast_ref::<EcmascriptUpdateInstruction>() else {
            return true;
        };
        match instruction {
            EcmascriptUpdateInstruction::ChunkList(update) => {
                !update.chunks.is_empty()
                    || update.merged.iter().any(Self::merged_has_changes)
            }
            EcmascriptUpdateInstruction::Merged(update) => Self::merged_has_changes(update),
        }
    }

    fn merged_has_changes(update: &EcmascriptMergedUpdate) -> bool {
        !update.entries.is_empty() || !update.chunks.is_empty()
    }

    pub fn add_instruction(&mut self, instruction: &UpdateInstruction) {
        let instruction = instruction
            .downcast_ref::<EcmascriptUpdateInstruction>()
            .expect("aggregate HMR only accepts ECMAScript update instructions");

        match instruction {
            EcmascriptUpdateInstruction::ChunkList(update) => {
                for (chunk_path, update) in &update.chunks {
                    self.chunks.insert(chunk_path.clone(), update.clone());
                }
                for update in &update.merged {
                    self.push_merged(update);
                }
            }
            EcmascriptUpdateInstruction::Merged(update) => self.push_merged(update),
        }
    }

    fn push_merged(&mut self, update: &EcmascriptMergedUpdate) {
        self.merged.insert(update.clone());
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty() && self.merged.is_empty() && self.affected_entries.is_empty()
    }

    pub fn build(self, to: TraitRef<Box<dyn Version>>) -> Update {
        Update::Partial(PartialUpdate {
            to,
            instruction: ChunkListUpdate {
                chunks: self.chunks,
                merged: self.merged.into_iter().collect(),
                // BTreeSet iteration is sorted, so affected entries are
                // deterministic across polls.
                affected_entries: self.affected_entries.into_iter().collect(),
            }
            .into_instruction(),
        })
    }
}

/// Per-chunk [`Update`]s computed against an `AggregateHmrVersion` snapshot.
/// `has_new_chunks` is true when the current snapshot contains chunks absent
/// from `from` (e.g. a new endpoint was written); callers decide whether that
/// affects the batch shape.
pub struct DiffResult {
    pub chunk_updates: Vec<(RcStr, ReadRef<Update>)>,
    pub has_new_chunks: bool,
    pub removed_entries: Vec<RcStr>,
}

fn find_removed_entries<'a, 'b>(
    previous_paths: impl IntoIterator<Item = &'a RcStr>,
    current_paths: impl IntoIterator<Item = &'b RcStr>,
) -> Vec<RcStr> {
    let current_paths = current_paths
        .into_iter()
        .map(|path| path.as_str())
        .collect::<FxHashSet<_>>();
    let mut removed_entries = previous_paths
        .into_iter()
        .filter(|path| !current_paths.contains(path.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    removed_entries.sort_unstable_by(|a, b| a.as_str().cmp(b.as_str()));
    removed_entries
}

/// Diffs each chunk against the [`AggregateHmrVersion`] held by `from`.
///
/// If `from` holds some other kind of `Version`, there's nothing meaningful to
/// diff against, so this returns no updates and leaves it to the caller to
/// decide what to do.
pub async fn diff_chunks_against(
    chunks: &[HmrChunkWithContent],
    from: Vc<VersionState>,
) -> Result<DiffResult> {
    let from_resolved = from.get().to_resolved().await?;
    let Some(from_aggregate) = ResolvedVc::try_downcast_type::<AggregateHmrVersion>(from_resolved)
    else {
        return Ok(DiffResult {
            chunk_updates: Vec::new(),
            has_new_chunks: false,
            removed_entries: Vec::new(),
        });
    };
    let from_aggregate = from_aggregate.await?;
    let removed_entries = find_removed_entries(
        from_aggregate.versions.keys(),
        chunks.iter().map(|chunk| &chunk.path),
    );

    let mut has_new_chunks = false;
    let chunk_updates = chunks
        .iter()
        .filter_map(|HmrChunkWithContent { path, content }| {
            let Some(prev) = from_aggregate.versions.get(path).cloned() else {
                has_new_chunks = true;
                return None;
            };
            Some((path.clone(), *content, TraitRef::cell(prev)))
        })
        .map(async |(path, content, prev)| {
            let update = content.update(prev).await?;
            anyhow::Ok((path, update))
        })
        .try_join()
        .await?;
    Ok(DiffResult {
        chunk_updates,
        has_new_chunks,
        removed_entries,
    })
}

#[cfg(test)]
mod tests {
    use turbo_tasks::{FxIndexMap, FxIndexSet, ResolvedVc, TraitRef};
    use turbo_tasks_backend::{BackendOptions, TurboTasksBackend, noop_backing_storage};
    use turbopack_core::{
        update_instruction::UpdateInstruction,
        version::{Update, Version},
    };
    use turbopack_ecmascript::chunk_list::{
        merged_update::{
            EcmascriptMergedChunkDeleted, EcmascriptMergedChunkUpdate, EcmascriptMergedUpdate,
        },
        update::{ChunkListUpdate, ChunkUpdate, EcmascriptUpdateInstruction},
    };

    use super::{AggregateHmrVersion, ChunkListUpdateBuilder, find_removed_entries};

    fn merged(chunk_path: &str) -> EcmascriptMergedUpdate {
        EcmascriptMergedUpdate {
            entries: Default::default(),
            chunks: [(
                chunk_path.into(),
                EcmascriptMergedChunkUpdate::Deleted(EcmascriptMergedChunkDeleted {
                    modules: Default::default(),
                }),
            )]
            .into_iter()
            .collect(),
        }
    }

    #[test]
    fn deduplicates_merged_updates_in_first_seen_order() {
        let first = merged("first.js");
        let second = merged("second.js");
        let mut builder = ChunkListUpdateBuilder::default();

        builder.add_instruction(&UpdateInstruction::new(
            EcmascriptUpdateInstruction::Merged(first.clone()),
        ));
        builder.add_instruction(&UpdateInstruction::new(
            EcmascriptUpdateInstruction::Merged(second.clone()),
        ));
        builder.add_instruction(&UpdateInstruction::new(
            EcmascriptUpdateInstruction::Merged(first.clone()),
        ));

        assert_eq!(builder.merged, FxIndexSet::from_iter([first, second]));
    }

    #[test]
    fn chunk_updates_use_last_writer_and_stable_order() {
        let mut builder = ChunkListUpdateBuilder::default();
        let first = ChunkListUpdate {
            chunks: FxIndexMap::from_iter([
                ("a.js".into(), ChunkUpdate::Total),
                ("b.js".into(), ChunkUpdate::Added),
            ]),
            merged: vec![],
            affected_entries: Default::default(),
        };
        let second = ChunkListUpdate {
            chunks: FxIndexMap::from_iter([
                ("a.js".into(), ChunkUpdate::Deleted),
                ("c.js".into(), ChunkUpdate::Total),
            ]),
            merged: vec![],
            affected_entries: Default::default(),
        };

        builder.add_instruction(&first.into_instruction());
        builder.add_instruction(&second.into_instruction());

        assert_eq!(
            builder
                .chunks
                .keys()
                .map(|path| path.as_str())
                .collect::<Vec<_>>(),
            ["a.js", "b.js", "c.js"]
        );
        assert_eq!(builder.chunks["a.js"], ChunkUpdate::Deleted);
    }

    #[test]
    fn tracks_all_entries_before_deduplicating_shared_updates() {
        let shared = merged("shared.js");
        let instruction =
            UpdateInstruction::new(EcmascriptUpdateInstruction::Merged(shared.clone()));
        let mut builder = ChunkListUpdateBuilder::default();

        builder.add_entry_instruction("z/route.js", &instruction);
        builder.add_entry_instruction("a/route.js", &instruction);

        assert_eq!(
            builder
                .affected_entries
                .iter()
                .map(|path| path.as_str())
                .collect::<Vec<_>>(),
            ["a/route.js", "z/route.js"]
        );
        assert_eq!(builder.merged, FxIndexSet::from_iter([shared]));
    }

    #[test]
    fn ignores_seed_and_version_only_instructions() {
        let mut builder = ChunkListUpdateBuilder::default();
        builder.add_entry_instruction(
            "one/route.js",
            &ChunkListUpdate {
                chunks: FxIndexMap::default(),
                merged: vec![EcmascriptMergedUpdate {
                    entries: Default::default(),
                    chunks: Default::default(),
                }],
                affected_entries: Default::default(),
            }
            .into_instruction(),
        );

        assert!(builder.affected_entries.is_empty());
    }

    #[test]
    fn reports_removed_entries_deterministically_including_the_final_entry() {
        let previous = [RcStr::from("z/route.js"), RcStr::from("a/route.js")];
        let current = [RcStr::from("a/route.js")];
        assert_eq!(
            find_removed_entries(previous.iter(), current.iter()),
            [RcStr::from("z/route.js")]
        );

        let no_current_entries: [RcStr; 0] = [];
        assert_eq!(
            find_removed_entries(previous.iter(), no_current_entries.iter()),
            [RcStr::from("a/route.js"), RcStr::from("z/route.js")]
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn affected_entry_only_update_advances_the_version() {
        let tt = turbo_tasks::TurboTasks::new(TurboTasksBackend::new(
            BackendOptions::default(),
            noop_backing_storage(),
        ));
        tt.run_once(async {
            let mut builder = ChunkListUpdateBuilder::default();
            builder.add_affected_entry("app/removed/route.js");

            let to = ResolvedVc::upcast::<Box<dyn Version>>(
                AggregateHmrVersion {
                    versions: Default::default(),
                }
                .resolved_cell(),
            )
            .into_trait_ref()
            .await?;
            let Update::Partial(update) = builder.build(to.clone()) else {
                panic!("an affected-entry-only update must be partial");
            };

            assert!(TraitRef::ptr_eq(&update.to, &to));
            assert_eq!(
                serde_json::to_value(&update.instruction)?,
                serde_json::json!({
                    "type": "ChunkListUpdate",
                    "affectedEntries": ["app/removed/route.js"]
                })
            );

            Ok(())
        })
        .await
        .unwrap();
    }
}
