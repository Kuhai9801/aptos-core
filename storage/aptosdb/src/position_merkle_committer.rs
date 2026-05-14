// Copyright (c) Aptos Foundation
// Licensed pursuant to the Innovation-Enabling Source Code License, available at https://github.com/aptos-labs/aptos-core/blob/main/LICENSE

//! Drives `JellyfishMerkleTree` batch commits against
//! `position_merkle_db`. Given a set of `MerkleLeafUpdate`s (per-write
//! state_key_hash + state_key + value_hash) at a target version,
//! produces the new `position_root` hash and writes the associated
//! `TreeUpdateBatch` into `position_merkle_db`'s JMT column families.
//!
//! This is the missing bridge between `PositionCommitter` (which
//! collects updates into an `Vec<MerkleLeafUpdate>`) and the
//! on-disk position JMT. Callers invoke `apply_batch` right after
//! `PositionCommitter::apply` in the same commit scope.

#![forbid(unsafe_code)]

use crate::{
    native_state_committer::MerkleLeafUpdate,
    position_db::NUM_NATIVE_VALUE_SHARDS,
    position_merkle_db::PositionMerkleDb,
    schema::{
        jellyfish_merkle_node::JellyfishMerkleNodeSchema, stale_node_index::StaleNodeIndexSchema,
        stale_node_index_cross_epoch::StaleNodeIndexCrossEpochSchema,
    },
};
use aptos_crypto::HashValue;
use aptos_jellyfish_merkle::{
    node_type::{Node, NodeKey},
    JellyfishMerkleTree, StaleNodeIndex, TreeUpdateBatch,
};
use aptos_schemadb::batch::SchemaBatch;
use aptos_storage_interface::{AptosDbError, Result};
use aptos_types::{state_store::state_key::StateKey, transaction::Version};
use std::sync::Arc;

pub struct PositionMerkleCommitter {
    merkle_db: Arc<PositionMerkleDb>,
}

impl PositionMerkleCommitter {
    pub fn new(merkle_db: Arc<PositionMerkleDb>) -> Self {
        Self { merkle_db }
    }

    /// Apply a batch of `MerkleLeafUpdate`s at `version`. Returns the
    /// new position subtree root.
    ///
    /// Always writes a root node at `(empty_path, version)` — either
    /// the real JMT output for a non-empty batch, or a carry-forward
    /// copy of the prior version's root for an empty batch. This
    /// guarantees that every version has a queryable root, so a
    /// chain of empty blocks never returns the wrong hash on the
    /// next non-empty block.
    ///
    /// On non-empty blocks the per-shard `persisted_version` passed
    /// to `batch_put_value_set_for_shard` is extracted from the
    /// prior root's `Children` entries (each `Child.version` records
    /// when that shard subtree was last written). This lets a
    /// non-empty block find shard roots that were last written K
    /// blocks ago without needing to also copy them into every
    /// intervening carry-forward.
    ///
    /// `previous_epoch_ending_version` is the version at which the
    /// preceding epoch ended (looked up by the caller from ledger
    /// metadata). It controls how stale-node-index rows are split
    /// between the regular `StaleNodeIndexSchema` CF (intra-epoch
    /// turnover — fair game for the merkle pruner) and the
    /// `StaleNodeIndexCrossEpochSchema` CF (nodes whose
    /// creation-version sits in a prior epoch — must be kept around
    /// for that epoch's snapshot to remain reconstructable). Pass
    /// `None` before any epoch has ended (e.g. genesis) and on tests
    /// that don't care about epoch-snapshot pruning. Matches
    /// `state_merkle_db::write_merkle_tree_update_batch_to_shard`.
    pub fn apply_batch(
        &self,
        version: Version,
        previous_epoch_ending_version: Option<Version>,
        updates: &[MerkleLeafUpdate],
    ) -> Result<HashValue> {
        // Read the prior empty_path root once; non-empty path needs
        // its children for per-shard persisted versions, and the
        // carry-forward path needs the node itself.
        let (prior_root_node, prior_root_existed) = match version.checked_sub(1) {
            Some(prev) => match self.merkle_db.get_root_node_option(prev)? {
                Some(node) => (node, true),
                None => (Node::Null, false),
            },
            None => (Node::Null, false),
        };

        // Invariant guard. The commit path never produces `Node::Leaf`
        // at `empty_path` — `MIN_LEAF_DEPTH = 1` in the underlying JMT
        // pushes single leaves down to the shard root and wraps them
        // in an `Internal{leaf_count:1}` at the top. If we ever see a
        // bare `Leaf` here, the per-shard-version extraction below
        // would silently rebuild the tree from scratch and lose
        // existing data. Fail loudly instead.
        assert!(
            matches!(prior_root_node, Node::Null | Node::Internal(_)),
            "position_merkle_db prior root at version {} is a bare Leaf — \
             violates MIN_LEAF_DEPTH=1 invariant",
            version.saturating_sub(1),
        );

        if updates.is_empty() {
            return self.carry_forward(
                version,
                previous_epoch_ending_version,
                prior_root_node,
                prior_root_existed,
            );
        }

        // Per-shard `persisted_version`s: a shard's subtree was last
        // written at `child.version`, which may be older than
        // `version - 1` if intervening blocks were carry-forwards.
        // Empty entries → `None` (the JMT builds the subtree from
        // scratch for that shard).
        let per_shard_persisted: [Option<Version>; NUM_NATIVE_VALUE_SHARDS] = {
            let mut arr = [None; NUM_NATIVE_VALUE_SHARDS];
            if let Node::Internal(internal) = &prior_root_node {
                for (nibble, child) in internal.children_sorted() {
                    arr[u8::from(*nibble) as usize] = Some(child.version);
                }
            }
            arr
        };

        // Build the value set: HashValue -> Option<&(HashValue, StateKey)>
        // where None encodes a tombstone leaf. The JMT api is strict
        // about reference lifetimes; own the StateKey + hash tuples.
        let mut leaf_store: Vec<(HashValue, Option<(HashValue, StateKey)>)> =
            Vec::with_capacity(updates.len());
        for u in updates {
            match u.value_hash {
                Some(v) => leaf_store.push((u.state_key_hash, Some((v, u.state_key.clone())))),
                None => leaf_store.push((u.state_key_hash, None)),
            }
        }
        // Re-borrow for the put_value_set API signature.
        let value_set_refs: Vec<(HashValue, Option<&(HashValue, StateKey)>)> =
            leaf_store.iter().map(|(h, v)| (*h, v.as_ref())).collect();

        let tree = JellyfishMerkleTree::new(self.merkle_db.as_ref());
        let mut tree_update_batch = TreeUpdateBatch::new();
        let mut shard_root_nodes = Vec::with_capacity(NUM_NATIVE_VALUE_SHARDS);

        for shard_id in 0..NUM_NATIVE_VALUE_SHARDS as u8 {
            let shard_value_set: Vec<_> = value_set_refs
                .iter()
                .filter(|(k, _)| k.nibble(0) == shard_id)
                .cloned()
                .collect();
            let (shard_root_node, shard_batch) = tree
                .batch_put_value_set_for_shard(
                    shard_id,
                    shard_value_set,
                    None,
                    per_shard_persisted[shard_id as usize],
                    version,
                )
                .map_err(|e| {
                    AptosDbError::Other(format!(
                        "position JMT batch_put_value_set_for_shard({shard_id}) failed: {e}"
                    ))
                })?;
            tree_update_batch.combine(shard_batch);
            shard_root_nodes.push(shard_root_node);
        }

        // `put_top_levels_nodes` marks the prior empty_path root
        // stale via its `persisted_version`. Pass V-1 only when we
        // actually read a node there, so genesis (and unreachable
        // prior-empty cases) don't emit a stale entry for a key
        // that was never written.
        let top_persisted = if prior_root_existed {
            version.checked_sub(1)
        } else {
            None
        };
        let (root_hash, _leaf_count, top_levels_batch) = tree
            .put_top_levels_nodes(shard_root_nodes, top_persisted, version)
            .map_err(|e| {
                AptosDbError::Other(format!("position JMT put_top_levels_nodes failed: {e}"))
            })?;
        tree_update_batch.combine(top_levels_batch);

        // Commit the TreeUpdateBatch to position_merkle_db.
        self.commit_tree_batch(tree_update_batch, previous_epoch_ending_version)?;
        Ok(root_hash)
    }

    /// Empty-batch path: write the prior empty_path root verbatim at
    /// `(empty_path, version)` and mark the prior key stale.
    ///
    /// The new node references the same shard children at their
    /// original versions — no shard-root copying. The next non-empty
    /// block recovers per-shard versions by reading this carry-
    /// forward root's children (see `apply_batch`).
    ///
    /// For genesis with no updates we still write `Node::Null` so
    /// future versions can resolve `version - 1` without a missing-
    /// node error.
    fn carry_forward(
        &self,
        version: Version,
        previous_epoch_ending_version: Option<Version>,
        prior_root_node: Node<StateKey>,
        prior_root_existed: bool,
    ) -> Result<HashValue> {
        let prior_root_hash = prior_root_node.hash();

        let mut write = SchemaBatch::new();
        write
            .put::<JellyfishMerkleNodeSchema>(
                &NodeKey::new_empty_path(version),
                &prior_root_node,
            )
            .map_err(|e| {
                AptosDbError::Other(format!("position carry-forward node put failed: {e}"))
            })?;

        if prior_root_existed {
            // `prior_root_existed` implies `version > 0`.
            let prev = version - 1;
            let stale_row = StaleNodeIndex {
                stale_since_version: version,
                node_key: NodeKey::new_empty_path(prev),
            };
            if previous_epoch_ending_version.is_some_and(|prev_epoch| prev <= prev_epoch) {
                write
                    .put::<StaleNodeIndexCrossEpochSchema>(&stale_row, &())
                    .map_err(|e| {
                        AptosDbError::Other(format!(
                            "carry-forward cross-epoch stale put failed: {e}"
                        ))
                    })?;
            } else {
                write
                    .put::<StaleNodeIndexSchema>(&stale_row, &())
                    .map_err(|e| {
                        AptosDbError::Other(format!("carry-forward stale put failed: {e}"))
                    })?;
            }
        }

        self.merkle_db
            .inner()
            .write_schemas(write)
            .map_err(|e| {
                AptosDbError::Other(format!("position_merkle_db carry-forward commit failed: {e}"))
            })?;

        Ok(prior_root_hash)
    }

    fn commit_tree_batch(
        &self,
        batch: TreeUpdateBatch<StateKey>,
        previous_epoch_ending_version: Option<Version>,
    ) -> Result<()> {
        let mut write = SchemaBatch::new();
        for (node_key, node) in batch.node_batch.iter().flatten() {
            write
                .put::<JellyfishMerkleNodeSchema>(node_key, node)
                .map_err(|e| AptosDbError::Other(format!("JMT node put failed: {e}")))?;
        }
        for stale in batch.stale_node_index_batch.iter().flatten() {
            let row = StaleNodeIndex {
                stale_since_version: stale.stale_since_version,
                node_key: stale.node_key.clone(),
            };
            // Stale entries whose node was created at or before the
            // previous epoch boundary go to the cross-epoch CF — the
            // epoch-snapshot keeper needs them to remain queryable
            // independent of the regular pruner. Mirrors
            // `state_merkle_db::write_merkle_tree_update_batch_to_shard`.
            if previous_epoch_ending_version
                .is_some_and(|prev| row.node_key.version() <= prev)
            {
                write
                    .put::<StaleNodeIndexCrossEpochSchema>(&row, &())
                    .map_err(|e| {
                        AptosDbError::Other(format!("JMT cross-epoch stale index put failed: {e}"))
                    })?;
            } else {
                write
                    .put::<StaleNodeIndexSchema>(&row, &())
                    .map_err(|e| AptosDbError::Other(format!("JMT stale index put failed: {e}")))?;
            }
        }
        self.merkle_db
            .inner()
            .write_schemas(write)
            .map_err(|e| AptosDbError::Other(format!("position_merkle_db commit failed: {e}")))?;
        Ok(())
    }
}
