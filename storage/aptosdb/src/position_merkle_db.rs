// Copyright (c) Aptos Foundation
// Licensed pursuant to the Innovation-Enabling Source Code License, available at https://github.com/aptos-labs/aptos-core/blob/main/LICENSE

//! Dedicated JMT for native-position keys. Only
//! `TradingNativeKey::Position` entries become leaves; the per-user
//! markets index is intrinsic to `UserState.positions.keys()` and
//! has no merkle footprint.
//!
//! Reuses the existing `state_merkle_db` schemas
//! (`jellyfish_merkle_node`, `stale_node_index`,
//! `stale_node_index_cross_epoch`) inside this separate DB.
//! Batch-commit lives in [`crate::position_merkle_committer`];
//! composite-root composition into `state_root_hash` lives in
//! [`compose_state_root`] / [`state_root_hash_at_version`].
//! `AptosDB::init_native_position` opens the underlying RocksDB.

#![forbid(unsafe_code)]

use crate::schema::{
    jellyfish_merkle_node::JellyfishMerkleNodeSchema, DB_METADATA_CF_NAME,
    JELLYFISH_MERKLE_NODE_CF_NAME, STALE_NODE_INDEX_CF_NAME,
    STALE_NODE_INDEX_CROSS_EPOCH_CF_NAME,
};
use aptos_crypto::{hash::SPARSE_MERKLE_PLACEHOLDER_HASH, HashValue};
use aptos_jellyfish_merkle::{
    iterator::JellyfishMerkleIterator,
    node_type::{LeafNode, Node, NodeKey},
    JellyfishMerkleTree, TreeReader,
};
use aptos_schemadb::{ColumnFamilyName, DB, DEFAULT_COLUMN_FAMILY_NAME};
use aptos_storage_interface::{AptosDbError, Result};
use aptos_types::{state_store::state_key::StateKey, transaction::Version};
use std::sync::Arc;

/// Column families hosted by `position_merkle_db`. Mirrors
/// `state_merkle_db_column_families` shape: `default` so RocksDB's
/// mandatory default CF picks up our tuning, `db_metadata` for
/// JMT commit-progress bookkeeping, and the three JMT-node CFs.
pub const POSITION_MERKLE_DB_COLUMN_FAMILIES: [ColumnFamilyName; 5] = [
    /* empty cf */ DEFAULT_COLUMN_FAMILY_NAME,
    DB_METADATA_CF_NAME,
    JELLYFISH_MERKLE_NODE_CF_NAME,
    STALE_NODE_INDEX_CF_NAME,
    STALE_NODE_INDEX_CROSS_EPOCH_CF_NAME,
];

/// Thin wrapper around the position merkle RocksDB instance.
#[derive(Debug)]
pub struct PositionMerkleDb {
    db: Arc<DB>,
}

impl PositionMerkleDb {
    pub fn new(db: Arc<DB>) -> Self {
        Self { db }
    }

    pub fn name() -> &'static str {
        "position_merkle_db"
    }

    pub fn column_families() -> &'static [ColumnFamilyName] {
        &POSITION_MERKLE_DB_COLUMN_FAMILIES
    }

    pub fn inner(&self) -> &Arc<DB> {
        &self.db
    }

    /// Root hash of the position subtree at `version`. Returns the
    /// empty-tree placeholder if no position JMT nodes have been
    /// persisted yet at that version.
    pub fn get_root_hash(&self, version: Version) -> Result<HashValue> {
        let tree = JellyfishMerkleTree::new(self);
        match tree.get_root_hash_option(version) {
            Ok(Some(h)) => Ok(h),
            Ok(None) => Ok(*SPARSE_MERKLE_PLACEHOLDER_HASH),
            Err(e) => Err(AptosDbError::Other(format!(
                "position_merkle_db get_root_hash: {e}"
            ))),
        }
    }

    /// Raw root node at `version`, or `None` if nothing has been
    /// persisted at that version's empty-path key. Used by the
    /// commit path to extract per-shard persisted versions from the
    /// prior root's children (so non-empty blocks following one or
    /// more carry-forward blocks still find their shard subtrees).
    pub fn get_root_node_option(&self, version: Version) -> Result<Option<Node<StateKey>>> {
        self.get_node_option(&NodeKey::new_empty_path(version), "get_root")
    }

    /// Iterate every Position leaf live at `version`, yielding the
    /// original [`StateKey`] alongside the leaf's `value_hash`.
    ///
    /// Cold-load + backup use this to enumerate the position set
    /// (the value-CF is hash-keyed, so the JMT is the reverse index).
    /// Pair with [`crate::position_db::PositionDb::get_position_value`]
    /// keyed on `state_key.hash()` to fetch the matching `StateValue`.
    ///
    /// Iteration order is JMT-leaf order — depth-first over the tree,
    /// which equals lexicographic order on `state_key.hash()`.
    pub fn iter_active_leaves(
        self: &Arc<Self>,
        version: Version,
    ) -> Result<impl Iterator<Item = Result<(StateKey, HashValue)>>> {
        let iter = JellyfishMerkleIterator::<Self, StateKey>::new(
            Arc::clone(self),
            version,
            HashValue::zero(),
        )?;
        Ok(iter.map(|res| res.map(|(key_hash, (state_key, _leaf_version))| (state_key, key_hash))))
    }
}

impl TreeReader<StateKey> for PositionMerkleDb {
    fn get_node_option(&self, node_key: &NodeKey, _tag: &str) -> Result<Option<Node<StateKey>>> {
        self.db
            .get::<JellyfishMerkleNodeSchema>(node_key)
            .map_err(|e| AptosDbError::Other(format!("position_merkle_db get_node_option: {e}")))
    }

    fn get_rightmost_leaf(
        &self,
        version: Version,
    ) -> Result<Option<(NodeKey, LeafNode<StateKey>)>> {
        // Single-DB (unsharded) variant. The naive impl: walk all
        // JMT rows for the target version, track the leaf with the
        // lexicographically largest account_key. Matches the
        // semantics of state_merkle_db's `get_rightmost_leaf_naive`
        // — "rightmost leaf written at exactly this version", which
        // is what state-restore-through-JMT actually needs at chunk
        // boundaries.
        let mut iter = self
            .db
            .iter::<JellyfishMerkleNodeSchema>()
            .map_err(|e| AptosDbError::Other(format!("rightmost_leaf iter: {e}")))?;
        // Seek past version's root to the first node-by-nibble
        // entry; same starting point as `get_rightmost_leaf_naive`.
        iter.seek(&(version, 1u8))
            .map_err(|e| AptosDbError::Other(format!("rightmost_leaf seek: {e}")))?;
        let mut rightmost: Option<(NodeKey, LeafNode<StateKey>)> = None;
        for row in iter {
            let (node_key, node) =
                row.map_err(|e| AptosDbError::Other(format!("rightmost_leaf row: {e}")))?;
            if node_key.version() != version {
                break;
            }
            if let Node::Leaf(leaf_node) = node {
                match &rightmost {
                    None => rightmost = Some((node_key, leaf_node)),
                    Some(other) => {
                        if leaf_node.account_key() > other.1.account_key() {
                            rightmost = Some((node_key, leaf_node));
                        }
                    },
                }
            }
        }
        Ok(rightmost)
    }
}

/// Domain-separated hash tag for the composite state root formula:
///
/// ```text
/// state_root_hash = H("APTOS::StateRoot" || main_state_root || position_root)
/// ```
pub const COMPOSITE_STATE_ROOT_DOMAIN: &[u8] = b"APTOS::StateRoot";

/// The empty-JMT root used before any Position has been written.
/// Must agree with [`PositionMerkleDb::get_root_hash`]'s empty-tree
/// fallback so a node with no `position_merkle_db` attached and a
/// node with one attached but empty produce the same composite root.
pub fn empty_position_root() -> HashValue {
    *SPARSE_MERKLE_PLACEHOLDER_HASH
}

/// Compose two subtree roots into the block-level `state_root_hash`.
pub fn compose_state_root(main_state_root: HashValue, position_root: HashValue) -> HashValue {
    let mut hasher = aptos_crypto::hash::DefaultHasher::new(COMPOSITE_STATE_ROOT_DOMAIN);
    hasher.update(main_state_root.as_ref());
    hasher.update(position_root.as_ref());
    hasher.finish()
}

/// Select the correct `state_root_hash` at a given `Version` given
/// the `NATIVE_POSITION` activation version. Callers that produce
/// `TransactionInfo` values use this to choose between legacy
/// (just `main_state_root`) and composite form.
///
/// - If `version < activation_version`: return `main_state_root` as-is.
/// - Otherwise: return `compose_state_root(main_state_root, position_root)`.
///
/// Propagating this choice into `TransactionInfo` and every proof
/// verifier is out of scope for the initial drop; this helper gives
/// downstream code one function to call once those consumers are
/// updated.
pub fn state_root_hash_at_version(
    version: aptos_types::transaction::Version,
    activation_version: Option<aptos_types::transaction::Version>,
    main_state_root: HashValue,
    position_root: HashValue,
) -> HashValue {
    match activation_version {
        Some(act) if version >= act => compose_state_root(main_state_root, position_root),
        _ => main_state_root,
    }
}

/// Verify a composite state root against its components.
pub fn verify_composite_state_root(
    expected: HashValue,
    main_state_root: HashValue,
    position_root: HashValue,
) -> bool {
    compose_state_root(main_state_root, position_root) == expected
}
