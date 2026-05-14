// Copyright (c) Aptos Foundation
// Licensed pursuant to the Innovation-Enabling Source Code License, available at https://github.com/aptos-labs/aptos-core/blob/main/LICENSE

//! Sharded RocksDB tier for native-position value storage.
//!
//! 16 shards keyed by `state_key.get_shard_id()` (the leading nibble
//! of the StateKey hash, matching `state_kv_db` and the JMT internal
//! shard convention). Column families: `position_value`,
//! `stale_position_value_index`, `db_metadata`. The per-key CFs
//! (`position_value`, `stale_position_value_index`) are partitioned
//! across shards; `db_metadata` (pruner-progress bookkeeping) is
//! only ever written to shard 0. All 16 RocksDB instances expose
//! the same CF list — shard 0 just happens to own the singleton.
//!
//! Lifecycle metadata (exchange-id allocations, deny-list) lives in
//! the `aptos_experimental::native_position::ExchangeRegistry` Move
//! resource at `@aptos_framework`, not here. There is no
//! `position_metadata` CF.
//!
//! Shard boundaries are chosen at every read/write site via
//! [`PositionDb::shard_of_state_key`] (constant-time, uses the
//! pre-computed `StateKey::get_shard_id`) on the commit path, or
//! [`PositionDb::shard_of_hash`] for ad-hoc consumers that already
//! hold a `state_key_hash`.
//!
//! See `PLAN_native_position.md` for design rationale.

#![forbid(unsafe_code)]

use crate::schema::{
    position_value::PositionValueSchema, DB_METADATA_CF_NAME, POSITION_VALUE_CF_NAME,
    STALE_POSITION_VALUE_INDEX_CF_NAME,
};
use aptos_crypto::HashValue;
use aptos_schemadb::{
    batch::SchemaBatch, ColumnFamilyName, DB, DEFAULT_COLUMN_FAMILY_NAME,
};
use aptos_storage_interface::{AptosDbError, Result};
use aptos_types::{
    state_store::{state_key::StateKey, state_value::StateValue, NUM_STATE_SHARDS},
    transaction::Version,
};
use std::{path::Path, sync::Arc};

/// Number of value-DB shards. Mirrors `aptos_types::state_store::NUM_STATE_SHARDS`.
pub const NUM_NATIVE_VALUE_SHARDS: usize = NUM_STATE_SHARDS;

/// Column families hosted by every `position_db` shard. The
/// previously-defined `position_metadata` CF is gone — exchange
/// registrations and the deny-list now live in the
/// `aptos_experimental::native_position::ExchangeRegistry` Move
/// resource at `@aptos_framework`. `db_metadata` is retained for
/// pruner-progress bookkeeping (lives in shard 0). `default` is
/// included so RocksDB's mandatory default CF picks up our standard
/// `gen_table_options` tuning instead of RocksDB defaults — matches
/// every other sub-DB's CF list.
pub const POSITION_DB_COLUMN_FAMILIES: [ColumnFamilyName; 4] = [
    /* empty cf */ DEFAULT_COLUMN_FAMILY_NAME,
    DB_METADATA_CF_NAME,
    POSITION_VALUE_CF_NAME,
    STALE_POSITION_VALUE_INDEX_CF_NAME,
];

/// Sharded handle for the position value tier. Holds 16 independent
/// RocksDB instances; the metadata CFs are read/written only via
/// shard 0 (see module docs).
#[derive(Debug)]
pub struct PositionDb {
    shards: [Arc<DB>; NUM_NATIVE_VALUE_SHARDS],
}

impl PositionDb {
    /// Construct a `PositionDb` from a fixed-size array of opened
    /// shard DBs. The caller is responsible for opening each shard at
    /// the path layout `<root>/shard_<i>/`.
    pub fn new(shards: [Arc<DB>; NUM_NATIVE_VALUE_SHARDS]) -> Self {
        Self { shards }
    }

    /// Convenience for tests / single-shard contexts: replicate the
    /// same DB across all 16 shard slots. NOT for production — the 16
    /// slots will all point at the same RocksDB instance, defeating
    /// the parallelism benefit, but it lets tests reuse the existing
    /// open helpers without a multi-instance dance.
    #[cfg(any(test, feature = "fuzzing"))]
    pub fn new_uniform_for_test(db: Arc<DB>) -> Self {
        let shards: [Arc<DB>; NUM_NATIVE_VALUE_SHARDS] = std::array::from_fn(|_| Arc::clone(&db));
        Self { shards }
    }

    pub fn name() -> &'static str {
        "position_db"
    }

    pub fn column_families() -> &'static [ColumnFamilyName] {
        &POSITION_DB_COLUMN_FAMILIES
    }

    /// Borrow the underlying RocksDB instance for shard `idx`.
    /// Used by the merkle-DB scaffold and tests; production callers
    /// should use the higher-level read/write methods.
    pub fn shard(&self, idx: usize) -> &Arc<DB> {
        &self.shards[idx]
    }

    /// All shards, useful for cross-shard fan-out (scan, prune).
    pub fn shards(&self) -> &[Arc<DB>; NUM_NATIVE_VALUE_SHARDS] {
        &self.shards
    }

    /// Shard chosen for `state_key`. Constant-time — uses the
    /// pre-computed hash already cached on the StateKey.
    pub fn shard_of_state_key(state_key: &StateKey) -> usize {
        state_key.get_shard_id()
    }

    /// Shard for a precomputed `state_key_hash`. Matches
    /// [`StateKey::get_shard_id`]: leading-nibble of the hash.
    pub fn shard_of_hash(state_key_hash: HashValue) -> usize {
        usize::from(state_key_hash.nibble(0))
    }

    /// Read the latest non-tombstone Position value at `<= version`,
    /// or `None` if absent or tombstoned. Routes to the matching shard.
    pub fn get_position_value(
        &self,
        state_key_hash: HashValue,
        version: Version,
    ) -> Result<Option<StateValue>> {
        let shard = Self::shard_of_hash(state_key_hash);
        let mut iter = self.shards[shard].iter::<PositionValueSchema>()?;
        iter.seek(&(state_key_hash, version))?;
        if let Some(Ok((key_pair, value_opt))) = iter.next() {
            if key_pair.0 == state_key_hash {
                return Ok(value_opt);
            }
        }
        Ok(None)
    }

    /// Append a batch of Position writes at `version`. Groups writes
    /// by their target shard, then writes each shard's batch
    /// independently.
    ///
    /// Stale-index emission is the caller's responsibility; the
    /// commit-path applier (`NativeStateCommitter`) emits stale-index
    /// entries inline alongside value writes via [`Self::shard`] +
    /// `write_schemas`. This helper is for ad-hoc consumers (state-
    /// sync apply-chunk, backup/restore) that don't need stale-index
    /// emission.
    pub fn write_position_batch(
        &self,
        version: Version,
        writes: impl IntoIterator<Item = (HashValue, Option<StateValue>)>,
    ) -> Result<()> {
        let mut per_shard: [Option<SchemaBatch>; NUM_NATIVE_VALUE_SHARDS] =
            std::array::from_fn(|_| None);
        for (state_key_hash, maybe_value) in writes {
            let shard = Self::shard_of_hash(state_key_hash);
            let batch = per_shard[shard].get_or_insert_with(SchemaBatch::new);
            batch.put::<PositionValueSchema>(&(state_key_hash, version), &maybe_value)?;
        }
        for (shard, maybe_batch) in per_shard.into_iter().enumerate() {
            if let Some(batch) = maybe_batch {
                self.shards[shard].write_schemas(batch)?;
            }
        }
        Ok(())
    }

    /// Most recent prior version `< at_version` at which
    /// `state_key_hash` was written. Used by the commit path to emit
    /// stale-index entries that drive the pruner. Routes to the
    /// matching shard.
    pub fn find_prior_version(
        &self,
        state_key_hash: HashValue,
        at_version: Version,
    ) -> Result<Option<Version>> {
        if at_version == 0 {
            return Ok(None);
        }
        let shard = Self::shard_of_hash(state_key_hash);
        let mut iter = self.shards[shard].iter::<PositionValueSchema>()?;
        iter.seek(&(state_key_hash, at_version - 1))?;
        if let Some(Ok(((row_hash, row_version), _value))) = iter.next() {
            if row_hash == state_key_hash {
                return Ok(Some(row_version));
            }
        }
        Ok(None)
    }
}

/// Open (or create) a sharded `position_db` rooted at `path`. Each
/// shard lives at `<path>/shard_<i>/`. Used by `AptosDB` wiring;
/// actual RocksDB options come from `db_options.rs`.
pub fn open_shards(path: &Path, opener: impl Fn(&Path, usize) -> Result<DB>) -> Result<PositionDb> {
    let shards: Vec<Arc<DB>> = (0..NUM_NATIVE_VALUE_SHARDS)
        .map(|i| {
            let shard_path = path.join(format!("shard_{i}"));
            opener(&shard_path, i).map(Arc::new)
        })
        .collect::<Result<Vec<_>>>()?;
    let shards: [Arc<DB>; NUM_NATIVE_VALUE_SHARDS] = shards
        .try_into()
        .map_err(|_| AptosDbError::Other("position_db shard array conversion failed".into()))?;
    Ok(PositionDb::new(shards))
}
