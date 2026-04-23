# Global Storage

This document describes how MonoMove manages per-transaction access to global state.
Specifically, it covers:

- how transactions record their reads,
- how transactions record their pending writes,
- how transaction make their writes visible to other transactions and bound their memory footprint (in Block-STM),
- how transactions checkpoint execution and roll-back to previously saved state (e.g., save prologue state, roll back on epilogue failure).

## TODOs

- Zero-copy reads
- Natives accessing storage (tables, aggregators)
- Gas
- Long-living data cache
- WriteOps and compatibility with current system
- Resource groups
- Memory reclamation in Block-STM (dedicated worker?)
- Per-key internal versioning of resource writes for checkpoints.
- Checkpoints in interpreter to save frames, gas, PC?

## Requirements

- Execution of every transaction records the reads it performs against the global state.
  The read-set is needed for:
  1) Block-STM validation,
  2) local caching so repeated reads do not re-enter the shared concurrent data structure,
  3) charging gas on first load only (TBDif it has to be BCS-based).

- Execution of every transaction records its pending writes as a write-set.
  The write-set can be made visible to Block-STM so that other transactions can observe it.
  Write-set is charged gas for IO (TBD if it has to be BCS-based).

- Within a single transaction, the VM has an ability to checkpoint the write-set.
  At later points, the VM can choose to continue or roll back to a prior checkpoint.
  History is linear: discarded states during roll back cannot be recovered.
  For example, if current checkpoints are states S1 - S2 - S3, rolling back to S1 discards S2 and S3.
  This approach can be used to handle prologue, user and epilogue sessions efficiently in Aptos VM.
  (TBD if checkpointing is needed for other VM structures, e.g., to handle try/catch).

- Reads should be zero-copy (sharing a pointer).
  Hence, VM must allow pointers into another transaction's memory region.

- Values in caches keep the flat-memory representation.


## Context

Under Block-STM execution, same transaction can be executed multiple times.
Every such execution is called **incarnation**.
Re-executing a transaction produces a new incarnation; incarnations are not reused.

Every transaction incarnation has its own **execution context**.
Execution context owns a bump-allocated memory region VM can use.
Initially, memory region is a small allocation (~1-2Mb) but can grow as needed (up to 10 MB).
During transaction execution, the region is used by the interpreter as a (**heap**).
Temporaries such as vectors, enums or large structs are allocated on the heap.
If there is no space to allocate new data, GC runs to compactify the region and allocate more memory if available.


## Per-Transaction Working Map

To keep track of global reads and writes, execution context maintains a **working map**.
It maps resource keys (e.g., `Foo<u8>`) to pointers onto heap (or other transaction's memory region, as discussed later).
Both reads and pending writes land in this map.
Each entry carries:

- a "state" tag identifying how the transaction has interacted with the key (e.g. existence check, full borrow, local copy for mutation),
- a pointer to the value data wherever it currently lives,
- additional validation metadata (e.g., version in storage) used by Block-STM to check the read later.

At any point in time, working map is part of the root set for garbage collection.

### State tags

The state tag captures how a transaction has interacted with a key so far:

- **Existence-only.**
  The transaction asked whether the resource exists but never used the value otherwise.
  Block-STM validation checks the existence bit, not the value / version.
- **Read-only observation.**
  The transaction read the value and did not modify it.
- **Copied-on-write.**
  The transaction borrowed the resource mutably.
  The pointer resolves to a copy in the incarnation's memory region, which is the authoritative version for further access from this transaction.
- **Created.**
  The transaction produced a new resource via a `move_to` operation.
  The pointer resolves to a fresh allocation in the memory region.
- **Removed.**
  The transaction destroyed the resource via a `move_from` operation.
  The data pointer is null.


## Per-Transaction Journal

A transaction incarnation can take a checkpoint at a boundary (prologue, epilogue, etc.), continue, and later decide to commit past the checkpoint or roll back to it.
A single linear undo log (**journal**) records changes to the working map used to roll back to a prior checkpoint.
Each checkpoint records two things: the journal's length at the time of the checkpoint, and the bump allocator's pointer.
Rollback walks the journal back to the saved length, applies each undo record to the working map, and resets the bump pointer to the saved value.

### Alternatives & Complexity

Instead of the journal, a per-key stack of versions can be used.
Each working map entry holds a stack of versions tagged by checkpoint epoch.
Rollback pops the affected stacks.
This approach is more efficient if rollbacks are common, but is not required for initial version.
It is also more efficient because rollback is lazy per used key: if key is never used, there is no work to be done.

With the journal approach:

- Rollback cost is proportional to the number of writes since the top checkpoint.
  Rollback of a nested checkpoint pops the top entry of the checkpoint stack and walks the journal back to its saved length.
- Memory reclamation on rollback is a single pointer assignment (resetting the bump pointer).
- Reads are not logged; the read-set survives every rollback because **every** read is needed for Block-STM validation.

### Example

Starting at a checkpoint called C1, the map holds two entries:

    R1:  Read-only observation,  pointing into the block cache
    R2:  Copied-on-write,        pointing at "a" location in the heap

The journal is empty.
C1 records the current bump mark.

The transaction calls borrow-global-mut on R1.
R1's data lives in the block cache, not in a post-checkpoint block, so the invariant forces a fresh copy into a new heap allocation "b".
The map entry for R1 transitions to "copied-on-write, pointing at b".
The journal records one undo entry: "restore R1 to read-only observation pointing into the block cache."

The transaction writes a field of R1 through the borrow.
This writes bytes inside heap-allocated region "b".
It does not change the map entry and does not append to the journal.

The transaction executes `move_to` on a new resource R3, allocating it at "c".
The map gains the entry "R3: created, pointing at c".
The journal gains one more undo entry: "R3 was not present; on rollback, remove it."

Rollback to C1 applies undo entries in reverse order:

1. Remove R3 from the map.
2. Restore R1 to "read-only observation pointing into the block cache."

Then reset the bump pointer to C1's saved mark.
Blocks "b" and "c" are now some heap garbage (TBD if these have to be zeroed out).
Block "a" is untouched, so R2's map entry remains valid.
The map is identical to its state at C1.

### Why reads are not recorded in undo log

For Block-STM validation to be correct, the read-set must be conservative.
Every key the transaction observed must remain in the read-set, even if the observation happened on a path that was later rolled back.
Otherwise, if the rolled-back path read X and X later changes, validation misses the dependency and the transaction commits on an observation it never validated against.

Reads are therefore always persisted into the working map when they happen and stay through every rollback.
(TBD on gas semantics if these reads are charged or not).
This is also a nice performance benefit: the journal's size is bounded by the number of writes since the top checkpoint, not by the number of reads.

### The invariant that enables bump rewind

Rewinding the bump pointer to the saved mark reclaims exactly the post-checkpoint allocations if and only if pre- and post-checkpoint allocations sit on opposite sides of the mark.
The design enforces this with one rule: the first write to a given resource key after the topmost checkpoint allocates a new bump block (a fresh copy-on-write), and further writes to the same key within the same checkpoint epoch mutate in place.
Pre-checkpoint blocks are therefore never mutated after the checkpoint.
Rewinding the bump pointer discards exactly the post-checkpoint blocks and preserves every pre-checkpoint block that a restored map entry may point to.

### Driver-orchestrated sub-transactions

The primary use case is a driver (typically AptosVM) orchestrating prologue, user, and epilogue phases around a single transaction.
The driver explicitly invokes checkpoint, commit, and rollback operations at session boundaries.
Move bytecode does not raise a rollback on abort; an abort propagates out to the driver, which decides whether to roll back.

Exception-handling semantics — catching aborts in Move and automatically rolling back to the nearest enclosing checkpoint — are a possible future extension.
They require no change to this layer, only a caller that invokes rollback on specific abort conditions.

### Garbage collection during a checkpoint

GC can run between a checkpoint being taken and the matching commit or rollback.
When it does, the saved bump mark for that checkpoint becomes unusable in two ways:

1. The mark is an address in from-space, which the collector has freed.
2. Even a post-collection address does not preserve the pre/post-checkpoint partition.
   Before collection, pre- and post-checkpoint allocations sat in two contiguous regions separated by the mark.
   After collection compacts surviving objects into to-space, both groups of live objects are interleaved in one contiguous run.
   No address distinguishes them.

#### What survives

The working map and the journal are in the collector's root set, so their pointers are updated in place.
Lookups and journal replay continue to work.
Block-cache pointers are outside the per-transaction arena and are not moved.
Rollback's state-restoration step (walking the journal) is unaffected.

#### What breaks

Rollback's memory-reclamation step (the bump rewind) does not work for any checkpoint whose mark predates a heap-full collection.

#### Direction

Accept the degradation.
At heap-full collection time, any live checkpoint's mark is flagged stale.
A rollback with a stale mark skips the bump rewind and performs only the journal replay.
Allocations that the rollback makes unreachable remain in to-space as garbage and are reclaimed by the next collection or by publication's compaction pass.

#### Alternatives

Forbid heap-full collection while a checkpoint is live.
Keeps the bump mark always valid but disables collection during the user session (often the longest session).
An allocation-heavy user session with collection disabled can hit OOM instead of collecting, which is a worse failure mode than transient garbage.

#### Consequences

- Correctness is preserved.
- Transient garbage may linger across one collection cycle in the compound case of "heap-full collection during a session that is later rolled back."
- Heap-full collection is rare (arenas are sized for the common case), rollback is rare (abort path), and the compound case is the product of two rare events.







## Publishing Writes to Block-STM

Once transaction incarnation finishes its execution, it does the following:

1. Compactifies transaction's heap so that only storage reads/writes remain.
   Compaction uses existing GC to prune dead data.

2. For every key the new incarnation wrote, overwrite the transaction's entry at that key with a pointer into the new region.
   For every key the previous incarnation wrote but this one does not, remove the transaction's entry at that key.
   These are the same operations Block-STM currently uses.


## Reading Global State

The interpreter reads global state through an interface that abstracts over local and cross-transaction pointers.
The storage layer returns a **resource view** to the interpreter, not a raw pointer.
The view abstracts over two possible internal representations:

- a pointer into the incarnation's own memory, or
- a pointer into another incarnation's region accompanied by a lifetime witness (TBD).

The first implementation always materializes reads into the incarnation's own memory.
A read copies the resource value into the working map's entry for the key.

To support global storage accesses, new instructions are added:

```
enum MicroOp {
    Exists {
        ty: InternedType,
        ty_args: InternedTypeList, // may be empty
    },
    ..
}
```

When storage instruction is executed, interpreter:
1.  checks if ty exists in working map, if so, writes a reference.
2.  if not - fetch

For moves

1.  set ptr to working map to data.
2.  null ptr in the working map.

