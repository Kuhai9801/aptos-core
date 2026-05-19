// Copyright (c) Aptos Foundation
// Licensed pursuant to the Innovation-Enabling Source Code License, available at https://github.com/aptos-labs/aptos-core/blob/main/LICENSE

//! Interpreter-internal error types.

use mono_move_core::{ExecutionErrorKind, IntoExecutionError};
use mono_move_gas::GasExhaustedError;
use std::fmt;
use thiserror::Error;

pub type RuntimeResult<T> = Result<T, RuntimeError>;

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error(transparent)]
    GasExhausted(#[from] GasExhaustedError),

    // TODO: replace with a typed loader error once the loader has one.
    #[error(transparent)]
    Loader(anyhow::Error),

    #[error("{op}: overflow")]
    ArithmeticOverflow { op: ArithOp },

    #[error("{op}: underflow")]
    ArithmeticUnderflow { op: ArithOp },

    #[error("{op}: division by zero")]
    DivisionByZero { op: ArithOp },

    #[error("{op}: shift amount {amount} exceeds 63")]
    ShiftAmountOutOfRange { op: ArithOp, amount: u64 },

    #[error("VecPopBack on empty vector")]
    PopFromEmptyVector,

    #[error("{op} index out of bounds: idx={idx} len={len}")]
    VectorIndexOutOfBounds { op: VecOp, idx: u64, len: u64 },

    #[error("CallClosure: null closure pointer")]
    NullClosure,

    #[error("stack overflow")]
    StackOverflow,

    #[error("out of heap memory after GC (requested {requested} bytes)")]
    OutOfHeapMemory { requested: usize },

    #[error("heap_alloc: size {requested} exceeds maximum single allocation size")]
    AllocationTooLarge { requested: usize },

    #[error("alloc_vec: size overflow")]
    VecAllocSizeOverflow,

    #[error("CallClosure: callee has {num_params} params, exceeds 64-bit mask capacity")]
    TooManyClosureParams { num_params: usize },

    #[error("AbortMsg: message is not valid UTF-8")]
    InvalidAbortMessage,

    #[error("AbortMsg: message size {len} exceeds maximum {max}")]
    AbortMessageTooLong { len: usize, max: usize },

    #[error("invariant violation: {0}")]
    InvariantViolation(#[from] RuntimeInvariantViolation),
}

impl IntoExecutionError for RuntimeError {
    fn kind(&self) -> ExecutionErrorKind {
        use RuntimeError::*;
        match self {
            GasExhausted(_) => ExecutionErrorKind::OutOfGas,

            Loader(_) => ExecutionErrorKind::LinkError,

            ArithmeticOverflow { .. }
            | ArithmeticUnderflow { .. }
            | DivisionByZero { .. }
            | ShiftAmountOutOfRange { .. }
            | PopFromEmptyVector
            | VectorIndexOutOfBounds { .. }
            | NullClosure
            | InvalidAbortMessage => ExecutionErrorKind::InvalidOperation,

            StackOverflow
            | OutOfHeapMemory { .. }
            | AllocationTooLarge { .. }
            | VecAllocSizeOverflow
            | TooManyClosureParams { .. }
            | AbortMessageTooLong { .. } => ExecutionErrorKind::RuntimeLimitExceeded,

            InvariantViolation(_) => ExecutionErrorKind::InvariantViolation,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithOp {
    AddU64,
    AddU64Imm,
    SubU64,
    SubU64Imm,
    RSubU64Imm,
    MulU64,
    MulU64Imm,
    DivU64,
    ModU64,
    ShlU64,
    ShrU64,
}

impl fmt::Display for ArithOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArithOp::AddU64 => write!(f, "AddU64"),
            ArithOp::AddU64Imm => write!(f, "AddU64Imm"),
            ArithOp::SubU64 => write!(f, "SubU64"),
            ArithOp::SubU64Imm => write!(f, "SubU64Imm"),
            ArithOp::RSubU64Imm => write!(f, "RSubU64Imm"),
            ArithOp::MulU64 => write!(f, "MulU64"),
            ArithOp::MulU64Imm => write!(f, "MulU64Imm"),
            ArithOp::DivU64 => write!(f, "DivU64"),
            ArithOp::ModU64 => write!(f, "ModU64"),
            ArithOp::ShlU64 => write!(f, "ShlU64"),
            ArithOp::ShrU64 => write!(f, "ShrU64"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VecOp {
    LoadElem,
    StoreElem,
    Borrow,
}

impl fmt::Display for VecOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VecOp::LoadElem => write!(f, "VecLoadElem"),
            VecOp::StoreElem => write!(f, "VecStoreElem"),
            VecOp::Borrow => write!(f, "VecBorrow"),
        }
    }
}

/// Conditions that should never occur if the compiler, verifier, and
/// runtime maintain their invariants. Surfaced rather than panicked so
/// callers can produce a clean per-transaction outcome and alert
/// operationally on `ExecutionErrorKind::InvariantViolation`.
#[derive(Debug, Error)]
pub enum RuntimeInvariantViolation {
    #[error("pc out of bounds: pc={pc} but function {func_name} has {code_len} instructions")]
    PcOutOfBounds {
        pc: usize,
        func_name: String,
        code_len: usize,
    },

    #[error("heap_alloc: size {requested} overflows after alignment")]
    AllocSizeOverflow { requested: usize },

    #[error("alloc_obj called with non-allocatable descriptor {descriptor_id}")]
    NonAllocatableDescriptor { descriptor_id: u32 },

    #[error("GC scan: invalid object size {size} (expected non-zero, MAX_ALIGN-byte aligned)")]
    GcInvalidObjectSize { size: usize },

    #[error("GC found forwarding marker in to-space")]
    GcForwardingMarkerInToSpace,

    #[error("CallClosure: unsupported func_ref tag {tag} (only Resolved supported in v0)")]
    UnsupportedClosureFuncRef { tag: u8 },

    #[error("CallClosure: null function pointer in closure")]
    NullFuncRefInClosure,

    #[error(
        "CallClosure: unsupported captured-data tag {tag} (only Materialized supported in v0)"
    )]
    UnsupportedClosureCapturedData { tag: u8 },

    #[error("CallClosure: null captured_data for closure with captured params")]
    NullCapturedData,

    #[error("CallClosure: provided_args[{provided_idx}].size {provided_size} != callee param_sizes[{param_idx}] {param_size}")]
    ClosureArgSizeMismatch {
        provided_idx: usize,
        provided_size: u32,
        param_idx: usize,
        param_size: u32,
    },

    #[error("CallClosure: not enough provided args")]
    NotEnoughProvidedArgs,

    #[error(
        "CallClosure: {provided} provided_args but only {consumed} non-captured params consumed"
    )]
    ClosureArgsCountMismatch { provided: usize, consumed: usize },
}

/// Successful terminal outcomes from `Interpreter::run`. Runtime
/// failures flow through the `Err` channel as [`RuntimeError`] — abort
/// and failure are structurally separate.
#[derive(Debug)]
pub enum RuntimeStatus {
    Success,
    // TODO: carry the abort's `Location` (which module raised it) once
    // we have a `Location` type defined.
    Aborted {
        code: u64,
        message: Option<String>,
    },
}

/// Returns from the enclosing function with an [`RuntimeError::InvariantViolation`]
/// wrapping the named [`RuntimeInvariantViolation`] variant. Works for both
/// unit and struct variants:
///
/// ```ignore
/// invariant_violation!(NullFuncRefInClosure);
/// invariant_violation!(PcOutOfBounds { pc, func_name, code_len });
/// ```
#[macro_export]
macro_rules! invariant_violation {
    ($($body:tt)+) => {
        return ::core::result::Result::Err(
            $crate::error::RuntimeError::InvariantViolation(
                $crate::error::RuntimeInvariantViolation::$($body)+,
            ),
        )
    };
}
