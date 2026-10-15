//! The core's primitive operations, one per row of the tables of primitives, each with a kernel
//! written in burn's tensor operations, the same code at both precisions.

mod euclidean;
mod positive;
mod rule;

pub use euclidean::{
    Add, Atan2, Concatenate, Cos, Exp, Log, MatMul, Mul, Neg, Scale, Sin, Slice, Sub, Sum,
    Transpose,
};
pub use positive::{PositiveAsReal, PositiveExp, PositiveLog, PositiveMul, PositiveRecip, Select};
