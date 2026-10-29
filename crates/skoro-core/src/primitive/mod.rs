//! The core's primitive operations, one per row of the tables of primitives, each with a kernel
//! written in burn's tensor operations, the same code at both precisions.

mod discrete;
mod euclidean;
mod positive;
mod rule;

pub use crate::embedded::{Include, Project, Retract};
pub use discrete::{
    IntegerAdd, IntegerEq, IntegerLe, IntegerLt, IntegerMul, IntegerSub, IsNonnegative, ToReal,
};
pub use euclidean::{
    Add, Atan2, Concatenate, Cos, Exp, Log, MatMul, Mul, Neg, Scale, Sin, Slice, Sub, Sum,
    Transpose,
};
pub use positive::{PositiveAsReal, PositiveExp, PositiveLog, PositiveMul, PositiveRecip, Select};
