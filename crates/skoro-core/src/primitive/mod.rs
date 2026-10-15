//! The core's primitive operations, one per row of the tables of primitives, each with a kernel
//! written in burn's tensor operations, the same code at both precisions.

mod euclidean;
mod rule;

pub use euclidean::{
    Add, Concatenate, Exp, Log, MatMul, Mul, Neg, Scale, Slice, Sub, Sum, Transpose,
};
