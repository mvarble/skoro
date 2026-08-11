//! The discrete `Q` operation library.
//!
//! These operators and methods let users build discrete sub-graphs (counts,
//! categories, booleans) entirely through the `Q` API, backed by built-in
//! [`DiscreteKernel`](skoro_diff::discrete::DiscreteKernel)s wrapped in
//! [`ForwardOnly`]. Discrete arithmetic and logic use operator sugar (`+`, `-`,
//! `*`, `&`, `|`, `!`); comparisons use named methods (since `==`/`<` are
//! reserved).
//!
//! There is no implicit coercion between discrete and continuous quantities;
//! crossing domains is always explicit via a bridge (see [`bridge`]).

pub mod bridge;
pub mod ops;

use skoro_core::error::SkoroResult;
use std::ops::{BitAnd, BitOr, Not};

use skoro_backend::Backend;
use skoro_core::discrete::{Boolean, Integer};
use skoro_graph::operation::ForwardOnly;

use crate::map::{register_map1, register_map2};
use crate::ops::{QAdd, QMul, QSub};
use crate::q::Q;

use ops::{BoolLogic, BoolNot, Comparison, IntAdd, IntCompare, IntMul, IntSub, Logic};

// ----- Discrete arithmetic: operators via the unified `Q*` traits (β) -----

impl<B: Backend> QAdd<B, Integer> for Integer {
    type OutputType = Integer;
    type OpType = ForwardOnly<IntAdd>;
    fn quantity_type(&self, _: &Integer) -> Integer {
        Integer
    }
    fn op_type(&self, _: &Integer) -> ForwardOnly<IntAdd> {
        ForwardOnly(IntAdd)
    }
    fn register(lhs: &Q<B, Integer>, rhs: &Q<B, Integer>) -> SkoroResult<Q<B, Integer>> {
        register_map2(Integer, ForwardOnly(IntAdd), (lhs, rhs))
    }
}

impl<B: Backend> QSub<B, Integer> for Integer {
    type OutputType = Integer;
    type OpType = ForwardOnly<IntSub>;
    fn quantity_type(&self, _: &Integer) -> Integer {
        Integer
    }
    fn op_type(&self, _: &Integer) -> ForwardOnly<IntSub> {
        ForwardOnly(IntSub)
    }
    fn register(lhs: &Q<B, Integer>, rhs: &Q<B, Integer>) -> SkoroResult<Q<B, Integer>> {
        register_map2(Integer, ForwardOnly(IntSub), (lhs, rhs))
    }
}

impl<B: Backend> QMul<B, Integer> for Integer {
    type OutputType = Integer;
    type OpType = ForwardOnly<IntMul>;
    fn quantity_type(&self, _: &Integer) -> Integer {
        Integer
    }
    fn op_type(&self, _: &Integer) -> ForwardOnly<IntMul> {
        ForwardOnly(IntMul)
    }
    fn register(lhs: &Q<B, Integer>, rhs: &Q<B, Integer>) -> SkoroResult<Q<B, Integer>> {
        register_map2(Integer, ForwardOnly(IntMul), (lhs, rhs))
    }
}

// ----- Comparisons: named methods producing a `Q<B, Boolean>` -----

/// Comparison operators on integer quantities, producing booleans.
pub trait DiscreteCompare<B: Backend> {
    fn eq(&self, other: &Q<B, Integer>) -> SkoroResult<Q<B, Boolean>>;
    fn ne(&self, other: &Q<B, Integer>) -> SkoroResult<Q<B, Boolean>>;
    fn lt(&self, other: &Q<B, Integer>) -> SkoroResult<Q<B, Boolean>>;
    fn le(&self, other: &Q<B, Integer>) -> SkoroResult<Q<B, Boolean>>;
    fn gt(&self, other: &Q<B, Integer>) -> SkoroResult<Q<B, Boolean>>;
    fn ge(&self, other: &Q<B, Integer>) -> SkoroResult<Q<B, Boolean>>;
}

impl<B: Backend> DiscreteCompare<B> for Q<B, Integer> {
    fn eq(&self, other: &Q<B, Integer>) -> SkoroResult<Q<B, Boolean>> {
        register_map2(
            Boolean,
            ForwardOnly(IntCompare(Comparison::Eq)),
            (self, other),
        )
    }
    fn ne(&self, other: &Q<B, Integer>) -> SkoroResult<Q<B, Boolean>> {
        register_map2(
            Boolean,
            ForwardOnly(IntCompare(Comparison::Ne)),
            (self, other),
        )
    }
    fn lt(&self, other: &Q<B, Integer>) -> SkoroResult<Q<B, Boolean>> {
        register_map2(
            Boolean,
            ForwardOnly(IntCompare(Comparison::Lt)),
            (self, other),
        )
    }
    fn le(&self, other: &Q<B, Integer>) -> SkoroResult<Q<B, Boolean>> {
        register_map2(
            Boolean,
            ForwardOnly(IntCompare(Comparison::Le)),
            (self, other),
        )
    }
    fn gt(&self, other: &Q<B, Integer>) -> SkoroResult<Q<B, Boolean>> {
        register_map2(
            Boolean,
            ForwardOnly(IntCompare(Comparison::Gt)),
            (self, other),
        )
    }
    fn ge(&self, other: &Q<B, Integer>) -> SkoroResult<Q<B, Boolean>> {
        register_map2(
            Boolean,
            ForwardOnly(IntCompare(Comparison::Ge)),
            (self, other),
        )
    }
}

// ----- Logical: operators on `Q<B, Boolean>` -----

impl<B: Backend> BitAnd for Q<B, Boolean> {
    type Output = SkoroResult<Q<B, Boolean>>;
    fn bitand(self, rhs: Q<B, Boolean>) -> SkoroResult<Q<B, Boolean>> {
        register_map2(Boolean, ForwardOnly(BoolLogic(Logic::And)), (&self, &rhs))
    }
}

impl<B: Backend> BitOr for Q<B, Boolean> {
    type Output = SkoroResult<Q<B, Boolean>>;
    fn bitor(self, rhs: Q<B, Boolean>) -> SkoroResult<Q<B, Boolean>> {
        register_map2(Boolean, ForwardOnly(BoolLogic(Logic::Or)), (&self, &rhs))
    }
}

impl<B: Backend> Not for Q<B, Boolean> {
    type Output = SkoroResult<Q<B, Boolean>>;
    fn not(self) -> SkoroResult<Q<B, Boolean>> {
        register_map1(Boolean, ForwardOnly(BoolNot), &self)
    }
}
