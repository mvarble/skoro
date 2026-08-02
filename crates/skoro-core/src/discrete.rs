//! Concrete discrete (integer-valued, non-differentiable) quantity types.

use crate::pack::{Cardinality, NumComponents, Packable};

/// An unbounded integer quantity (e.g. a count), with no finite cardinality.
///
/// Packs into a single-component integer tensor. This is the natural codomain of
/// discrete arithmetic (sums and products of counts).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Integer;

impl NumComponents for Integer {
    fn num_components(&self) -> usize {
        1
    }
}

macro_rules! impl_integer_packable {
    ($ty:ty) => {
        impl Packable<$ty, $ty> for Integer {
            fn pack(&self, data: &$ty, offset: usize, dst: &mut [$ty]) {
                dst[offset] = *data;
            }

            fn unpack(&self, tensor: &[$ty], offset: usize) -> $ty {
                tensor[offset]
            }
        }
    };
}

impl_integer_packable!(i8);
impl_integer_packable!(i16);
impl_integer_packable!(i32);
impl_integer_packable!(i64);

impl Cardinality for Integer {
    fn cardinality(&self) -> Option<usize> {
        None
    }
}

/// A categorical quantity: a single index in `0..cardinality`.
///
/// Packs into a 1-component integer tensor holding the category index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Categorical(pub usize);

impl NumComponents for Categorical {
    fn num_components(&self) -> usize {
        1
    }
}

macro_rules! impl_categorical_packable {
    ($ty:ty) => {
        impl Packable<$ty, usize> for Categorical {
            fn pack(&self, data: &usize, offset: usize, dst: &mut [$ty]) {
                dst[offset] = *data as $ty;
            }

            fn unpack(&self, tensor: &[$ty], offset: usize) -> usize {
                tensor[offset] as usize
            }
        }
    };
}

impl_categorical_packable!(i8);
impl_categorical_packable!(i16);
impl_categorical_packable!(i32);
impl_categorical_packable!(i64);

impl Cardinality for Categorical {
    fn cardinality(&self) -> Option<usize> {
        Some(self.0)
    }
}

/// A boolean quantity (the two-category case of [`Categorical`]).
///
/// Packs into a 1-component integer tensor holding `0` or `1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Boolean;

impl NumComponents for Boolean {
    fn num_components(&self) -> usize {
        1
    }
}

macro_rules! impl_boolean_packable {
    ($ty:ty) => {
        impl Packable<$ty, bool> for Boolean {
            fn pack(&self, data: &bool, offset: usize, dst: &mut [$ty]) {
                dst[offset] = if *data { 1 } else { 0 };
            }

            fn unpack(&self, tensor: &[$ty], offset: usize) -> bool {
                tensor[offset] != 0
            }
        }
    };
}

impl_boolean_packable!(i8);
impl_boolean_packable!(i16);
impl_boolean_packable!(i32);
impl_boolean_packable!(i64);

impl Cardinality for Boolean {
    fn cardinality(&self) -> Option<usize> {
        Some(2)
    }
}
