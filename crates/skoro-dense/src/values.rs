//! Keys, and a vector split into one column per key.

use std::collections::BTreeMap;
use std::fmt::Debug;
use std::hash::Hash;

use skoro_core::error::{SkoroError, SkoroResult};

use crate::dense::Dense;

/// What a factor graph can name its variables by.
///
/// This crate never sees a graph `Id`; a caller outside it (the bridge from a model) uses `Id` as
/// the key, and the tests here use plain integers. `Ord` fixes a deterministic key order, which is
/// what every positional index in an error refers to.
pub trait Key: Copy + Eq + Hash + Ord + Debug {}

impl<T: Copy + Eq + Hash + Ord + Debug> Key for T {}

/// One column per key: a step `δ`, a gradient `g`, or a right-hand side.
///
/// Each value is an `n × 1` [`Dense`], `n` being the key's dimension. Keys iterate in `Ord` order.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorValues<K: Key>(BTreeMap<K, Dense>);

impl<K: Key> Default for VectorValues<K> {
    fn default() -> Self {
        Self(BTreeMap::new())
    }
}

impl<K: Key> VectorValues<K> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set `key`'s column, replacing any earlier one. Errors [`SkoroError::ShapeMismatch`] if
    /// `value` is not a single column.
    pub fn insert(&mut self, key: K, value: Dense) -> SkoroResult<()> {
        if value.cols() != 1 {
            return Err(SkoroError::ShapeMismatch {
                expected: 1,
                found: value.cols(),
            });
        }
        self.0.insert(key, value);
        Ok(())
    }

    pub fn get(&self, key: K) -> Option<&Dense> {
        self.0.get(&key)
    }

    pub fn contains(&self, key: K) -> bool {
        self.0.contains_key(&key)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (K, &Dense)> {
        self.0.iter().map(|(k, v)| (*k, v))
    }

    /// The columns of `order`, stacked in that order.
    ///
    /// Errors [`SkoroError::ShapeMismatch`] on an empty `order`, and
    /// [`SkoroError::StructureMismatch`] if a key in it has no column.
    pub fn flatten(&self, order: &[K]) -> SkoroResult<Dense> {
        let parts = order
            .iter()
            .map(|k| self.0.get(k).ok_or(SkoroError::StructureMismatch))
            .collect::<SkoroResult<Vec<_>>>()?;
        Dense::vstack_all(&parts)
    }

    /// Split a stacked column back into one column per key, `order[i]` taking `dims[i]` rows.
    ///
    /// Errors [`SkoroError::ShapeMismatch`] if the lengths disagree or `flat` is not a column.
    pub fn unflatten(order: &[K], dims: &[usize], flat: &Dense) -> SkoroResult<Self> {
        if order.len() != dims.len() {
            return Err(SkoroError::ShapeMismatch {
                expected: order.len(),
                found: dims.len(),
            });
        }
        let total: usize = dims.iter().sum();
        if flat.cols() != 1 || flat.rows() != total {
            return Err(SkoroError::ShapeMismatch {
                expected: total,
                found: flat.rows(),
            });
        }
        let mut out = Self::new();
        let mut at = 0;
        for (key, &dim) in order.iter().zip(dims) {
            out.0.insert(*key, flat.block(at, 0, dim, 1));
            at += dim;
        }
        Ok(out)
    }
}
