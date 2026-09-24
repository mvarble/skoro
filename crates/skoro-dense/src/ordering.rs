//! The order keys are eliminated in.
//!
//! Fill-in, and so the cost of elimination, is decided entirely here. COLAMD runs on the **block**
//! pattern — one row per factor, one column per key, one entry per (factor, key) — not on the
//! scalar pattern: every scalar column of a key has the same sparsity, so ordering keys is both
//! cheaper and exactly as good. This is GTSAM's choice too.
//!
//! There is no CCOLAMD in `faer`, so [`Ordering::ConstrainedLast`] orders the pattern with the kept
//! columns deleted and appends them. That ignores the fill the kept keys would cause among
//! themselves, which is harmless: they are eliminated last, into one dense front, whatever order
//! COLAMD would have given them.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::sparse::SymbolicSparseColMatRef;
use faer::sparse::linalg::colamd;
use skoro_core::error::{SkoroError, SkoroResult};

use crate::factor_graph::Structure;
use crate::values::Key;

/// How to choose an elimination order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ordering<K: Key> {
    /// Approximate minimum degree on the block pattern.
    Colamd,
    /// COLAMD on every key but these, then these last, in the order given. The order a marginal
    /// over them is computed with, since what is eliminated last is what is left.
    ConstrainedLast(Vec<K>),
    /// The keys' own `Ord` order. Rarely good; useful for tests and for chains already in order.
    Natural,
    /// This exact order, which must name every key once.
    Custom(Vec<K>),
}

impl<K: Key> Ordering<K> {
    /// The order itself, over every key of `structure`.
    ///
    /// Errors [`SkoroError::StructureMismatch`] if a `Custom` order is not a permutation of the
    /// keys, or a `ConstrainedLast` list names an undeclared key or one twice.
    pub fn resolve(&self, structure: &Structure<K>) -> SkoroResult<Vec<K>> {
        let n = structure.keys.len();
        let indices = match self {
            Ordering::Natural => (0..n).collect(),
            Ordering::Colamd => colamd_over(structure, &vec![true; n])?,
            Ordering::Custom(order) => {
                let indices = key_indices(structure, order)?;
                if indices.len() != n {
                    return Err(SkoroError::StructureMismatch);
                }
                indices
            }
            Ordering::ConstrainedLast(kept) => {
                let kept = key_indices(structure, kept)?;
                let mut free = vec![true; n];
                for &k in &kept {
                    free[k] = false;
                }
                let mut order = colamd_over(structure, &free)?;
                order.extend(kept);
                order
            }
        };
        Ok(indices.into_iter().map(|i| structure.keys[i]).collect())
    }
}

/// Indices of `keys` in `structure`, refusing unknown or repeated keys.
pub(crate) fn key_indices<K: Key>(structure: &Structure<K>, keys: &[K]) -> SkoroResult<Vec<usize>> {
    let mut seen = vec![false; structure.keys.len()];
    keys.iter()
        .map(|k| {
            let i = structure
                .keys
                .binary_search(k)
                .map_err(|_| SkoroError::StructureMismatch)?;
            if std::mem::replace(&mut seen[i], true) {
                return Err(SkoroError::StructureMismatch);
            }
            Ok(i)
        })
        .collect()
}

/// COLAMD over the keys marked in `include`, as key indices in elimination order.
///
/// Factors keep only their included keys; a factor left with none is dropped from the pattern.
/// An included key no factor touches still appears — COLAMD places an empty column like any
/// other — and it is elimination, not ordering, that refuses it.
pub(crate) fn colamd_over<K: Key>(
    structure: &Structure<K>,
    include: &[bool],
) -> SkoroResult<Vec<usize>> {
    let columns: Vec<usize> = (0..include.len()).filter(|&i| include[i]).collect();
    let n = columns.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut local = vec![usize::MAX; include.len()];
    for (j, &key) in columns.iter().enumerate() {
        local[key] = j;
    }

    // Column-compressed pattern: for each included key, the factors touching it.
    let mut rows_of: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut m = 0;
    for keys in &structure.factors {
        let mut any = false;
        for &k in keys {
            if include[k] {
                rows_of[local[k]].push(m);
                any = true;
            }
        }
        if any {
            m += 1;
        }
    }
    let mut col_ptr = Vec::with_capacity(n + 1);
    let mut row_idx = Vec::new();
    col_ptr.push(0usize);
    for rows in &rows_of {
        row_idx.extend_from_slice(rows);
        col_ptr.push(row_idx.len());
    }
    let nnz = row_idx.len();
    let pattern = SymbolicSparseColMatRef::new_checked(m, n, &col_ptr, None, &row_idx);

    let mut perm = vec![0usize; n];
    let mut perm_inv = vec![0usize; n];
    let mut buffer = MemBuffer::new(colamd::order_scratch::<usize>(m, n, nnz));
    colamd::order(
        &mut perm,
        &mut perm_inv,
        pattern,
        colamd::Control::default(),
        MemStack::new(&mut buffer),
    )
    .map_err(|_| SkoroError::StructureMismatch)?;
    Ok(perm.into_iter().map(|j| columns[j]).collect())
}
