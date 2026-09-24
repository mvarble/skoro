//! Symbolic elimination: which keys are eliminated together, and what each leaves behind.
//!
//! Given an order, eliminating key `j` combines every factor that touches it (the ones whose
//! earliest key is `j`, plus what earlier eliminations left on `j`) into one conditional on `j`
//! and one new factor on the rest — the **separator** of `j`. That depends only on which keys
//! each factor touches, so it is computed once, here, without a single number, and reused for as
//! long as the structure holds. Parenting each key to the earliest key of its separator gives the
//! elimination tree.
//!
//! A chain of keys each of whose separator is its parent plus the parent's separator is
//! eliminated as one dense front — a clique of the **junction tree** — which is what makes the
//! numeric pass multifrontal rather than one key at a time. [`JunctionTree::sequential`] turns the
//! merging off, giving one key per clique; the two agree to rounding and the sequential one is the
//! baseline the multifrontal one is tested against.
//!
//! A tree can stop early. Eliminating only a subset leaves every factor that touches none of it
//! untouched — passed through by index, never read — and leaves the separators of the last
//! eliminated cliques on the kept keys. Those separator factors are the Schur complement, in
//! square-root form. This is the lazy part: the cost is in the eliminated keys' neighbourhood, not
//! in the size of the graph.

use skoro_core::error::{SkoroError, SkoroResult};

use crate::factor_graph::{GaussianFactorGraph, Structure};
use crate::ordering::{Ordering, colamd_over, key_indices};
use crate::values::Key;

/// A clique of the junction tree: keys eliminated together, and the separator they leave.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Clique {
    /// Key indices, in elimination order.
    pub(crate) frontals: Vec<usize>,
    /// Key indices, in elimination order. Keys past the eliminated prefix are kept keys.
    pub(crate) separator: Vec<usize>,
    /// Indices of the original factors whose earliest key is one of the frontals.
    pub(crate) factors: Vec<usize>,
    pub(crate) children: Vec<usize>,
    pub(crate) parent: Option<usize>,
}

/// The symbolic elimination of one factor-graph structure under one order.
///
/// Cliques are stored children before parents, so a forward sweep is a valid bottom-up
/// (elimination) order and a backward sweep a valid top-down (back-substitution) one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JunctionTree<K: Key> {
    pub(crate) structure: Structure<K>,
    /// Key indices in elimination order; the first `eliminated` of them are eliminated.
    pub(crate) order: Vec<usize>,
    pub(crate) eliminated: usize,
    pub(crate) cliques: Vec<Clique>,
    /// Factors touching no eliminated key, carried to the remaining graph untouched.
    pub(crate) passthrough: Vec<usize>,
}

impl<K: Key> JunctionTree<K> {
    /// Eliminate every key of `graph`, in the order `ordering` resolves to, multifrontally.
    pub fn new(graph: &GaussianFactorGraph<K>, ordering: &Ordering<K>) -> SkoroResult<Self> {
        let structure = graph.structure();
        let order = ordering.resolve(&structure)?;
        let n = order.len();
        Self::build(structure, &order, n, true)
    }

    /// As [`new`](Self::new), one key per clique. The baseline the multifrontal tree is tested
    /// against; slower, never different beyond rounding.
    pub fn sequential(graph: &GaussianFactorGraph<K>, ordering: &Ordering<K>) -> SkoroResult<Self> {
        let structure = graph.structure();
        let order = ordering.resolve(&structure)?;
        let n = order.len();
        Self::build(structure, &order, n, false)
    }

    /// Eliminate only `eliminate`, in a COLAMD order over them, keeping every other key.
    ///
    /// The kept keys are ordered after, in key order; what the elimination leaves on them is
    /// [`Elimination::remaining`](crate::bayes::Elimination::remaining).
    pub fn partial(graph: &GaussianFactorGraph<K>, eliminate: &[K]) -> SkoroResult<Self> {
        let structure = graph.structure();
        let chosen = key_indices(&structure, eliminate)?;
        let mut include = vec![false; structure.keys.len()];
        for &k in &chosen {
            include[k] = true;
        }
        let mut order = colamd_over(&structure, &include)?;
        let eliminated = order.len();
        order.extend((0..include.len()).filter(|&k| !include[k]));
        let order: Vec<K> = order.into_iter().map(|i| structure.keys[i]).collect();
        Self::build(structure, &order, eliminated, true)
    }

    /// The general form: `order` names every key, and its first `eliminated` are eliminated.
    ///
    /// Errors [`SkoroError::StructureMismatch`] if `order` is not a permutation of the keys, and
    /// [`SkoroError::KeyUnconstrained`] if an eliminated key is touched by no factor.
    pub fn with_order(
        graph: &GaussianFactorGraph<K>,
        order: &[K],
        eliminated: usize,
        multifrontal: bool,
    ) -> SkoroResult<Self> {
        Self::build(graph.structure(), order, eliminated, multifrontal)
    }

    fn build(
        structure: Structure<K>,
        order: &[K],
        eliminated: usize,
        multifrontal: bool,
    ) -> SkoroResult<Self> {
        let n = structure.keys.len();
        let order = key_indices(&structure, order)?;
        if order.len() != n || eliminated > n {
            return Err(SkoroError::StructureMismatch);
        }
        let mut position = vec![0; n];
        for (p, &k) in order.iter().enumerate() {
            position[k] = p;
        }

        // Each factor is homed at its earliest key, if that key is eliminated.
        let mut homed: Vec<Vec<usize>> = vec![Vec::new(); eliminated];
        let mut passthrough = Vec::new();
        for (f, keys) in structure.factors.iter().enumerate() {
            match keys.iter().map(|&k| position[k]).min() {
                Some(p) if p < eliminated => homed[p].push(f),
                _ => passthrough.push(f),
            }
        }
        // A key homed nowhere may still be reached as a later key of some factor, which leaves it
        // rows through its children; only a key no factor touches at all has none.
        let mut touched = vec![false; n];
        for keys in &structure.factors {
            for &k in keys {
                touched[k] = true;
            }
        }
        if let Some(&k) = order[..eliminated].iter().find(|&&k| !touched[k]) {
            return Err(SkoroError::KeyUnconstrained { index: k });
        }

        // Separators, as sorted positions, and the elimination tree over positions.
        let mut pending: Vec<Vec<usize>> = vec![Vec::new(); eliminated];
        let mut separator: Vec<Vec<usize>> = Vec::with_capacity(eliminated);
        let mut parent: Vec<Option<usize>> = Vec::with_capacity(eliminated);
        for p in 0..eliminated {
            let mut set = std::mem::take(&mut pending[p]);
            for &f in &homed[p] {
                set.extend(structure.factors[f].iter().map(|&k| position[k]));
            }
            set.sort_unstable();
            set.dedup();
            set.retain(|&q| q > p);
            let up = set.first().copied().filter(|&q| q < eliminated);
            if let Some(q) = up {
                pending[q].extend(set.iter().copied().filter(|&r| r != q));
            }
            parent.push(up);
            separator.push(set);
        }

        // Merge chains into cliques. A position joins its only child's clique when its separator
        // is the child's minus itself, so the pair eliminates as one dense front with no extra
        // fill.
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); eliminated];
        for (p, up) in parent.iter().enumerate() {
            if let Some(q) = *up {
                children[q].push(p);
            }
        }
        let mut clique_of = vec![usize::MAX; eliminated];
        let mut cliques: Vec<Clique> = Vec::new();
        for p in 0..eliminated {
            let merge_into = match children[p].as_slice() {
                [c] if multifrontal && separator[*c].len() == separator[p].len() + 1 => {
                    Some(clique_of[*c])
                }
                _ => None,
            };
            let id = match merge_into {
                Some(id) => {
                    cliques[id].frontals.push(order[p]);
                    cliques[id].factors.extend_from_slice(&homed[p]);
                    id
                }
                None => {
                    cliques.push(Clique {
                        frontals: vec![order[p]],
                        separator: Vec::new(),
                        factors: homed[p].clone(),
                        children: Vec::new(),
                        parent: None,
                    });
                    cliques.len() - 1
                }
            };
            clique_of[p] = id;
            cliques[id].separator = separator[p].iter().map(|&q| order[q]).collect();
        }
        // Cliques were created in increasing position of their first frontal, and a clique's
        // parent is homed at a later position, so creation order is already children-first.
        for id in 0..cliques.len() {
            let last = *cliques[id].frontals.last().expect("a clique has a frontal");
            if let Some(q) = parent[position[last]] {
                let up = clique_of[q];
                cliques[id].parent = Some(up);
                cliques[up].children.push(id);
            }
        }

        Ok(Self {
            structure,
            order,
            eliminated,
            cliques,
            passthrough,
        })
    }

    /// The structure this tree was computed for. A graph with a different one is refused by
    /// [`eliminate`](crate::bayes::eliminate).
    pub fn structure(&self) -> &Structure<K> {
        &self.structure
    }

    /// Every key, in elimination order; the first [`eliminated`](Self::eliminated) are eliminated.
    pub fn order(&self) -> Vec<K> {
        self.order.iter().map(|&i| self.structure.keys[i]).collect()
    }

    pub fn eliminated(&self) -> usize {
        self.eliminated
    }

    pub fn clique_count(&self) -> usize {
        self.cliques.len()
    }

    /// The largest front, in scalar columns (frontal and separator dimensions summed), which is
    /// what bounds the cost of the numeric pass.
    pub fn max_front_width(&self) -> usize {
        self.cliques
            .iter()
            .map(|c| {
                c.frontals
                    .iter()
                    .chain(&c.separator)
                    .map(|&k| self.structure.dims[k])
                    .sum::<usize>()
            })
            .max()
            .unwrap_or(0)
    }
}
