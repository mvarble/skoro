use std::ops::Range;
use std::sync::Arc;

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::sparse::SymbolicSparseColMatRef;
use faer::sparse::linalg::colamd;

use crate::Error;

/// The factor structure: blocks of given dimensions, and for each factor the blocks it involves,
/// in increasing order.
#[derive(Clone, PartialEq, Debug)]
pub struct Structure {
    dimensions: Vec<usize>,
    factors: Vec<Vec<usize>>,
    /// Where each block's coordinates start in the stacked tangent.
    offsets: Vec<usize>,
}

impl Structure {
    /// Refuses a factor that names a block twice, out of increasing order, or one that does not
    /// exist (`Structure`).
    pub fn new(dimensions: Vec<usize>, factors: Vec<Vec<usize>>) -> Result<Structure, Error> {
        for (f, blocks) in factors.iter().enumerate() {
            let increasing = blocks.windows(2).all(|w| w[0] < w[1]);
            if !increasing || blocks.iter().any(|&k| k >= dimensions.len()) {
                return Err(Error::Structure { factor: f });
            }
        }
        let mut offsets = Vec::with_capacity(dimensions.len());
        let mut start = 0;
        for d in &dimensions {
            offsets.push(start);
            start += d;
        }
        Ok(Structure {
            dimensions,
            factors,
            offsets,
        })
    }

    /// `D`, the sum of the blocks' dimensions.
    pub fn dimension(&self) -> usize {
        self.dimensions.iter().sum()
    }

    /// The blocks' dimensions.
    pub fn blocks(&self) -> &[usize] {
        &self.dimensions
    }

    pub fn factors(&self) -> &[Vec<usize>] {
        &self.factors
    }

    /// Block `k`'s coordinates in the stacked tangent.
    pub fn range(&self, k: usize) -> Range<usize> {
        self.offsets[k]..self.offsets[k] + self.dimensions[k]
    }

    /// The number of columns of factor `f`'s rows.
    pub(crate) fn width(&self, f: usize) -> usize {
        self.factors[f].iter().map(|&k| self.dimensions[k]).sum()
    }

    /// The coordinates of `tangent` at factor `f`'s blocks, in order.
    pub(crate) fn gather(&self, f: usize, tangent: &[f64]) -> Vec<f64> {
        self.factors[f]
            .iter()
            .flat_map(|&k| tangent[self.range(k)].iter().copied())
            .collect()
    }
}

/// A clique of the junction tree: blocks eliminated together, consecutive in the order, and the
/// separator their new factor is left on.
#[derive(Clone, PartialEq, Debug)]
pub(crate) struct Clique {
    pub(crate) frontals: Vec<usize>,
    /// Sorted by block.
    pub(crate) separator: Vec<usize>,
    pub(crate) parent: Option<usize>,
    pub(crate) children: Vec<usize>,
    /// The structure's factors first collected here.
    pub(crate) factors: Vec<usize>,
}

/// The symbolic factorization: an order, an elimination tree and the junction tree of its
/// cliques, for one structure and one set of blocks to eliminate. Shared by every problem of the
/// structure.
#[derive(Clone, PartialEq, Debug)]
pub struct Schedule {
    pub(crate) structure: Arc<Structure>,
    /// The eliminated blocks, clique by clique, children before parents.
    pub(crate) order: Vec<usize>,
    /// Children before parents.
    pub(crate) cliques: Vec<Clique>,
    /// The blocks kept, in the order a marginal or a reduced factor lays them out.
    pub(crate) kept: Vec<usize>,
    /// The structure's factors joined to no eliminated block.
    pub(crate) left: Vec<usize>,
}

impl Schedule {
    /// Every block, in a COLAMD order of the block pattern.
    pub fn all(structure: &Arc<Structure>) -> Result<Schedule, Error> {
        let n = structure.blocks().len();
        let order = colamd(structure, &vec![true; n]);
        Ok(Schedule::build(structure, order, Vec::new()))
    }

    /// Every block but `keep`, which are kept: the schedule of a marginal on `keep`, laid out in
    /// `keep`'s order (`Shape` for a block named twice or one that does not exist).
    pub fn keeping(structure: &Arc<Structure>, keep: &[usize]) -> Result<Schedule, Error> {
        let kept = distinct(structure, keep)?;
        let include: Vec<bool> = kept.iter().map(|&k| !k).collect();
        let order = colamd(structure, &include);
        Ok(Schedule::build(structure, order, keep.to_vec()))
    }

    /// The blocks `eliminate` alone, from the factors joined to them: the schedule of a
    /// replacement, the others kept in increasing order (`Shape` as `keeping`).
    pub fn eliminating(structure: &Arc<Structure>, eliminate: &[usize]) -> Result<Schedule, Error> {
        let include = distinct(structure, eliminate)?;
        let order = colamd(structure, &include);
        let kept = (0..include.len()).filter(|&k| !include[k]).collect();
        Ok(Schedule::build(structure, order, kept))
    }

    /// The blocks of `order` eliminated in that order, as far as the junction tree allows, and
    /// the others kept in increasing order: for the tests of order-independence (`Shape` as
    /// `keeping`).
    pub fn in_order(structure: &Arc<Structure>, order: Vec<usize>) -> Result<Schedule, Error> {
        let include = distinct(structure, &order)?;
        let kept = (0..include.len()).filter(|&k| !include[k]).collect();
        Ok(Schedule::build(structure, order, kept))
    }

    pub fn structure(&self) -> &Arc<Structure> {
        &self.structure
    }

    /// The eliminated blocks in the order they are eliminated: clique by clique, children before
    /// parents.
    pub fn order(&self) -> &[usize] {
        &self.order
    }

    /// The blocks kept.
    pub fn kept(&self) -> &[usize] {
        &self.kept
    }

    /// The number of cliques.
    pub fn cliques(&self) -> usize {
        self.cliques.len()
    }

    /// The blocks clique `i` eliminates together, in order.
    pub fn frontals(&self, i: usize) -> &[usize] {
        &self.cliques[i].frontals
    }

    /// The separator of clique `i`, sorted by block.
    pub fn separator(&self, i: usize) -> &[usize] {
        &self.cliques[i].separator
    }

    /// The elimination tree read off `order`, and its cliques by
    /// [the chain rule of the note](eq:math/square-root-elimination/clique-chain): a block joins
    /// the clique of a child whose separator is the block and the block's own separator.
    fn build(structure: &Arc<Structure>, order: Vec<usize>, kept: Vec<usize>) -> Schedule {
        let n = structure.blocks().len();
        let mut position = vec![None; n];
        for (i, &k) in order.iter().enumerate() {
            position[k] = Some(i);
        }
        // Each factor is collected by the first of its blocks to be eliminated.
        let mut assigned: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut left = Vec::new();
        for (f, blocks) in structure.factors().iter().enumerate() {
            match blocks
                .iter()
                .copied()
                .filter(|&k| position[k].is_some())
                .min_by_key(|&k| position[k])
            {
                Some(first) => assigned[first].push(f),
                None => left.push(f),
            }
        }
        let mut separators: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut parent: Vec<Option<usize>> = vec![None; n];
        // The clique of each eliminated block.
        let mut clique_of: Vec<usize> = vec![usize::MAX; n];
        let mut cliques: Vec<Clique> = Vec::new();
        for &k in &order {
            let mut separator: Vec<usize> = assigned[k]
                .iter()
                .flat_map(|&f| structure.factors()[f].iter().copied())
                .chain(
                    children[k]
                        .iter()
                        .flat_map(|&c| separators[c].iter().copied()),
                )
                .filter(|&j| j != k)
                .collect();
            separator.sort_unstable();
            separator.dedup();
            parent[k] = separator
                .iter()
                .copied()
                .filter(|&j| position[j].is_some())
                .min_by_key(|&j| position[j]);
            if let Some(p) = parent[k] {
                children[p].push(k);
            }
            let joins = children[k].iter().copied().rfind(|&c| {
                let clique = &cliques[clique_of[c]];
                clique.frontals.last() == Some(&c)
                    && separators[c].len() == separator.len() + 1
                    && separators[c]
                        .iter()
                        .filter(|&&j| j != k)
                        .eq(separator.iter())
            });
            match joins {
                Some(c) => {
                    let i = clique_of[c];
                    cliques[i].frontals.push(k);
                    cliques[i].factors.extend(assigned[k].iter().copied());
                    cliques[i].separator = separator.clone();
                    clique_of[k] = i;
                }
                None => {
                    clique_of[k] = cliques.len();
                    cliques.push(Clique {
                        frontals: vec![k],
                        separator: separator.clone(),
                        parent: None,
                        children: Vec::new(),
                        factors: assigned[k].clone(),
                    });
                }
            }
            separators[k] = separator;
        }
        for clique in &mut cliques {
            let last = *clique.frontals.last().unwrap_or(&0);
            clique.parent = parent[last].map(|p| clique_of[p]);
        }
        // A postorder of the junction tree: children before parents, roots in order of creation.
        let mut kids: Vec<Vec<usize>> = vec![Vec::new(); cliques.len()];
        for (i, clique) in cliques.iter().enumerate() {
            if let Some(p) = clique.parent {
                kids[p].push(i);
            }
        }
        let mut postorder = Vec::with_capacity(cliques.len());
        for root in (0..cliques.len()).filter(|&i| cliques[i].parent.is_none()) {
            let mut stack = vec![(root, false)];
            while let Some((i, expanded)) = stack.pop() {
                if expanded {
                    postorder.push(i);
                } else {
                    stack.push((i, true));
                    stack.extend(kids[i].iter().rev().map(|&c| (c, false)));
                }
            }
        }
        let mut renamed = vec![0; cliques.len()];
        for (new, &old) in postorder.iter().enumerate() {
            renamed[old] = new;
        }
        let cliques: Vec<Clique> = postorder
            .iter()
            .map(|&old| {
                let clique = &cliques[old];
                Clique {
                    frontals: clique.frontals.clone(),
                    separator: clique.separator.clone(),
                    parent: clique.parent.map(|p| renamed[p]),
                    children: kids[old].iter().map(|&c| renamed[c]).collect(),
                    factors: clique.factors.clone(),
                }
            })
            .collect();
        let order = cliques
            .iter()
            .flat_map(|c| c.frontals.iter().copied())
            .collect();
        Schedule {
            structure: structure.clone(),
            order,
            cliques,
            kept,
            left,
        }
    }
}

/// Whether each block is among `blocks`, which must be distinct blocks of the structure.
fn distinct(structure: &Structure, blocks: &[usize]) -> Result<Vec<bool>, Error> {
    let mut include = vec![false; structure.blocks().len()];
    for &k in blocks {
        if k >= include.len() || include[k] {
            return Err(Error::Shape { factor: None });
        }
        include[k] = true;
    }
    Ok(include)
}

/// COLAMD over the blocks marked in `include`, on the block pattern: one row per factor that
/// involves one of them, one column per block, one entry per factor and block. A block no factor
/// involves is placed like any other; it is the elimination that refuses it.
fn colamd(structure: &Structure, include: &[bool]) -> Vec<usize> {
    let columns: Vec<usize> = (0..include.len()).filter(|&k| include[k]).collect();
    let n = columns.len();
    let mut local = vec![usize::MAX; include.len()];
    for (j, &k) in columns.iter().enumerate() {
        local[k] = j;
    }
    let mut rows_of: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut m = 0;
    for blocks in structure.factors() {
        let mut any = false;
        for &k in blocks {
            if include[k] {
                rows_of[local[k]].push(m);
                any = true;
            }
        }
        if any {
            m += 1;
        }
    }
    if n == 0 || m == 0 {
        return columns;
    }
    let mut col_ptr = Vec::with_capacity(n + 1);
    let mut row_idx = Vec::new();
    col_ptr.push(0usize);
    for rows in &rows_of {
        row_idx.extend_from_slice(rows);
        col_ptr.push(row_idx.len());
    }
    let pattern = SymbolicSparseColMatRef::new_checked(m, n, &col_ptr, None, &row_idx);
    let (mut perm, mut perm_inv) = (vec![0usize; n], vec![0usize; n]);
    let mut buffer = MemBuffer::new(colamd::order_scratch::<usize>(m, n, row_idx.len()));
    match colamd::order(
        &mut perm,
        &mut perm_inv,
        pattern,
        colamd::Control::default(),
        MemStack::new(&mut buffer),
    ) {
        Ok(()) => perm.into_iter().map(|j| columns[j]).collect(),
        // COLAMD refuses only patterns too large to index, which a structure of blocks never is;
        // the order of the blocks is then as good as any.
        Err(_) => columns,
    }
}
