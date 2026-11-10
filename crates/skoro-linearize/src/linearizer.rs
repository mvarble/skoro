//! The linearizer: a density query's factors, read off its terms and refused by structure, the
//! factor structure over its free quantities, and the programs and seeds of its passes.

use std::collections::HashMap;
use std::sync::Arc;

use skoro_core::{Frames, Layout, NodeId, Program, QDyn, Query, Snapshot, Term};
use skoro_elimination::{Schedule, Structure};
use skoro_normal::{Form, roles};
use skoro_tensor::{Frame, Precision, Target};

use crate::Error;

/// Which rows a linearization is assembled from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Assembly {
    /// Each factor's residual: the linearization of the query's negative log-density.
    Residual,
    /// Each factor's residual and, where the scale is free, its information root's scale rows:
    /// the Fisher linearization.
    Fisher,
}

/// A term reached from a free quantity, and what the assembly reads of it.
#[derive(Clone, Debug)]
pub(crate) struct Factor {
    pub(crate) term: NodeId,
    /// The blocks of its variables, increasing.
    pub(crate) blocks: Vec<usize>,
    /// The node bound under the residual role, and its number of components.
    pub(crate) residual: QDyn,
    pub(crate) components: usize,
    /// The node bound under the log-volume role, where it is carried by the linear term.
    pub(crate) log_volume: Option<QDyn>,
}

/// A density query of a snapshot, ready to be linearized at any batch of points: its factors, its
/// structure, its programs and its seeds, all fixed by the model and the query.
pub struct Linearizer {
    pub(crate) snapshot: Snapshot,
    pub(crate) target: Target,
    pub(crate) assembly: Assembly,
    /// The query's density program: `f`, and the layout.
    pub(crate) density: Arc<Program>,
    pub(crate) factors: Vec<Factor>,
    pub(crate) structure: Arc<Structure>,
    /// The program of every bound node the assembly reads, under the query's observations.
    pub(crate) rows: Arc<Program>,
    /// The compressed identity frames at the rows program's free quantities, of one row each.
    pub(crate) seeds: Frames,
    /// Where each block's columns start in the seeds, by its colour.
    pub(crate) columns: Vec<usize>,
    /// The seeds' number of columns.
    pub(crate) width: usize,
}

impl std::fmt::Debug for Linearizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Linearizer")
            .field("assembly", &self.assembly)
            .field("quantities", &self.layout().quantities())
            .field("factors", &self.factors.len())
            .finish_non_exhaustive()
    }
}

/// The scale a term binds under one of the normal's four roles, if any.
pub(crate) fn bound_scale(term: &Term) -> Result<Option<QDyn>, Error> {
    for form in [
        Form::Isotropic,
        Form::Diagonal,
        Form::CovarianceFactor,
        Form::Whitening,
    ] {
        if let Some(scale) = term.bound(roles::scale(form))? {
            return Ok(Some(scale));
        }
    }
    Ok(None)
}

impl Linearizer {
    /// The linearizer of `query`, a density query of `snapshot`, on `target`. Refuses a query of
    /// nodes (`NotADensity`), a free discrete quantity (the core's `FreeDiscrete`), a factor that
    /// binds no residual (`NoResidual`), with `Assembly::Residual` a factor whose scale is reached
    /// from a free quantity (`ScaleNotFixed`), and with `Assembly::Fisher` such a factor that
    /// binds no information root or no tangent (`NoInformationRoot`).
    pub fn new(
        snapshot: &Snapshot,
        query: &Query,
        target: &Target,
        assembly: Assembly,
    ) -> Result<Linearizer, Error> {
        let terms = query.terms(snapshot).ok_or(Error::NotADensity)?;
        let density = snapshot.compile(query, target)?;
        density.refuse_free_discrete()?;
        let layout = density.layout();
        let index: HashMap<NodeId, usize> = layout
            .quantities()
            .iter()
            .enumerate()
            .map(|(i, &q)| (q, i))
            .collect();
        let mut factors = Vec::new();
        for id in terms {
            let term = snapshot
                .term(id)
                .ok_or(skoro_core::Error::NotATerm { id })?;
            let mut blocks: Vec<usize> = term
                .involves
                .iter()
                .filter_map(|q| index.get(q).copied())
                .collect();
            if blocks.is_empty() {
                continue;
            }
            blocks.sort_unstable();
            let residual = term
                .bound(&roles::RESIDUAL)
                .map_err(|_| Error::NoResidual { term: id })?;
            let components = snapshot
                .space(residual)
                .and_then(|space| space.dimension())
                .unwrap_or(0);
            let log_volume = term.bound(&roles::LOG_VOLUME)?;
            if let Some(scale) = bound_scale(term)? {
                let free = snapshot
                    .reached_from(scale)
                    .into_iter()
                    .find(|q| index.contains_key(q));
                if let Some(quantity) = free {
                    match assembly {
                        Assembly::Residual => {
                            return Err(Error::ScaleNotFixed { term: id, quantity });
                        }
                        Assembly::Fisher => {
                            let root = term.bound(&roles::INFORMATION_ROOT)?;
                            let tangent = term.bound(&roles::TANGENT)?;
                            if root.is_none() || tangent.is_none() {
                                return Err(Error::NoInformationRoot { term: id, quantity });
                            }
                        }
                    }
                }
            }
            factors.push(Factor {
                term: id,
                blocks,
                residual,
                components,
                log_volume,
            });
        }
        let dimensions: Vec<usize> = layout
            .blocks()
            .iter()
            .map(|block| block.tangent.len())
            .collect();
        let structure = Arc::new(Structure::new(
            dimensions.clone(),
            factors.iter().map(|f| f.blocks.clone()).collect(),
        )?);
        // The rows program: every node the assembly reads, under the query's observations.
        let targets = crate::assemble::targets(
            factors
                .iter()
                .flat_map(|f| std::iter::once(f.residual).chain(f.log_volume)),
        );
        let rows_query = Query::of(targets).given(query.observations().clone());
        let rows = snapshot.compile(&rows_query, target)?;
        let (columns, width) = colouring(&dimensions, &factors);
        let mut seeds = Frames::new();
        for free in rows.free() {
            let Some(d) = free.space.dimension() else {
                continue;
            };
            let k = *index.get(&free.node).ok_or(Error::NotFree {
                quantity: free.node,
            })?;
            let mut data = vec![0.0; d * width];
            for i in 0..d {
                data[i * width + columns[k] + i] = 1.0;
            }
            seeds.insert(free.node, Frame::from_host(&data, 1, d, width, target));
        }
        Ok(Linearizer {
            snapshot: snapshot.clone(),
            target: target.clone(),
            assembly,
            density,
            factors,
            structure,
            rows,
            seeds,
            columns,
            width,
        })
    }

    /// The unit roundoff of the target's precision.
    pub(crate) fn roundoff(&self) -> f64 {
        match self.target.precision() {
            Precision::Double => f64::EPSILON / 2.0,
            Precision::Single => f32::EPSILON as f64 / 2.0,
        }
    }

    /// The free continuous quantities, in the order of the blocks.
    pub fn layout(&self) -> &Layout {
        self.density.layout()
    }

    pub fn structure(&self) -> &Arc<Structure> {
        &self.structure
    }

    pub fn assembly(&self) -> Assembly {
        self.assembly
    }

    pub fn snapshot(&self) -> &Snapshot {
        &self.snapshot
    }

    pub fn target(&self) -> &Target {
        &self.target
    }

    /// Each factor's term, in the order of the structure's factors.
    pub fn terms(&self) -> Vec<NodeId> {
        self.factors.iter().map(|f| f.term).collect()
    }

    /// `Schedule::all` of the structure.
    pub fn schedule(&self) -> Result<Schedule, Error> {
        Ok(Schedule::all(&self.structure)?)
    }
}

/// A greedy colouring of the free quantities, in the layout's order: two quantities joined to a
/// common factor in the factor structure have different colours. Each colour owns a range of
/// columns as wide as its widest block; the start of each block's range, and the total width.
fn colouring(dimensions: &[usize], factors: &[Factor]) -> (Vec<usize>, usize) {
    let n = dimensions.len();
    let mut conflicts: Vec<Vec<usize>> = vec![Vec::new(); n];
    for factor in factors {
        for &a in &factor.blocks {
            for &b in &factor.blocks {
                if a != b {
                    conflicts[a].push(b);
                }
            }
        }
    }
    let mut colour: Vec<Option<usize>> = vec![None; n];
    let mut widths: Vec<usize> = Vec::new();
    for k in 0..n {
        let taken: Vec<usize> = conflicts[k].iter().filter_map(|&j| colour[j]).collect();
        let c = (0..).find(|c| !taken.contains(c)).unwrap_or(0);
        colour[k] = Some(c);
        if c == widths.len() {
            widths.push(0);
        }
        widths[c] = widths[c].max(dimensions[k]);
    }
    let mut starts = Vec::with_capacity(widths.len());
    let mut width = 0;
    for w in &widths {
        starts.push(width);
        width += w;
    }
    let columns = colour.iter().map(|c| starts[c.unwrap_or(0)]).collect();
    (columns, width)
}
