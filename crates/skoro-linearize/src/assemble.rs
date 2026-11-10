//! The passes at a point, and each row's problem.

use skoro_core::{Frames, Layout, NodeId, QDyn, Space, Values};
use skoro_elimination::{Matrix, Problem, Rows};
use skoro_tensor::{Points, Target, Value};

use crate::{Assembly, Error, Linearizer};

/// A batch's linearization: one problem per row, over the linearizer's structure.
#[derive(Clone, Debug)]
pub struct Linearized {
    pub(crate) problems: Vec<Problem>,
    pub(crate) assembly: Assembly,
    pub(crate) layout: Layout,
    /// The quantities of a curved type, whose covariances carry an error of order `σ²`.
    pub(crate) curved: Vec<NodeId>,
}

impl Linearized {
    /// One problem per row of the batch.
    pub fn problems(&self) -> &[Problem] {
        &self.problems
    }

    pub fn assembly(&self) -> Assembly {
        self.assembly
    }

    /// The free continuous quantities, in the order of the blocks.
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// The quantities of a curved type among them.
    pub fn curved(&self) -> &[NodeId] {
        &self.curved
    }
}

/// A node's value, row by row, on the host.
pub(crate) fn host(values: &Values, id: impl Into<NodeId>) -> Result<Vec<f64>, Error> {
    let id = id.into();
    match values.get(id) {
        Some(Value::Continuous(points)) => Ok(points.to_host()),
        _ => Err(Error::Core(skoro_core::Error::MissingValue { id })),
    }
}

/// A node's tangent frame on the host, `[rows, dimension, columns]` row-major.
pub(crate) fn frame(frames: &Frames, id: impl Into<NodeId>) -> Result<Vec<f64>, Error> {
    let id = id.into();
    frames
        .get(id)
        .map(|frame| frame.to_host())
        .ok_or(Error::Core(skoro_core::Error::MissingValue { id }))
}

/// Whether a space is curved: an embedded type, or a product with one among its factors.
pub(crate) fn curved(space: &Space) -> bool {
    match space {
        Space::Embedded(_) => true,
        Space::Product(factors) => factors.iter().any(curved),
        _ => false,
    }
}

/// What a row's assembly reads of a pass: values and tangent frames, row-major, by node.
pub(crate) struct Pass {
    pub(crate) values: Values,
    pub(crate) frames: Frames,
    /// The query's log-density, row by row.
    pub(crate) density: Vec<f64>,
    pub(crate) rows: usize,
}

impl Linearizer {
    /// The rows program's tangent pass at `point`, with the compressed seeds.
    pub(crate) fn pass(&self, point: &Values) -> Result<Pass, Error> {
        let (values, frames) = self.rows.tangent(point, &self.seeds)?;
        let density = host(&self.density.forward(point)?, NodeId::DENSITY)?;
        let rows = density.len();
        Ok(Pass {
            values,
            frames,
            density,
            rows,
        })
    }

    /// The columns of a factor's blocks in a frame of `dimension` rows at batch row `row`: the
    /// matrix `dimension × width` of its Jacobian, its blocks in the structure's order.
    pub(crate) fn cut(
        &self,
        frame: &[f64],
        row: usize,
        dimension: usize,
        blocks: &[usize],
    ) -> Matrix {
        let dimensions = self.structure.blocks();
        let width: usize = blocks.iter().map(|&k| dimensions[k]).sum();
        let mut matrix = Matrix::zeros(dimension, width);
        let mut data = Vec::with_capacity(dimension * width);
        for i in 0..dimension {
            let base = (row * dimension + i) * self.width;
            for &k in blocks {
                let start = base + self.columns[k];
                data.extend_from_slice(&frame[start..start + dimensions[k]]);
            }
        }
        if let Some(full) = Matrix::new(dimension, width, data) {
            matrix = full;
        }
        matrix
    }

    /// The linearization at `point`, one problem per row: each factor's Jacobian and right-hand
    /// side, the linear term from the bound log-volumes that are not folded, and the constant
    /// that makes `m(0) = f(p̂)`.
    pub fn at(&self, point: &Values) -> Result<Linearized, Error> {
        let pass = self.pass(point)?;
        let dimension = self.structure.dimension();
        let has_linear = self.factors.iter().any(|f| f.log_volume.is_some());
        // Each factor's values and frames, on the host once.
        let residuals: Vec<(Vec<f64>, Vec<f64>)> = self
            .factors
            .iter()
            .map(|f| {
                Ok((
                    host(&pass.values, f.residual)?,
                    frame(&pass.frames, f.residual)?,
                ))
            })
            .collect::<Result<_, Error>>()?;
        let log_volumes: Vec<Option<Vec<f64>>> = self
            .factors
            .iter()
            .map(|f| f.log_volume.map(|v| frame(&pass.frames, v)).transpose())
            .collect::<Result<_, Error>>()?;
        let folds = self.folds(&pass)?;
        let mut problems = Vec::with_capacity(pass.rows);
        for row in 0..pass.rows {
            let mut factors = Vec::with_capacity(self.factors.len());
            let mut squares = 0.0;
            let mut linear = has_linear.then(|| vec![0.0; dimension]);
            for (f, factor) in self.factors.iter().enumerate() {
                let d = factor.components;
                let (values, jacobian) = &residuals[f];
                let mut r = values[row * d..(row + 1) * d].to_vec();
                if let Some(c) = &folds[f] {
                    for (r, c) in r.iter_mut().zip(&c[row * d..(row + 1) * d]) {
                        *r += c;
                    }
                }
                squares += r.iter().map(|r| r * r).sum::<f64>();
                let a = self.cut(jacobian, row, d, &factor.blocks);
                factors.push(Rows {
                    a,
                    b: r.iter().map(|r| -r).collect(),
                });
                if let (Some(eta), Some(gradient)) = (&mut linear, &log_volumes[f]) {
                    let g = self.cut(gradient, row, 1, &factor.blocks);
                    let mut at = 0;
                    for &k in &factor.blocks {
                        for c in self.structure.range(k) {
                            eta[c] += g.get(0, at);
                            at += 1;
                        }
                    }
                }
            }
            let constant = -pass.density[row] - 0.5 * squares;
            problems.push(Problem::new(
                self.structure.clone(),
                factors,
                linear,
                constant,
                self.roundoff(),
            )?);
        }
        Ok(Linearized {
            problems,
            assembly: self.assembly,
            layout: self.layout().clone(),
            curved: self.curved_quantities(),
        })
    }

    /// Each folded factor's `ĉ = W⁻ᵀ ∇λ_μ(s)`, row by row: the gradient of its space's chart
    /// log-volume at `(μ̂, ŝ)`, for every folded factor of the space and every row at once, and
    /// `W` the top-left block of the information root's value.
    pub(crate) fn folds(&self, pass: &Pass) -> Result<Vec<Option<Vec<f64>>>, Error> {
        let mut folds: Vec<Option<Vec<f64>>> = vec![None; self.factors.len()];
        for (space, program) in &self.volumes {
            let members: Vec<usize> = (0..self.factors.len())
                .filter(|&f| {
                    self.factors[f]
                        .fold
                        .as_ref()
                        .is_some_and(|g| g.space == *space)
                })
                .collect();
            let (Some(storage), Some(d)) = (Some(space.storage()), space.dimension()) else {
                continue;
            };
            let (mut means, mut tangents) = (Vec::new(), Vec::new());
            let mut roots = Vec::new();
            for &f in &members {
                let Some(fold) = &self.factors[f].fold else {
                    continue;
                };
                means.extend(host(&pass.values, fold.mean)?);
                tangents.extend(host(&pass.values, fold.tangent)?);
                roots.push(host(&pass.values, fold.root)?);
            }
            let n = members.len() * pass.rows;
            let host_target = Target::host();
            let free = program.free();
            let (Some(point), Some(tangent)) = (free.first(), free.get(1)) else {
                continue;
            };
            let at = Values::new()
                .with(
                    point.node,
                    Value::Continuous(Points::from_host(&means, n, storage, &host_target)),
                )
                .with(
                    tangent.node,
                    Value::Continuous(Points::from_host(&tangents, n, d, &host_target)),
                );
            let (_, gradients) = program.gradient(&at)?;
            let gradient = frame(&gradients, tangent.node)?;
            for (m, &f) in members.iter().enumerate() {
                let root = &roots[m];
                let size = root.len() / pass.rows.max(1);
                // The root's rows: the residual's `d`, then the scale's `d(d + 1)/2`.
                let height = d + d * (d + 1) / 2;
                let mut c = Vec::with_capacity(pass.rows * d);
                for row in 0..pass.rows {
                    let entries = &root[row * size..(row + 1) * size];
                    // `C` is stored column by column: entry (i, j) at `j × height + i`.
                    let w = |i: usize, j: usize| entries[j * height + i];
                    let g = &gradient[(m * pass.rows + row) * d..(m * pass.rows + row + 1) * d];
                    // `Wᵀ ĉ = g`, `W` lower triangular: back substitution.
                    let mut x = vec![0.0; d];
                    for i in (0..d).rev() {
                        let rest: f64 = (i + 1..d).map(|j| w(j, i) * x[j]).sum();
                        x[i] = (g[i] - rest) / w(i, i);
                    }
                    c.extend(x);
                }
                folds[f] = Some(c);
            }
        }
        Ok(folds)
    }

    /// The chart linearization at `point` over `quantities`, each of whose spaces must have a
    /// volume (`NoVolume`): the linear term less `∇λ_p̂(0)` and the constant less `λ_p̂(0)`
    /// (`NotAnIntegral` of the Fisher assembly, `NotFree` for a quantity that is not free).
    /// Whether the linear term is then present is decided by the types: it is, if it was, or if a
    /// quantity's chart volume reads its point, or has a gradient at zero, a constant of its type
    /// when it does not read the point, that is not zero.
    pub fn at_in_chart(&self, point: &Values, quantities: &[NodeId]) -> Result<Linearized, Error> {
        if self.assembly == Assembly::Fisher {
            return Err(Error::NotAnIntegral);
        }
        let layout = self.layout();
        let mut blocks = Vec::with_capacity(quantities.len());
        for &q in quantities {
            let k = layout
                .quantities()
                .iter()
                .position(|&p| p == q)
                .ok_or(Error::NotFree { quantity: q })?;
            let space = self
                .snapshot
                .space(q)
                .cloned()
                .ok_or(Error::NotFree { quantity: q })?;
            if !space.has_measure() {
                return Err(Error::Core(skoro_core::Error::NoVolume {
                    space,
                    quantity: Some(q),
                }));
            }
            blocks.push((q, k, space));
        }
        let mut linearized = self.at(point)?;
        let rows = linearized.problems.len();
        let host_target = Target::host();
        // Each quantity's `λ_p̂(0)` and `∇λ_p̂(0)`, row by row.
        let mut charts = Vec::with_capacity(blocks.len());
        let mut present = linearized
            .problems
            .first()
            .is_some_and(|p| p.linear().is_some());
        for (q, k, space) in &blocks {
            let program = space.log_volume()?.compile(&host_target)?;
            let free = program.free();
            let (Some(at), Some(tangent)) = (free.first(), free.get(1)) else {
                continue;
            };
            let d = self.structure.blocks()[*k];
            let values = host(point, *q)?;
            let inputs = Values::new()
                .with(
                    at.node,
                    Value::Continuous(Points::from_host(
                        &values,
                        rows,
                        space.storage(),
                        &host_target,
                    )),
                )
                .with(
                    tangent.node,
                    Value::Continuous(Points::from_host(
                        &vec![0.0; rows * d],
                        rows,
                        d,
                        &host_target,
                    )),
                );
            let (out, gradients) = program.gradient(&inputs)?;
            let value = host(&out, program.targets()[0])?;
            let gradient = frame(&gradients, tangent.node)?;
            // A volume that does not read its point has the same gradient at zero everywhere: a
            // constant of the type, read off the first row.
            let constant_zero = gradient.iter().take(d).all(|&g| g == 0.0);
            present |= space.volume_reads_point()? || !constant_zero;
            charts.push((*k, value, gradient));
        }
        for (row, problem) in linearized.problems.iter_mut().enumerate() {
            let mut linear = problem.linear().map(<[f64]>::to_vec);
            if present && linear.is_none() {
                linear = Some(vec![0.0; self.structure.dimension()]);
            }
            let mut constant = problem.constant();
            for (k, value, gradient) in &charts {
                constant -= value[row];
                if let Some(eta) = &mut linear {
                    let range = self.structure.range(*k);
                    let d = range.len();
                    for (e, g) in eta[range].iter_mut().zip(&gradient[row * d..(row + 1) * d]) {
                        *e -= g;
                    }
                }
            }
            *problem = Problem::new(
                problem.structure().clone(),
                problem.factors().to_vec(),
                linear,
                constant,
                problem.roundoff(),
            )?;
        }
        Ok(linearized)
    }

    /// The quantities of a curved type among the free ones.
    pub(crate) fn curved_quantities(&self) -> Vec<NodeId> {
        self.layout()
            .quantities()
            .iter()
            .copied()
            .filter(|&q| self.snapshot.space(q).is_some_and(curved))
            .collect()
    }
}

/// The node handles a linearizer reads, for the targets of its rows program.
pub(crate) fn targets(nodes: impl IntoIterator<Item = QDyn>) -> Vec<QDyn> {
    let mut targets: Vec<QDyn> = nodes.into_iter().collect();
    targets.sort_unstable_by_key(|q| q.id());
    targets.dedup();
    targets
}
