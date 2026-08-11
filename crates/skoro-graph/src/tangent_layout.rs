//! A contiguous ordering of the tangent spaces of a set of quantities.
//!
//! Inference algorithms want dense matrices — a covariance `P`, a Jacobian `J`, a
//! correction `δ` — while the graph is keyed by [`Id`]. A [`TangentLayout`] is the
//! translation: it assigns each quantity a contiguous range of indices, once, so that
//! flattening and unflattening are unambiguous and reproducible.

use std::collections::HashMap;

use skoro_backend::{Backend, CotangentFrame, TangentVector, Tensor};
use skoro_core::{
    error::{SkoroError, SkoroResult},
    id::Id,
};

use crate::graph::Graph;

/// One quantity's slot in a [`TangentLayout`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TangentBlock {
    pub id: Id,
    /// Index of this block's first column in the flattened vector.
    pub offset: usize,
    /// The quantity's *manifold* dimension.
    pub dimension: usize,
}

/// A contiguous ordering of the tangent spaces of a set of quantities.
///
/// Laid out by **manifold dimension**, never by storage size: `SO3` occupies 3 slots and
/// `SE3` occupies 6, not 9 and 12. Every shape downstream depends on this — it is what
/// lets a filter over `SE3` do its arithmetic in ℝ⁶ with no special casing, leaving the
/// retraction as the only manifold-aware step.
///
/// Ordering is caller-specified by design: a block structure the caller chose is one they
/// can reason about when reading a covariance or a Jacobian.
///
/// A layout holds no tensors and carries no backend parameter, so it is cheap to clone
/// and store. It is **not** revalidated against a graph after construction — reusing one
/// after [`Graph::remove`] is a caller error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TangentLayout {
    blocks: Vec<TangentBlock>,
    index: HashMap<Id, usize>,
    total: usize,
}

impl TangentLayout {
    /// Build a layout over `ids`, preserving the caller's ordering.
    ///
    /// Errors [`SkoroError::NotContinuous`] for a discrete id, [`SkoroError::NodeNotFound`]
    /// for an unknown one, and [`SkoroError::DuplicateLayoutId`] if an id appears twice.
    pub fn from_ids<B: Backend>(graph: &Graph<B>, ids: &[Id]) -> SkoroResult<Self> {
        let mut blocks = Vec::with_capacity(ids.len());
        let mut index = HashMap::with_capacity(ids.len());
        let mut offset = 0;
        for &id in ids {
            if index.contains_key(&id) {
                return Err(SkoroError::DuplicateLayoutId { id });
            }
            let dimension = graph.get_manifold(id)?.dimension();
            index.insert(id, blocks.len());
            blocks.push(TangentBlock {
                id,
                offset,
                dimension,
            });
            offset += dimension;
        }
        Ok(Self {
            blocks,
            index,
            total: offset,
        })
    }

    /// The same ids in ascending [`Id`] order, for callers who do not care which.
    pub fn from_ids_sorted<B: Backend>(graph: &Graph<B>, ids: &[Id]) -> SkoroResult<Self> {
        let mut sorted = ids.to_vec();
        sorted.sort();
        Self::from_ids(graph, &sorted)
    }

    /// The summed manifold dimension — the width of a flattened tangent vector.
    pub fn total(&self) -> usize {
        self.total
    }

    /// The blocks, in layout order.
    pub fn blocks(&self) -> &[TangentBlock] {
        &self.blocks
    }

    /// The block belonging to `id`, if it is in this layout.
    pub fn block(&self, id: Id) -> Option<TangentBlock> {
        self.index.get(&id).map(|&i| self.blocks[i])
    }

    /// Whether `id` has a block here.
    pub fn contains(&self, id: Id) -> bool {
        self.index.contains_key(&id)
    }

    /// The ids, in layout order.
    pub fn ids(&self) -> impl Iterator<Item = Id> + '_ {
        self.blocks.iter().map(|block| block.id)
    }

    /// Whether two layouts describe the same tangent space over different quantities:
    /// equal block dimensions, in order.
    ///
    /// This is the rule a sequential estimator pairs `state` and `state_next` with. It
    /// deliberately ignores ids, since the whole point is that the two layouts name
    /// different quantities.
    pub fn same_shape(&self, other: &Self) -> bool {
        self.blocks.len() == other.blocks.len()
            && std::iter::zip(&self.blocks, &other.blocks).all(|(a, b)| a.dimension == b.dimension)
    }

    /// Check a flattened tensor's width against this layout.
    fn check_width(&self, found: usize) -> SkoroResult<()> {
        if found == self.total {
            Ok(())
        } else {
            Err(SkoroError::LayoutDimensionMismatch {
                expected: self.total,
                found,
            })
        }
    }
}

impl TangentLayout {
    /// Gather per-quantity tangents into one `[batch, N]` tensor.
    ///
    /// Ids missing from `map` are filled with zeros rather than erroring: a VJP
    /// legitimately returns no cotangent for a leaf that does not reach the target, and
    /// the right answer there is a zero column, not a failure.
    pub fn flatten_vectors<B: Backend>(
        &self,
        map: &HashMap<Id, TangentVector<B>>,
        batch: usize,
        device: &B::Device,
    ) -> SkoroResult<Tensor<B, 2>> {
        let mut parts = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let part = match map.get(&block.id) {
                Some(tangent) => {
                    let dims = tangent.dims();
                    if dims != [batch, block.dimension] {
                        return Err(SkoroError::BatchSizeMismatch {
                            id: block.id,
                            expected: batch,
                            found: dims[0],
                        });
                    }
                    tangent.clone()
                }
                None => Tensor::zeros([batch, block.dimension], device),
            };
            parts.push(part);
        }
        Ok(Tensor::cat(parts, 1))
    }

    /// Split a `[batch, N]` tensor back into per-quantity tangents.
    pub fn unflatten_vector<B: Backend>(
        &self,
        flat: Tensor<B, 2>,
    ) -> SkoroResult<HashMap<Id, TangentVector<B>>> {
        self.check_width(flat.dims()[1])?;
        Ok(self
            .blocks
            .iter()
            .map(|block| {
                let slice = flat
                    .clone()
                    .slice_dim(1, block.offset..block.offset + block.dimension);
                (block.id, slice)
            })
            .collect())
    }

    /// Gather per-quantity cotangent frames into one `[batch, N, frames]` tensor.
    ///
    /// Missing ids are zero-filled, for the same reason as [`Self::flatten_vectors`].
    pub fn flatten_frames<B: Backend>(
        &self,
        map: &HashMap<Id, CotangentFrame<B>>,
        batch: usize,
        frames: usize,
        device: &B::Device,
    ) -> SkoroResult<Tensor<B, 3>> {
        let mut parts = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let part = match map.get(&block.id) {
                Some(frame) => {
                    let dims = frame.dims();
                    if dims != [batch, block.dimension, frames] {
                        return Err(SkoroError::InconsistentFrameCount {
                            expected: frames,
                            found: dims[2],
                        });
                    }
                    frame.clone()
                }
                None => Tensor::zeros([batch, block.dimension, frames], device),
            };
            parts.push(part);
        }
        Ok(Tensor::cat(parts, 1))
    }

    /// Split a `[batch, N, frames]` tensor back into per-quantity cotangent frames.
    pub fn unflatten_frame<B: Backend>(
        &self,
        flat: Tensor<B, 3>,
    ) -> SkoroResult<HashMap<Id, CotangentFrame<B>>> {
        self.check_width(flat.dims()[1])?;
        Ok(self
            .blocks
            .iter()
            .map(|block| {
                let slice = flat
                    .clone()
                    .slice_dim(1, block.offset..block.offset + block.dimension);
                (block.id, slice)
            })
            .collect())
    }

    /// Per-id blocks of the `N × N` identity, where `N = self.total()`: block `j` is rows
    /// `o_j .. o_j + d_j`, shape `[batch, d_j, N]`.
    ///
    /// Both Jacobian paths seed through this one helper, which is what keeps their index
    /// conventions from drifting apart — the JVP seeds columns and the VJP seeds rows, and
    /// a disagreement between the two would surface as a silently transposed `J`.
    pub fn identity_frames<B: Backend>(
        &self,
        batch: usize,
        device: &B::Device,
    ) -> HashMap<Id, Tensor<B, 3>> {
        let identity: Tensor<B, 2> = Tensor::eye(self.total, device);
        self.blocks
            .iter()
            .map(|block| {
                let rows = identity
                    .clone()
                    .slice_dim(0, block.offset..block.offset + block.dimension);
                (block.id, rows.unsqueeze_dim::<3>(0).repeat_dim(0, batch))
            })
            .collect()
    }
}
