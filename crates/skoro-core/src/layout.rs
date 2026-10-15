use std::sync::{Arc, Mutex, PoisonError};

use skoro_tensor::burn::Tensor;
use skoro_tensor::{Frame, Points, Target, Value};

use crate::{Block, Error, Frames, NodeId, Program, Snapshot, Space, Values};

/// The flattened coordinates of some continuous quantities: the product of their spaces.
#[derive(Debug)]
pub struct Layout {
    quantities: Vec<NodeId>,
    product: Space,
    /// The target frames for the layout are made on: a program's, or the host's.
    target: Target,
    /// The product's retraction, compiled once per target.
    retractions: Mutex<Vec<(Target, Arc<Program>)>>,
}

impl Clone for Layout {
    fn clone(&self) -> Layout {
        Layout::of(
            self.quantities.clone(),
            self.product.clone(),
            self.target.clone(),
        )
    }
}

impl Layout {
    /// The layout of `quantities`, each continuous (`NotContinuous`) and in the snapshot
    /// (`UnknownNode`).
    pub fn new<Q: Into<NodeId> + Copy>(
        snapshot: &Snapshot,
        quantities: &[Q],
    ) -> Result<Layout, Error> {
        let ids: Vec<NodeId> = quantities.iter().map(|&q| q.into()).collect();
        let spaces = ids
            .iter()
            .map(|&id| snapshot.space(id).cloned().ok_or(Error::UnknownNode { id }))
            .collect::<Result<Vec<Space>, Error>>()?;
        Ok(Layout::of(ids, Space::product(spaces)?, Target::host()))
    }

    pub(crate) fn of(quantities: Vec<NodeId>, product: Space, target: Target) -> Layout {
        Layout {
            quantities,
            product,
            target,
            retractions: Mutex::new(Vec::new()),
        }
    }

    /// The quantities, in the order of the product's factors.
    pub fn quantities(&self) -> &[NodeId] {
        &self.quantities
    }

    pub fn product(&self) -> &Space {
        &self.product
    }

    pub fn dimension(&self) -> usize {
        self.product.dimension().unwrap_or(0)
    }

    /// The product's blocks: each quantity's ranges in storage and in tangent coordinates.
    pub fn blocks(&self) -> Vec<Block> {
        self.product.blocks()
    }

    pub(crate) fn target(&self) -> &Target {
        &self.target
    }

    /// The step of an estimator: the quantities' values stacked into one point of the product,
    /// moved by the product's retraction along `tangent`, `[rows, dimension]`, on `target`, and
    /// sliced back.
    pub fn retract(&self, at: &Values, tangent: &Points, target: &Target) -> Result<Values, Error> {
        let parts = self
            .quantities
            .iter()
            .map(|&id| match at.get(id) {
                Some(Value::Continuous(points)) => Ok(points.clone().into_burn()),
                Some(Value::Discrete(_)) => Err(Error::NotContinuous {
                    space: self.product.clone(),
                }),
                None => Err(Error::MissingValue { id }),
            })
            .collect::<Result<Vec<Tensor<2>>, Error>>()?;
        if tangent.storage() != self.dimension() {
            return Err(Error::ValueShape {
                space: Space::Vector(self.dimension()),
            });
        }
        let point = Points::from_burn(Tensor::cat(parts, 1), target)?;
        let program = self.retraction(target)?;
        let inputs = program.free();
        let (Some(x), Some(t)) = (inputs.first(), inputs.get(1)) else {
            unreachable!("a retraction takes a point and a tangent")
        };
        let values = Values::new()
            .with(x.node, Value::Continuous(point))
            .with(t.node, Value::Continuous(tangent.clone()));
        let moved = program.forward(&values)?;
        let Some(Value::Continuous(moved)) =
            program.targets().first().and_then(|&id| moved.get(id))
        else {
            unreachable!("a retraction gives a point of its space")
        };
        let moved = moved.clone().into_burn();
        let rows = moved.dims()[0];
        let mut out = Values::new();
        for (&id, block) in self.quantities.iter().zip(self.blocks()) {
            let part = moved.clone().slice([0..rows, block.storage]);
            out.insert(id, Value::Continuous(Points::from_burn(part, target)?));
        }
        Ok(out)
    }

    fn retraction(&self, target: &Target) -> Result<Arc<Program>, Error> {
        let mut cache = self
            .retractions
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some((_, program)) = cache.iter().find(|(t, _)| t == target) {
            return Ok(program.clone());
        }
        let program = Arc::new(self.product.retraction()?.compile(target)?);
        cache.push((target.clone(), program.clone()));
        Ok(program)
    }

    /// The quantities' frames laid end to end, their rows by `blocks()`: `[rows, dimension,
    /// columns]`, at the layout's target (`MissingValue` for a quantity with no frame, `FrameShape`
    /// for frames that do not stack).
    pub fn flatten(&self, frames: &Frames) -> Result<Frame, Error> {
        let parts = self
            .quantities
            .iter()
            .map(|&id| frames.get(id).cloned().ok_or(Error::MissingValue { id }))
            .collect::<Result<Vec<Frame>, Error>>()?;
        let shape = |f: &Frame| (f.rows(), f.columns());
        if let Some(first) = parts.first()
            && let Some((i, _)) = parts
                .iter()
                .enumerate()
                .find(|(_, f)| shape(f) != shape(first))
        {
            return Err(Error::FrameShape {
                id: self.quantities[i],
            });
        }
        let tensors: Vec<Tensor<3>> = parts.into_iter().map(Frame::into_burn).collect();
        if tensors.is_empty() {
            return Ok(Frame::zeros(1, 0, 1, &self.target));
        }
        Ok(Frame::from_burn(Tensor::cat(tensors, 1), &self.target)?)
    }
}

impl Frames {
    /// The velocity frame whose columns are the coordinate directions of `layout`, one row shared
    /// by every row of a pass: with it, a Hessian pass gives the whole chart Hessian as `β`.
    pub fn identity(layout: &Layout) -> Frames {
        let d = layout.dimension();
        let mut frames = Frames::new();
        for (&id, block) in layout.quantities().iter().zip(layout.blocks()) {
            let data: Vec<f64> = block
                .tangent
                .clone()
                .flat_map(|i| (0..d).map(move |c| if c == i { 1.0 } else { 0.0 }))
                .collect();
            frames.insert(
                id,
                Frame::from_host(&data, 1, block.tangent.len(), d, layout.target()),
            );
        }
        frames
    }
}
