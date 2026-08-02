use std::{collections::HashMap, sync::Arc};

use skoro_backend::{Backend, Point, TangentVector, Tensor};
use skoro_core::{
    error::{SkoroError, SkoroResult},
    id::Id,
};
use skoro_operation::Input;

use crate::quantity::DynQuantityType;

#[derive(Debug, Clone, Default)]
pub struct TensorMapBuilder<B: Backend, const D: usize>(HashMap<Id, Tensor<B, D>>);

impl<B: Backend, const D: usize> From<TensorMapBuilder<B, D>> for HashMap<Id, Tensor<B, D>> {
    fn from(value: TensorMapBuilder<B, D>) -> Self {
        value.0
    }
}

impl<B: Backend, const D: usize> From<HashMap<Id, Tensor<B, D>>> for TensorMapBuilder<B, D> {
    fn from(value: HashMap<Id, Tensor<B, D>>) -> Self {
        Self(value)
    }
}

impl<B: Backend, const D: usize> std::ops::Deref for TensorMapBuilder<B, D> {
    type Target = HashMap<Id, Tensor<B, D>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<B: Backend, const D: usize> std::ops::DerefMut for TensorMapBuilder<B, D> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<B: Backend, const D: usize> AsRef<HashMap<Id, Tensor<B, D>>> for TensorMapBuilder<B, D> {
    fn as_ref(&self) -> &HashMap<Id, Tensor<B, D>> {
        &self.0
    }
}

impl<B: Backend, const D: usize> TensorMapBuilder<B, D> {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn build(self) -> HashMap<Id, Tensor<B, D>> {
        self.into()
    }

    pub fn with(mut self, id: Id, tensor: Tensor<B, D>) -> Self {
        self.insert(id, tensor);
        self
    }
}

pub type TangentVectorMapBuilder<B> = TensorMapBuilder<B, 2>;
pub type CotangentVectorMapBuilder<B> = TensorMapBuilder<B, 2>;
pub type TangentFrameMapBuilder<B> = TensorMapBuilder<B, 3>;
pub type CotangentFrameMapBuilder<B> = TensorMapBuilder<B, 3>;

/// Builder for the unified value map (`HashMap<Id, Input<B>>`) supplied
/// to evaluation entry points. Continuous tensors and discrete tensors both
/// convert into [`Input`] via `Into`.
#[derive(Debug, Clone, Default)]
pub struct PointMapBuilder<B: Backend>(HashMap<Id, Input<B>>);

impl<B: Backend> From<PointMapBuilder<B>> for HashMap<Id, Input<B>> {
    fn from(value: PointMapBuilder<B>) -> Self {
        value.0
    }
}

impl<B: Backend> From<HashMap<Id, Input<B>>> for PointMapBuilder<B> {
    fn from(value: HashMap<Id, Input<B>>) -> Self {
        Self(value)
    }
}

impl<B: Backend> std::ops::Deref for PointMapBuilder<B> {
    type Target = HashMap<Id, Input<B>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<B: Backend> std::ops::DerefMut for PointMapBuilder<B> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<B: Backend> PointMapBuilder<B> {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn build(self) -> HashMap<Id, Input<B>> {
        self.into()
    }

    pub fn with(mut self, id: Id, point: impl Into<Input<B>>) -> Self {
        self.0.insert(id, point.into());
        self
    }
}

pub fn points<B: Backend>() -> PointMapBuilder<B> {
    Default::default()
}

pub fn tangents<B: Backend>() -> TangentVectorMapBuilder<B> {
    Default::default()
}

pub fn cotangents<B: Backend>() -> CotangentVectorMapBuilder<B> {
    Default::default()
}

pub fn cotangent_frames<B: Backend>() -> CotangentFrameMapBuilder<B> {
    Default::default()
}

pub fn tangent_frames<B: Backend>() -> TangentFrameMapBuilder<B> {
    Default::default()
}

pub type PointQuantityRefPair<'a, B> = (&'a Input<B>, &'a Arc<dyn DynQuantityType<B>>);
pub type IdPointQuantityRefTuple<'a, B> = (&'a Id, &'a Input<B>, &'a Arc<dyn DynQuantityType<B>>);

/// A type-tagged map of batched node values, continuous or discrete.
///
/// Values are stored as kind-erased [`Input`]s alongside the quantity
/// type of their node. Manifold operations ([`retract`](PointMap::retract),
/// [`inverse_retract`](PointMap::inverse_retract)) require all entries to be
/// continuous and fail with [`SkoroError::NotContinuous`] otherwise.
#[derive(Debug, Clone, Default)]
pub struct PointMap<B: Backend> {
    points: HashMap<Id, Input<B>>,
    quantity_types: HashMap<Id, Arc<dyn DynQuantityType<B>>>,
}

pub struct WithQuantityTypesIter<'a, B: Backend, I> {
    iterator: I,
    quantity_types: &'a HashMap<Id, Arc<dyn DynQuantityType<B>>>,
}

impl<'a, B: Backend, I: Iterator<Item = (&'a Id, &'a Input<B>)>> Iterator
    for WithQuantityTypesIter<'a, B, I>
{
    type Item = IdPointQuantityRefTuple<'a, B>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some((id, point)) = self.iterator.next() {
            Some((id, point, &self.quantity_types[id]))
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iterator.size_hint()
    }

    fn count(self) -> usize
    where
        Self: Sized,
    {
        self.iterator.count()
    }

    fn last(self) -> Option<Self::Item>
    where
        Self: Sized,
    {
        if let Some((id, point)) = self.iterator.last() {
            Some((id, point, &self.quantity_types[id]))
        } else {
            None
        }
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        if let Some((id, point)) = self.iterator.nth(n) {
            Some((id, point, &self.quantity_types[id]))
        } else {
            None
        }
    }
}

impl<B: Backend> PointMap<B> {
    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    pub fn keys(&self) -> impl Iterator<Item = &Id> {
        self.points.keys()
    }

    pub fn get(&self, id: &Id) -> Option<PointQuantityRefPair<'_, B>> {
        if let Some(point) = self.points.get(id) {
            Some((point, &self.quantity_types[id]))
        } else {
            None
        }
    }

    pub fn remove(&mut self, id: &Id) -> Option<(Input<B>, Arc<dyn DynQuantityType<B>>)> {
        if let Some(point) = self.points.remove(id) {
            Some((point, self.quantity_types.remove(id).unwrap()))
        } else {
            None
        }
    }

    pub fn insert(
        &mut self,
        id: Id,
        point: Input<B>,
        quantity_type: Arc<dyn DynQuantityType<B>>,
    ) -> Option<(Input<B>, Arc<dyn DynQuantityType<B>>)> {
        let pout = self.points.insert(id, point);
        let qout = self.quantity_types.insert(id, quantity_type);
        if let Some(pout) = pout {
            Some((pout, qout.unwrap()))
        } else {
            None
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = IdPointQuantityRefTuple<'_, B>> {
        WithQuantityTypesIter {
            iterator: self.points.iter(),
            quantity_types: &self.quantity_types,
        }
    }

    /// The continuous (float) point at `id`, and its manifold structure.
    fn manifold_point(
        &self,
        id: &Id,
    ) -> SkoroResult<(&Point<B>, &dyn skoro_riemann::DynManifold<B>)> {
        let (point, quantity_type) = self.get(id).ok_or(SkoroError::MissingPoint { id: *id })?;
        let manifold = quantity_type
            .as_manifold()
            .ok_or(SkoroError::NotContinuous { id: *id })?;
        let point = point
            .as_float_opt()
            .ok_or(SkoroError::NotContinuous { id: *id })?;
        Ok((point, manifold))
    }

    pub fn retract_mut(&mut self, tangents: &HashMap<Id, TangentVector<B>>) -> SkoroResult<()> {
        // raise an error if there is a tangent vector without a matching point
        if self.points.len() < tangents.len() {
            for id in tangents.keys() {
                if !self.points.contains_key(id) {
                    return Err(SkoroError::MissingPoint { id: *id });
                }
            }
        }
        // perform retractions
        for id in self.points.keys().copied().collect::<Vec<_>>() {
            let Some(tangent) = tangents.get(&id) else {
                return Err(SkoroError::MissingPoint { id });
            };
            let (point, manifold) = self.manifold_point(&id)?;
            let new_point = manifold.retract(point.clone(), tangent.clone());
            self.points.insert(id, Input::Float(new_point));
        }
        Ok(())
    }

    pub fn retract(&self, tangents: &HashMap<Id, TangentVector<B>>) -> SkoroResult<Self> {
        let mut out = self.clone();
        out.retract_mut(tangents)?;
        Ok(out)
    }

    pub fn inverse_retract(
        &self,
        others: PointMap<B>,
    ) -> SkoroResult<HashMap<Id, TangentVector<B>>> {
        // raise an error if there is a mismatch of points
        if self.points.len() < others.len() {
            for id in others.keys() {
                if !self.points.contains_key(id) {
                    return Err(SkoroError::MissingPoint { id: *id });
                }
            }
        }
        // perform inverse retractions, filling output
        let mut out = HashMap::new();
        for id in self.points.keys() {
            let (point, manifold) = self.manifold_point(id)?;
            let quantity_type = &self.quantity_types[id];
            let Some((other_point, other_quantity_type)) = others.get(id) else {
                return Err(SkoroError::MissingPoint { id: *id });
            };
            if !quantity_type.compare(other_quantity_type.as_any()) {
                return Err(SkoroError::ManifoldMismatch { id: *id });
            }
            let other_point = other_point
                .as_float_opt()
                .ok_or(SkoroError::NotContinuous { id: *id })?;
            out.insert(
                *id,
                manifold.inverse_retract(point.clone(), other_point.clone()),
            );
        }
        Ok(out)
    }
}

impl<B: Backend> AsRef<HashMap<Id, Input<B>>> for PointMap<B> {
    fn as_ref(&self) -> &HashMap<Id, Input<B>> {
        &self.points
    }
}

impl<B: Backend> From<PointMap<B>> for HashMap<Id, Input<B>> {
    fn from(value: PointMap<B>) -> Self {
        value.points
    }
}
