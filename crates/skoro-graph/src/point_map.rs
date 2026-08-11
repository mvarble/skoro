use std::{collections::HashMap, sync::Arc};

use skoro_backend::{Backend, Point, TangentVector, Tensor};
use skoro_core::{
    error::{SkoroError, SkoroResult},
    id::Id,
};
use skoro_operation::Input;

use crate::{quantity::DynQuantityType, tangent_layout::TangentLayout};

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

    /// Retract only the quantities named by `layout`, leaving every other point untouched.
    ///
    /// Deliberately unlike [`Self::retract_mut`], which requires a tangent for *every*
    /// point in the map and errors [`SkoroError::MissingPoint`] otherwise. An optimizer's
    /// `PointMap` also holds constants and nuisance quantities that are not being stepped,
    /// so partial retraction is the operation it actually needs.
    ///
    /// `delta` is `[batch, layout.total()]`; each block is unflattened and applied through
    /// its own manifold's retraction. Errors [`SkoroError::LayoutDimensionMismatch`] if
    /// the widths disagree, [`SkoroError::MissingPoint`] for a layout id absent here.
    pub fn retract_flat_mut(
        &mut self,
        layout: &TangentLayout,
        delta: Tensor<B, 2>,
    ) -> SkoroResult<()> {
        let tangents = layout.unflatten_vector(delta)?;
        for block in layout.blocks() {
            let tangent = tangents
                .get(&block.id)
                .expect("internal: unflatten yields every block in the layout");
            let (point, manifold) = self.manifold_point(&block.id)?;
            let new_point = manifold.retract(point.clone(), tangent.clone());
            self.points.insert(block.id, Input::Float(new_point));
        }
        Ok(())
    }

    /// [`Self::retract_flat_mut`] on a clone.
    pub fn retract_flat(&self, layout: &TangentLayout, delta: Tensor<B, 2>) -> SkoroResult<Self> {
        let mut out = self.clone();
        out.retract_flat_mut(layout, delta)?;
        Ok(out)
    }

    /// Borrows `others`; it is only read. Symmetric with [`Self::retract`].
    pub fn inverse_retract(
        &self,
        others: &PointMap<B>,
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
