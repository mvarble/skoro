use std::collections::HashMap;

use skoro_backend::{Backend, Tensor};
use skoro_core::{id::Id, kind::Kind, pack::Packable};
use skoro_graph::{point_map::PointMap, quantity::QuantityType};
use skoro_operation::Input;

use crate::q::Q;

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

    pub fn with_tensor(mut self, id: Id, tensor: Tensor<B, D>) -> Self {
        self.insert(id, tensor);
        self
    }

    /// Insert by quantity rather than by raw id, matching
    /// [`InputMapBuilder::insert_data`].
    pub fn with_quantity<T>(mut self, q: &Q<B, T>, tensor: Tensor<B, D>) -> Self {
        self.insert(q.id(), tensor);
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
pub struct InputMapBuilder<B: Backend>(HashMap<Id, Input<B>>);

impl<B: Backend> From<InputMapBuilder<B>> for HashMap<Id, Input<B>> {
    fn from(value: InputMapBuilder<B>) -> Self {
        value.0
    }
}

impl<B: Backend> From<HashMap<Id, Input<B>>> for InputMapBuilder<B> {
    fn from(value: HashMap<Id, Input<B>>) -> Self {
        Self(value)
    }
}

impl<B: Backend> std::ops::Deref for InputMapBuilder<B> {
    type Target = HashMap<Id, Input<B>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<B: Backend> std::ops::DerefMut for InputMapBuilder<B> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<B: Backend> InputMapBuilder<B> {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn build(self) -> HashMap<Id, Input<B>> {
        self.into()
    }

    #[inline(always)]
    pub fn insert_tensor(&mut self, id: Id, point: impl Into<Input<B>>) -> Option<Input<B>> {
        self.0.insert(id, point.into())
    }

    pub fn with_tensor(mut self, id: Id, point: impl Into<Input<B>>) -> Self {
        self.insert_tensor(id, point);
        self
    }

    /// Insert by quantity rather than by raw id, matching [`Self::insert_data`].
    pub fn with_quantity<T>(mut self, q: &Q<B, T>, point: impl Into<Input<B>>) -> Self {
        self.insert_tensor(q.id(), point);
        self
    }

    pub fn insert_data<K, T, D>(&mut self, tag: &Q<B, T>, data: &[D]) -> Option<K::Point>
    where
        K: Kind<B>,
        T: QuantityType<B, K> + Packable<K::Elem, D>,
        D: 'static,
    {
        let quantity_type = tag.get_quantity_type();
        let tensor = quantity_type.pack_values(data);
        let out = self.0.insert(tag.id, K::wrap_point(tensor));
        out.map(|input| K::unwrap_point(input).unwrap())
    }

    #[inline(always)]
    pub fn with_data<K, T, D>(mut self, tag: &Q<B, T>, data: &[D]) -> Self
    where
        K: Kind<B>,
        T: QuantityType<B, K> + Packable<K::Elem, D>,
        D: 'static,
    {
        self.insert_data(tag, data);
        self
    }
}

/// Builder for a [`PointMap`], which pairs each value with its quantity type and is
/// therefore the map that knows how to retract.
///
/// [`InputMapBuilder`] produces a bare [`ValueMap`](skoro_operation::ValueMap), which is
/// what the evaluation entry points take. An optimizer needs the richer map as well, since
/// [`PointMap::retract_flat`] has to reach each block's manifold — and outside
/// `skoro-graph` the only supply of one was a `ForwardOutput`. This closes that gap.
///
/// **What belongs in it.** Every `Parameter` / `Stochastic` leaf the evaluation reaches,
/// free or not. *Observed* quantities do not: an observed node lowers to a compiled
/// constant carrying its own value, so it is held by the graph rather than by this map and
/// survives an `evaluate → retract_flat → evaluate` loop untouched.
#[derive(Debug, Clone, Default)]
pub struct PointMapBuilder<B: Backend>(PointMap<B>);

impl<B: Backend> From<PointMapBuilder<B>> for PointMap<B> {
    fn from(value: PointMapBuilder<B>) -> Self {
        value.0
    }
}

impl<B: Backend> std::ops::Deref for PointMapBuilder<B> {
    type Target = PointMap<B>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<B: Backend> PointMapBuilder<B> {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn build(self) -> PointMap<B> {
        self.into()
    }

    /// Insert by quantity, which is what supplies the quantity type.
    ///
    /// There is no `with_tensor` counterpart: an id alone cannot name a manifold, and a
    /// `PointMap` whose quantity types were guessed would retract wrongly rather than
    /// error.
    pub fn with_quantity<K, T>(mut self, q: &Q<B, T>, point: impl Into<Input<B>>) -> Self
    where
        K: Kind<B>,
        T: QuantityType<B, K>,
    {
        let quantity_type = q.get_quantity_type::<K>();
        self.0
            .insert(q.id(), point.into(), quantity_type.into_dyn());
        self
    }
}

pub fn points<B: Backend>() -> InputMapBuilder<B> {
    Default::default()
}

pub fn point_map<B: Backend>() -> PointMapBuilder<B> {
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
