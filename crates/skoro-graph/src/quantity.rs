use std::fmt::Debug;
use std::sync::Arc;

use skoro_backend::{Backend, Int, Tensor, TensorData};
use skoro_core::{
    kind::{Continuous, Discrete, Kind, QuantityKind},
    pack::{Cardinality, NumComponents, Packable},
};
use skoro_operation::Input;
use skoro_riemann::{DynManifold, Manifold};

/// A quantity value, which is either a continuous (float) tensor or a discrete (integer) tensor.
///
/// This is the storage form (rank 1, unbatched) of a quantity's value when the quantity is
/// constant. All functions map (via [`DiffOperation`](skoro_operation::diff::DiffOperation) or
/// [`DiscreteOperation`](skoro_operation::discrete::DiscreteOperation)) over batches of quantity
/// values (via [`Input`]).
#[derive(Clone, Debug)]
pub enum QValue<B: Backend> {
    /// A continuous value packed into the backend float element type.
    Float(Tensor<B, 1>),
    /// A discrete value packed into the backend integer element type.
    Int(Tensor<B, 1, Int>),
}

impl<B: Backend> QValue<B> {
    /// Returns the contained float tensor, panicking if the value is discrete.
    pub fn unwrap_float(self) -> Tensor<B, 1> {
        match self {
            QValue::Float(t) => t,
            QValue::Int(_) => panic!("expected a continuous (float) value, found a discrete one"),
        }
    }

    /// Returns the contained int tensor, panicking if the value is continuous.
    pub fn unwrap_int(self) -> Tensor<B, 1, Int> {
        match self {
            QValue::Int(t) => t,
            QValue::Float(_) => panic!("expected a discrete (int) value, found a continuous one"),
        }
    }

    /// Whether this value is discrete (integer-valued).
    pub fn is_discrete(&self) -> bool {
        matches!(self, QValue::Int(_))
    }

    /// Batch this stored (rank-1) value into a rank-2 [`Input`].
    pub fn expand(&self, batch_size: usize) -> Input<B> {
        match self {
            QValue::Float(t) => Input::Float(t.clone().unsqueeze_dim(0).repeat_dim(0, batch_size)),
            QValue::Int(t) => Input::Int(t.clone().unsqueeze_dim(0).repeat_dim(0, batch_size)),
        }
    }

    /// Reduce a batched (rank-2) [`Input`] to storage form.
    pub fn unbatch(input: Input<B>) -> Self {
        match input {
            Input::Float(t) => QValue::Float(t.squeeze_dim::<1>(0)),
            Input::Int(t) => QValue::Int(t.squeeze_dim::<1>(0)),
        }
    }
}

/// Trait specifying how a quantity of kind `K` is embedded into tensor storage.
///
/// This is the storage/identity layer common to *all* quantities. The kind parameter `K`
/// ([`Continuous`] or [`Discrete`]) selects both the tensor element type values and the APIs the
/// quantity may be used with: continuous quantities participate in automatic differentiation and
/// carry a [`Manifold`] structure; discrete quantities do not.
///
/// Because the two blanket implementations below live on *different trait instantiations*
/// (`QuantityType<B, Continuous>` and `QuantityType<B, Discrete>`), they never conflict, and `K`
/// is inferred at every call site from the concrete quantity type.
pub trait QuantityType<B: Backend, K: Kind<B>>:
    Clone
    + NumComponents
    + std::fmt::Debug
    + std::cmp::PartialEq
    + std::any::Any
    + Send
    + Sync
    + 'static
{
    /// Pack a concrete value into node storage form.
    fn pack_value<D>(&self, data: &D) -> QValue<B>
    where
        D: 'static,
        Self: Packable<K::Elem, D>;

    /// Erase this quantity type into its dyn-compatible form for node storage.
    fn into_dyn(self) -> Arc<dyn DynQuantityType<B>>;
}

/// Blanket implementation: every manifold is a continuous quantity.
impl<B: Backend, T> QuantityType<B, Continuous> for T
where
    T: Manifold<B>
        + Clone
        + NumComponents
        + std::fmt::Debug
        + std::cmp::PartialEq
        + std::any::Any
        + Send
        + Sync
        + 'static,
{
    fn pack_value<D>(&self, data: &D) -> QValue<B>
    where
        D: 'static,
        Self: Packable<B::FloatElem, D>,
    {
        let d = self.num_components();
        let mut vec = vec![B::FloatElem::default(); d];
        self.pack(data, 0, &mut vec);
        QValue::Float(Tensor::from_data(
            TensorData::new(vec, [d]),
            &B::Device::default(),
        ))
    }

    fn into_dyn(self) -> Arc<dyn DynQuantityType<B>> {
        Arc::new(DynContinuous(self))
    }
}

/// Blanket implementation: every object with cardinality and number of components.
impl<B: Backend, T> QuantityType<B, Discrete> for T
where
    T: Cardinality
        + Clone
        + NumComponents
        + std::fmt::Debug
        + std::cmp::PartialEq
        + std::any::Any
        + Send
        + Sync
        + 'static,
{
    fn pack_value<D>(&self, data: &D) -> QValue<B>
    where
        D: 'static,
        Self: Packable<B::IntElem, D>,
    {
        let d = self.num_components();
        let mut vec = vec![B::IntElem::default(); d];
        self.pack(data, 0, &mut vec);
        QValue::Int(Tensor::from_data(
            TensorData::new(vec, [d]),
            &B::Device::default(),
        ))
    }

    fn into_dyn(self) -> Arc<dyn DynQuantityType<B>> {
        Arc::new(DynDiscrete(self))
    }
}

/// A continuous quantity: one that carries a differentiable [`Manifold`] structure (a tangent
/// space, retraction, and inverse-retraction). All quantities that participate in automatic
/// differentiation are continuous.
///
/// This is a compatibility alias for `QuantityType<B, Continuous> + Manifold<B>`.
pub trait ContinuousQuantity<B: Backend>: QuantityType<B, Continuous> + Manifold<B> {}

/// Blanket implementation of [`ContinuousQuantity`].
impl<B: Backend, T: QuantityType<B, Continuous> + Manifold<B>> ContinuousQuantity<B> for T {}

/// A discrete quantity: integer-valued, with no tangent space and no differentiable retraction.
/// Discrete quantities are non-differentiable inputs to the graph; their consumers receive no
/// tangents/cotangents for them.
///
/// This is a compatibility alias for `QuantityType<B, Discrete> + Cardinality`.
pub trait DiscreteQuantity<B: Backend>: QuantityType<B, Discrete> + Cardinality {}

/// Blanket implementation of [`DiscreteQuantity`].
impl<B: Backend, T> DiscreteQuantity<B> for T where T: QuantityType<B, Discrete> + Cardinality {}

/// A dyn-compatible, kind-erased quantity type, as stored in graph nodes.
pub trait DynQuantityType<B: Backend>: std::fmt::Debug + Send + Sync {
    /// The runtime kind discriminator.
    fn kind(&self) -> QuantityKind;
    /// The concrete quantity type, for downcasting.
    fn as_any(&self) -> &dyn std::any::Any;
    /// The number of tensor components a packed value occupies.
    fn num_components(&self) -> usize;
    /// Compare against another type-erased quantity type.
    fn compare(&self, other: &dyn std::any::Any) -> bool;
    /// The manifold structure, if the quantity is continuous.
    fn as_manifold(&self) -> Option<&dyn DynManifold<B>>;
    /// The number of distinct values the quantity may take, if discrete with
    /// finite support. Always `None` for continuous quantities.
    fn cardinality(&self) -> Option<usize>;
}

/// The [`DynQuantityType`] wrapper for continuous (manifold) quantity types.
#[derive(Debug, Clone)]
pub(crate) struct DynContinuous<T>(pub T);

impl<B: Backend, T> DynQuantityType<B> for DynContinuous<T>
where
    T: ContinuousQuantity<B>,
{
    fn kind(&self) -> QuantityKind {
        QuantityKind::Continuous
    }

    fn as_any(&self) -> &dyn std::any::Any {
        &self.0
    }

    fn num_components(&self) -> usize {
        self.0.num_components()
    }

    fn compare(&self, other: &dyn std::any::Any) -> bool {
        other
            .downcast_ref::<T>()
            .map(|other| self.0 == *other)
            .unwrap_or(false)
    }

    fn as_manifold(&self) -> Option<&dyn DynManifold<B>> {
        Some(&self.0)
    }

    fn cardinality(&self) -> Option<usize> {
        None
    }
}

/// The [`DynQuantityType`] wrapper for discrete quantity types.
#[derive(Debug, Clone)]
pub(crate) struct DynDiscrete<T>(pub T);

impl<B: Backend, T> DynQuantityType<B> for DynDiscrete<T>
where
    T: DiscreteQuantity<B>,
{
    fn kind(&self) -> QuantityKind {
        QuantityKind::Discrete
    }

    fn as_any(&self) -> &dyn std::any::Any {
        &self.0
    }

    fn num_components(&self) -> usize {
        self.0.num_components()
    }

    fn compare(&self, other: &dyn std::any::Any) -> bool {
        other
            .downcast_ref::<T>()
            .map(|other| self.0 == *other)
            .unwrap_or(false)
    }

    fn as_manifold(&self) -> Option<&dyn DynManifold<B>> {
        None
    }

    fn cardinality(&self) -> Option<usize> {
        <T as Cardinality>::cardinality(&self.0)
    }
}
