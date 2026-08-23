use std::fmt::Debug;
use std::sync::Arc;

use skoro_backend::{
    Backend, DiscretePoint, Int, Point, Tensor, TensorData, UnbatchedDiscretePoint, UnbatchedPoint,
};
use skoro_core::{
    kind::{Continuous, Discrete, Kind, QuantityKind},
    pack::{Cardinality, NumComponents, Packable},
};
use skoro_operation::Input;
use skoro_riemann::{DynManifold, Manifold};

use crate::operation::{DiffGraphOperation, DynGraphOperation};

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
    /// Pack a concrete value into unbatched point.
    fn pack_value<D>(&self, data: &D) -> K::UnbatchedPoint
    where
        D: 'static,
        Self: Packable<K::Elem, D>;

    /// Pack concrete values into a point.
    fn pack_values<D>(&self, data: &[D]) -> K::Point
    where
        D: 'static,
        Self: Packable<K::Elem, D>;

    /// Erase this quantity type into its dyn-compatible form for node storage.
    fn into_dyn(self) -> Arc<dyn DynQuantityType<B>>;

    /// Get inner tensor of input
    fn get_inner_tensor(&self, input: Input<B>) -> K::Point;
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
    fn pack_value<D>(&self, data: &D) -> UnbatchedPoint<B>
    where
        D: 'static,
        Self: Packable<B::FloatElem, D>,
    {
        let d = self.num_components();
        let mut vec = vec![B::FloatElem::default(); d];
        self.pack(data, 0, &mut vec);
        Tensor::from_data(TensorData::new(vec, [d]), &B::Device::default())
    }

    fn pack_values<D>(&self, data: &[D]) -> Point<B>
    where
        D: 'static,
        Self: Packable<B::FloatElem, D>,
    {
        let batches = data.len();
        let num_components = self.num_components();
        let mut vec = vec![B::FloatElem::default(); batches * num_components];
        for (i, d) in data.iter().enumerate() {
            self.pack(d, i * num_components, &mut vec);
        }
        Tensor::<B, 2>::from_data(
            TensorData::new(vec, [batches, num_components]),
            &Default::default(),
        )
    }

    fn into_dyn(self) -> Arc<dyn DynQuantityType<B>> {
        Arc::new(DynContinuous(self))
    }

    fn get_inner_tensor(&self, input: Input<B>) -> Tensor<B, 2> {
        input.unwrap_float()
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
    fn pack_value<D>(&self, data: &D) -> UnbatchedDiscretePoint<B>
    where
        D: 'static,
        Self: Packable<B::IntElem, D>,
    {
        let d = self.num_components();
        let mut vec = vec![B::IntElem::default(); d];
        self.pack(data, 0, &mut vec);
        Tensor::from_data(TensorData::new(vec, [d]), &B::Device::default())
    }

    fn pack_values<D>(&self, data: &[D]) -> DiscretePoint<B>
    where
        D: 'static,
        Self: Packable<B::IntElem, D>,
    {
        let batches = data.len();
        let num_components = self.num_components();
        let mut vec = vec![B::IntElem::default(); batches * num_components];
        for (i, d) in data.iter().enumerate() {
            self.pack(d, i * num_components, &mut vec);
        }
        Tensor::<B, 2, Int>::from_data(
            TensorData::new(vec, [batches, num_components]),
            &Default::default(),
        )
    }

    fn into_dyn(self) -> Arc<dyn DynQuantityType<B>> {
        Arc::new(DynDiscrete(self))
    }

    fn get_inner_tensor(&self, input: Input<B>) -> Tensor<B, 2, Int> {
        input.unwrap_int()
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

    /// The quantity's inverse-retraction as a type-erased graph operation, if it is continuous.
    ///
    /// It exists so a factor can be built over quantities whose manifolds are only known at run
    /// time -- a joint likelihood over a Markov blanket, where each member contributes its own
    /// tangent block. A runtime product manifold would need exactly the same escape.
    ///
    /// Note that the inverse retraction operation of a `DynManifold` cannot be obtained.
    fn inverse_retraction_op(&self) -> Option<Box<dyn DynGraphOperation<B>>>;

    /// A new handle to this same erased quantity type.
    ///
    /// `Arc::clone` needs a sized receiver, which a `&dyn` is not; a node reusing another's
    /// quantity type -- a frozen anchor beside the quantity it anchors -- goes through here.
    fn clone_arc(&self) -> Arc<dyn DynQuantityType<B>>;

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

    fn inverse_retraction_op(&self) -> Option<Box<dyn DynGraphOperation<B>>> {
        Some(Box::new(DiffGraphOperation(
            <T as Manifold<B>>::inverse_retraction(&self.0),
        )))
    }

    fn clone_arc(&self) -> Arc<dyn DynQuantityType<B>> {
        Arc::new(DynContinuous(self.0.clone()))
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

    fn inverse_retraction_op(&self) -> Option<Box<dyn DynGraphOperation<B>>> {
        None
    }

    fn clone_arc(&self) -> Arc<dyn DynQuantityType<B>> {
        Arc::new(DynDiscrete(self.0.clone()))
    }

    fn cardinality(&self) -> Option<usize> {
        <T as Cardinality>::cardinality(&self.0)
    }
}
