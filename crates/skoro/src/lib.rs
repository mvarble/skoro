//! # Skoro
//!
//! Skoro is a probabilistic modeling library designed for speed, modularity, and extensibility.
//! It bridges hierarchical probabilistic modeling, a bespoke automatic-differentiation engine
//! (first- and second-order, manifold-aware), and high-performance tensor execution on
//! [`burn`](https://burn.dev) backends (CPU via `ndarray`, GPU via `cubecl`).
//!
//! This is the top-level umbrella crate. It re-exports every workspace crate
//! under its short name and provides a curated [`prelude`] with the types most
//! users need to build and evaluate models.
//!
//! ## Quick orientation
//!
//! - Build a model through a [`ModelClient`](skoro_model::model::ModelClient)
//!   (or the Gaussian-restricted
//!   [`GaussianModelClient`](skoro_normal::facade::GaussianModelClient)).
//! - Quantities are typed proxies [`Q<B, M>`](skoro_model::q::Q) over a manifold
//!   `M` (e.g. [`Real`](skoro_riemann::real::Real),
//!   [`Euclidean`](skoro_riemann::euclidean::Euclidean),
//!   [`SO3`](skoro_so3::quantity::SO3)) or a discrete type
//!   (e.g. [`Categorical`](skoro_core::discrete::Categorical)).
//! - Operations on quantities are exposed through `*Ops` extension traits and
//!   `std::ops` operator overloads.
//! - Recoverable errors are reported via
//!   [`SkoroError`](skoro_core::error::SkoroError) /
//!   [`SkoroResult`](skoro_core::error::SkoroResult).
//!
//! ```
//! use skoro::prelude::*;
//!
//! type B = NdArrayBackend<f32, i32>;
//!
//! let model = ModelClient::<B>::default();
//! let a = model.add_parameter(Real);
//! let b = model.add_constant(Real, &2.0);
//! let _c = a + b; // operator sugar builds a mapped quantity
//! ```

// ---------------------------------------------------------------------------
// Crate re-exports (each workspace crate under its short name).
// ---------------------------------------------------------------------------
pub use skoro_backend as backend;
pub use skoro_core as core;
pub use skoro_graph as graph;
pub use skoro_linalg as linalg;
pub use skoro_model as model;
pub use skoro_normal as normal;
pub use skoro_operation as operation;
pub use skoro_pointwise as pointwise;
pub use skoro_riemann as riemann;
pub use skoro_se3 as se3;
pub use skoro_so3 as so3;

/// The curated prelude: import everything most model-building code needs.
///
/// ```
/// use skoro::prelude::*;
/// ```
pub mod prelude {
    // ----- backends & tensor essentials -----
    pub use skoro_backend::{Backend, CubeBackend, NdArrayBackend, Point, Tensor, TensorData};

    // ----- errors -----
    pub use skoro_core::error::{SkoroError, SkoroResult};

    // ----- kind-erased values (the unified value map threaded through evaluation) -----
    pub use skoro_core::kind::{Continuous, Discrete, Kind, QuantityKind};
    pub use skoro_graph::point_map::{PointMap, points};
    pub use skoro_operation::{Input, ValueMap};

    // ----- the model interface -----
    pub use skoro_model::ModelClientOps;
    pub use skoro_model::model::{ModelClient, ModelServer};
    pub use skoro_model::q::Q;

    // ----- distributions -----
    pub use skoro_core::sampler::Sampler;
    pub use skoro_model::distribution::Distribution;
    pub use skoro_normal::distribution::{Normal, NormalSampler};

    // ----- Gaussian facades -----
    pub use skoro_normal::facade::{
        GaussianModelClient, GaussianModelClientOps, GaussianModelServer,
        constant_covariance_diagonal, constant_covariance_identity,
    };

    // ----- manifolds / quantity types -----
    pub use skoro_graph::quantity::{ContinuousQuantity, DiscreteQuantity, QuantityType};
    pub use skoro_riemann::Manifold;
    pub use skoro_riemann::euclidean::{
        Euclidean, Euclidean1, Euclidean2, Euclidean3, Euclidean4, Euclidean5, Euclidean6,
        EuclideanD, VectorD,
    };
    pub use skoro_riemann::matrix::{EuclideanMatrix, EuclideanMatrixD};
    pub use skoro_riemann::real::Real;

    // discrete quantity types
    pub use skoro_core::discrete::{Boolean, Categorical, Integer};

    // ----- linear algebra & pointwise op traits (operator/method sugar on Q) -----
    pub use skoro_linalg::matrix::{MatmulOps, TransposeOps};
    pub use skoro_linalg::tril::{Tril, TrilOps};
    pub use skoro_linalg::vector::{CrossProductOps, LinalgVectorOps};
    pub use skoro_pointwise::PointwiseOps;

    // ----- discrete op traits / bridges -----
    pub use skoro_model::discrete::DiscreteCompare;
    pub use skoro_model::discrete::bridge::{DiscreteEmbed, Discretize};

    // ----- SO(3) / SE(3) -----
    pub use skoro_se3::quantity::SE3;
    pub use skoro_so3::ops::{SO3ComposeOps, SO3ExpOps, SO3InvertOps, SO3LogOps, SO3RotateOps};
    pub use skoro_so3::quantity::SO3;
}
