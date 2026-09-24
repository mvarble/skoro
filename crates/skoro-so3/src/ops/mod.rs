use skoro_graph::quantity::ContinuousQuantity;
use skoro_riemann::euclidean::{Euclidean3, EuclideanD};

use crate::quantity::SO3;

mod body;
pub mod compose;
pub mod exp;
pub mod inverse_retraction;
pub mod invert;
pub mod log;
pub mod retraction;
pub mod rotate;

pub use compose::*;
pub use exp::{SO3Exp, SO3ExpBackend, SO3ExpOps, exp};
pub use inverse_retraction::*;
pub use invert::*;
pub use log::{SO3Log, SO3LogBackend, SO3LogOps, log};
pub use retraction::*;
pub use rotate::*;

pub trait SO3Backend:
    skoro_backend::Backend
    + SO3ExpBackend<EuclideanD>
    + SO3ExpBackend<Euclidean3>
    + SO3LogBackend<EuclideanD>
    + SO3LogBackend<Euclidean3>
where
    SO3: ContinuousQuantity<Self>,
{
}

impl<B> SO3Backend for B
where
    SO3: ContinuousQuantity<B>,
    B: skoro_backend::Backend
        + SO3ExpBackend<EuclideanD>
        + SO3ExpBackend<Euclidean3>
        + SO3LogBackend<EuclideanD>
        + SO3LogBackend<Euclidean3>,
{
}
