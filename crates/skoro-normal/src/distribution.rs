use burn::tensor::Distribution as BurnDistribution;
use skoro_core::error::{SkoroError, SkoroResult};
use skoro_riemann::euclidean::EuclideanD;
use std::marker::PhantomData;

use skoro_backend::{Backend, ElementConversion, Point, Tensor};
use skoro_core::{id::Id, kind::Continuous, sampler::Sampler};
use skoro_graph::quantity::ContinuousQuantity;
use skoro_linalg::{
    tril::{Tril, TrilOps, TrilVectorProduct},
    vector::LinalgVectorOps,
};
use skoro_model::{
    distribution::{Distribution, Factor},
    q::Q,
    roles::{FISHER_ROOT, LOCATION, NORMALIZER_PARENT, RESIDUAL, WHITENING},
};
use skoro_operation::{
    Input,
    diff::{DiffModeState, DiffOperation, DiffOperationForward},
};
use skoro_pointwise::PointwiseOps;
use skoro_riemann::DynManifold;

use crate::information::information_root;

/// Sampler associated with a normal distribution.
pub struct NormalSampler<B: Backend, T: ContinuousQuantity<B>> {
    quantity_type: T,
    _phantom: PhantomData<B>,
}

impl<B: Backend, T: ContinuousQuantity<B>> std::fmt::Debug for NormalSampler<B, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NormalSampler")
            .field("quantity_type", &self.quantity_type)
            .finish_non_exhaustive()
    }
}

impl<B: Backend, T: ContinuousQuantity<B>> Clone for NormalSampler<B, T> {
    fn clone(&self) -> Self {
        Self {
            quantity_type: self.quantity_type.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<B: Backend, T: ContinuousQuantity<B>> NormalSampler<B, T> {
    pub fn new(quantity_type: T) -> Self {
        Self {
            quantity_type,
            _phantom: PhantomData,
        }
    }

    pub fn sample(&self, mean: Point<B>, std_dev: Point<B>) -> Point<B> {
        let batch_size = mean.dims()[0];
        let tangent_dim = self.quantity_type.dimension();
        let tangent = Tensor::<B, 2>::random(
            [batch_size, tangent_dim],
            BurnDistribution::Normal(0.0, 1.0),
            &mean.device(),
        );
        let DiffOperationForward::ForwardOnly { output: tangent } = TrilVectorProduct.forward(
            vec![Input::Float(std_dev), Input::Float(tangent)],
            DiffModeState::ForwardOnly,
        ) else {
            panic!("DiffModeState::ForwardOnly should map to DiffOperationForward::ForwardOnly.")
        };
        self.quantity_type.retract(mean, tangent)
    }
}

impl<B: Backend, M: ContinuousQuantity<B>> Sampler<B, Continuous> for NormalSampler<B, M> {
    fn sample(&self, batch_size: usize, parameters: Vec<Input<B>>) -> Point<B> {
        let [mean, std_dev] = parameters.try_into().unwrap();
        let mean = mean.unwrap_float();
        let std_dev = std_dev.unwrap_float();
        assert_eq!(batch_size, mean.dims()[0]);
        self.sample(mean, std_dev)
    }
}

/// Normal distribution.
pub struct Normal<B: Backend, M: ContinuousQuantity<B>> {
    mean: Q<B, M>,
    std_dev: Q<B, Tril>,
    quantity_type: M,
    constant: f64,
    ids: [Id; 2],
    _phantom: PhantomData<B>,
}

impl<B: Backend, M: ContinuousQuantity<B>> std::fmt::Debug for Normal<B, M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Normal")
            .field("mean", &self.mean)
            .field("std_dev", &self.std_dev)
            .finish_non_exhaustive()
    }
}

impl<B: Backend, M: ContinuousQuantity<B>> Clone for Normal<B, M> {
    fn clone(&self) -> Self {
        Self {
            mean: self.mean.clone(),
            ids: self.ids,
            std_dev: self.std_dev.clone(),
            quantity_type: self.mean.get_quantity_type(),
            constant: self.constant,
            _phantom: PhantomData,
        }
    }
}

impl<B: Backend, M: ContinuousQuantity<B>> Normal<B, M>
where
    B::FloatElem: std::ops::Div<B::FloatElem, Output = B::FloatElem>
        + std::ops::Mul<B::FloatElem, Output = B::FloatElem>
        + std::ops::Add<B::FloatElem, Output = B::FloatElem>
        + std::ops::Sub<B::FloatElem, Output = B::FloatElem>
        + std::ops::Neg<Output = B::FloatElem>
        + std::ops::AddAssign<B::FloatElem>,
{
    pub fn new(mean: Q<B, M>, std_dev: Q<B, Tril>) -> Self {
        let quantity_type = mean.get_quantity_type();
        Self {
            ids: [mean.id(), std_dev.id()],
            mean,
            std_dev,
            constant: -0.5 * quantity_type.dimension() as f64 * std::f64::consts::TAU.ln(),
            quantity_type,
            _phantom: PhantomData,
        }
    }

    pub fn mean(&self) -> &Q<B, M> {
        &self.mean
    }

    pub fn std_dev(&self) -> &Q<B, Tril> {
        &self.std_dev
    }

    pub fn constant(&self) -> B::FloatElem {
        B::FloatElem::from_elem(self.constant)
    }
}

impl<B: Backend, M: ContinuousQuantity<B>> Normal<B, M>
where
    B::FloatElem: std::ops::Div<B::FloatElem, Output = B::FloatElem>
        + std::ops::Mul<B::FloatElem, Output = B::FloatElem>
        + std::ops::Add<B::FloatElem, Output = B::FloatElem>
        + std::ops::Sub<B::FloatElem, Output = B::FloatElem>
        + std::ops::Neg<Output = B::FloatElem>
        + std::ops::AddAssign<B::FloatElem>,
{
    /// The whitened residual `r(x) = L⁻¹ (x ⊖ μ)`.
    ///
    /// The negative log-density is `½‖r(x)‖² + n(θ)`, where the normalizer `n` collects
    /// the constant and the `log |L⁻¹|` term. A least-squares consumer minimizes the
    /// first term alone, which is only the same problem when `n` does not vary with the
    /// quantities being optimized — a condition on the covariance factor, not on this
    /// node.
    pub fn whitened_residual(&self, x: &Q<B, M>) -> SkoroResult<Q<B, EuclideanD>> {
        let colored_tangent = self.mean.inverse_retract(x)?;
        self.std_dev.invert()?.vector_product(&colored_tangent)
    }
}

impl<B: Backend, M: ContinuousQuantity<B>> Distribution<B, Continuous> for Normal<B, M>
where
    B::FloatElem: std::ops::Div<B::FloatElem, Output = B::FloatElem>
        + std::ops::Mul<B::FloatElem, Output = B::FloatElem>
        + std::ops::Add<B::FloatElem, Output = B::FloatElem>
        + std::ops::Sub<B::FloatElem, Output = B::FloatElem>
        + std::ops::Neg<Output = B::FloatElem>
        + std::ops::AddAssign<B::FloatElem>,
{
    type Qty = M;
    type Sampler = NormalSampler<B, M>;

    fn quantity_type(&self) -> M {
        self.quantity_type.clone()
    }

    fn sampler(&self) -> Self::Sampler {
        NormalSampler::new(self.quantity_type.clone())
    }

    fn parent_ids(&self) -> &[Id] {
        &self.ids
    }

    /// The factor `log p(x) = constant − ½‖r‖² + log|diag(L⁻¹)|`, with `r = L⁻¹ (x ⊖ μ)`.
    ///
    /// `L⁻¹` is built once and shared by the residual, the normalizer, and the Fisher root. The
    /// factor binds [`RESIDUAL`] to `r`, [`LOCATION`] to `μ`, [`WHITENING`] and
    /// [`NORMALIZER_PARENT`] to `L`, and [`FISHER_ROOT`] to the root over `[x, μ, L]` (see
    /// [`information`](crate::information)).
    ///
    /// Returns [`SkoroError::ShapeMismatch`] if `L` is not sized for the quantity's dimension.
    fn factor(&self, x: &Q<B, M>) -> SkoroResult<Factor<B>> {
        let d = self.quantity_type.dimension();
        let n = self.std_dev.get_quantity_type::<Continuous>().0;
        if n != d {
            return Err(SkoroError::ShapeMismatch {
                expected: d,
                found: n,
            });
        }
        let w = self.std_dev.invert()?;
        let r = w.vector_product(&self.mean.inverse_retract(x)?)?;
        let log_det = w.diag()?.abs()?.log()?.sum()?;
        let density = ((r.norm_squared()? * -0.5)? + self.constant)? + log_det;
        let root = information_root(&w, d)?;
        Ok(Factor::new(density?)
            .bind(RESIDUAL, &r)
            .bind(LOCATION, &self.mean)
            .bind(WHITENING, &self.std_dev)
            .bind(NORMALIZER_PARENT, &self.std_dev)
            .bind(FISHER_ROOT, &root))
    }
}
