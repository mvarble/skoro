use skoro_backend::{Backend, CotangentFrame, Element, Point, Tensor};
use skoro_core::pack::{NumComponents, Packable};
use skoro_operation::{
    Input,
    diff::{DiffModeState, DiffOperation, DiffOperationForward},
    ops::Addition,
    sum_opt_tensors,
};

use crate::Manifold;

/// A vector with a runtime dimension.
#[derive(Clone, PartialEq)]
pub struct VectorD<F: Element>(ndarray::Array1<F>);

impl<F: Element> std::fmt::Debug for VectorD<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple(&format!("VectorD({})", self.len()))
            .field(&self.0.to_vec())
            .finish_non_exhaustive()
    }
}

impl<F: Element> VectorD<F> {
    #[inline]
    pub fn new(elems: &[F]) -> Self {
        Self::from_vec(elems.to_vec())
    }

    #[inline]
    pub fn from_vec(elems: Vec<F>) -> Self {
        Self(ndarray::Array1::from_vec(elems))
    }

    #[inline]
    pub fn slice(&self) -> &[F] {
        self.0.as_slice().unwrap()
    }

    #[inline]
    pub fn slice_mut(&mut self) -> &mut [F] {
        self.0.as_slice_mut().unwrap()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }
}

impl<F: Element> FromIterator<F> for VectorD<F> {
    fn from_iter<I: IntoIterator<Item = F>>(iter: I) -> Self {
        Self(ndarray::Array1::from_iter(iter))
    }
}

/// A vector with a static dimension.
#[derive(Clone, Copy, PartialEq)]
pub struct Vector<const D: usize, F: Element>([F; D]);

impl<const D: usize, F: Element> std::fmt::Debug for Vector<D, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple(&format!("Vector<{D}>"))
            .field(&self.0.to_vec())
            .finish_non_exhaustive()
    }
}

pub type Vector1<F> = Vector<1, F>;
pub type Vector2<F> = Vector<2, F>;
pub type Vector3<F> = Vector<3, F>;
pub type Vector4<F> = Vector<4, F>;
pub type Vector5<F> = Vector<5, F>;
pub type Vector6<F> = Vector<6, F>;

impl<const D: usize, F: Element> Vector<D, F> {
    #[inline]
    pub fn new(elems: [F; D]) -> Self {
        Self(elems)
    }

    #[inline]
    pub fn slice(&self) -> &[F; D] {
        &self.0
    }

    #[inline]
    pub fn slice_mut(&mut self) -> &mut [F; D] {
        &mut self.0
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        false
    }

    #[inline]
    pub fn len(&self) -> usize {
        D
    }
}

/// Euclidean manifold operations of a specific runtime dimension.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EuclideanD(pub usize);

impl<B: Backend> Manifold<B> for EuclideanD {
    fn dimension(&self) -> usize {
        self.0
    }

    fn retraction(&self) -> impl DiffOperation<B> {
        Addition
    }

    fn inverse_retraction(&self) -> impl DiffOperation<B> {
        EuclideanInverseRetraction
    }
}

impl NumComponents for EuclideanD {
    fn num_components(&self) -> usize {
        self.0
    }
}

impl<F: Element> Packable<F, VectorD<F>> for EuclideanD {
    fn pack(&self, data: &VectorD<F>, offset: usize, target: &mut [F]) {
        assert_eq!(data.len(), self.0);
        target[offset..(offset + self.0)].copy_from_slice(data.slice());
    }
    fn unpack(&self, source: &[F], offset: usize) -> VectorD<F> {
        VectorD::<F>::new(&source[offset..(offset + self.0)])
    }
}

/// Euclidean manifold operations of a specific static dimension.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Euclidean<const D: usize>;

pub type Euclidean1 = Euclidean<1>;
pub type Euclidean2 = Euclidean<2>;
pub type Euclidean3 = Euclidean<3>;
pub type Euclidean4 = Euclidean<4>;
pub type Euclidean5 = Euclidean<5>;
pub type Euclidean6 = Euclidean<6>;

impl<const D: usize, B: Backend> Manifold<B> for Euclidean<D> {
    fn dimension(&self) -> usize {
        D
    }

    fn retraction(&self) -> impl DiffOperation<B> {
        Addition
    }

    fn inverse_retraction(&self) -> impl DiffOperation<B> {
        EuclideanInverseRetraction
    }
}

impl<const D: usize> NumComponents for Euclidean<D> {
    fn num_components(&self) -> usize {
        D
    }
}

impl<const D: usize, F: Element> Packable<F, Vector<D, F>> for Euclidean<D> {
    fn pack(&self, data: &Vector<D, F>, offset: usize, target: &mut [F]) {
        target[offset..(offset + D)].copy_from_slice(data.slice());
    }
    fn unpack(&self, source: &[F], offset: usize) -> Vector<D, F> {
        let mut out = Vector::<D, F>::new([F::default(); D]);
        out.slice_mut()
            .copy_from_slice(&source[offset..(offset + D)]);
        out
    }
}

/// Subtraction with the components reversed, `(x, y) => y - x`.
#[derive(Debug, Clone, Copy)]
pub struct EuclideanInverseRetraction;

impl<B: Backend> DiffOperation<B> for EuclideanInverseRetraction {
    type VjpState = [bool; 2];
    type HvpState = ();

    fn forward(
        &self,
        points: Vec<Input<B>>,
        diff_state: DiffModeState<B>,
    ) -> DiffOperationForward<B, Self::VjpState, Self::HvpState> {
        let points: Vec<Point<B>> = points.into_iter().map(Input::into_float).collect();
        let [y, x] = points.try_into().unwrap();
        match diff_state {
            DiffModeState::ForwardOnly => DiffOperationForward::forward_only(x - y),
            DiffModeState::Jvp(tangents) => {
                let [dy, dx] = tangents.try_into().unwrap();
                DiffOperationForward::jvp(x - y, sum_opt_tensors([dy.map(|dy| -dy), dx]))
            }
            DiffModeState::Vjp(diffs) => {
                DiffOperationForward::vjp(x - y, diffs.try_into().unwrap())
            }
            DiffModeState::Hvp {
                diffs,
                tangents: tvecs,
            } => {
                let [dy, dx] = tvecs.try_into().unwrap();
                let vjp_state = diffs.try_into().unwrap();
                DiffOperationForward::hvp(
                    x - y,
                    sum_opt_tensors([dy.map(|dy| -dy), dx]),
                    vjp_state,
                    (),
                )
            }
        }
    }

    fn vjp(
        &self,
        state: &Self::VjpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        match state {
            [true, true] => vec![Some(-omega.clone()), Some(omega)],
            [true, false] => vec![Some(-omega), None],
            [false, true] => vec![None, Some(omega)],
            [false, false] => vec![None, None],
        }
    }

    fn hvp(
        &self,
        state: &Self::VjpState,
        _: &Self::HvpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        if state[0] || state[1] {
            let zeros = Tensor::zeros_like(&omega);
            state
                .map(|b| if b { Some(zeros.clone()) } else { None })
                .to_vec()
        } else {
            vec![None, None]
        }
    }
}
