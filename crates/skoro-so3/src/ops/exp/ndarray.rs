use burn::Tensor;
use num_traits::{Float, One};
use skoro_backend::{
    Backend, CotangentFrame, NdArrayBackend, NdArrayFloat, NdArrayInt, NdArrayTensor, Point,
    SharedArray, TangentFrame, TangentVector, TensorData,
};
use skoro_linalg::vector::cross_product::ndarray::cross_product_impl as cross_product;
use skoro_operation::{
    Input,
    diff::{DiffModeState, DiffOperation, DiffOperationForward},
};

use super::SO3Exp;

impl<F, I> DiffOperation<NdArrayBackend<F, I>> for SO3Exp
where
    F: NdArrayFloat + Float,
    I: NdArrayInt,
    NdArrayTensor: From<SharedArray<F>> + From<SharedArray<I>>,
{
    type VjpState = Option<Point<NdArrayBackend<F, I>>>;
    type HvpState = Option<TangentVector<NdArrayBackend<F, I>>>;

    fn forward(
        &self,
        inputs: Vec<Input<NdArrayBackend<F, I>>>,
        diff_state: DiffModeState<NdArrayBackend<F, I>>,
    ) -> DiffOperationForward<NdArrayBackend<F, I>, Self::VjpState, Self::HvpState> {
        let inputs: Vec<Point<NdArrayBackend<F, I>>> =
            inputs.into_iter().map(Input::into_float).collect();
        let [omega] = inputs.try_into().unwrap();
        match diff_state {
            DiffModeState::ForwardOnly => DiffOperationForward::forward_only(exp(omega)),
            DiffModeState::Jvp(tvecs) => {
                let [v] = tvecs.try_into().unwrap();
                DiffOperationForward::jvp(exp(omega.clone()), v.map(|v| exp_jvp(omega, v)))
            }
            DiffModeState::Vjp(diffs) => {
                let [is_diff] = diffs.try_into().unwrap();
                DiffOperationForward::vjp(
                    exp(omega.clone()),
                    if is_diff { Some(omega) } else { None },
                )
            }
            DiffModeState::Hvp {
                diffs,
                tangents: tvecs,
            } => {
                let [v] = tvecs.try_into().unwrap();
                DiffOperationForward::hvp(
                    exp(omega.clone()),
                    v.clone()
                        .map(|v| exp_jvp(omega.clone(), v.unsqueeze_dim(2)).squeeze_dim(2)),
                    if diffs[0] { Some(omega) } else { None },
                    v,
                )
            }
        }
    }

    fn vjp(
        &self,
        state: &Self::VjpState,
        alpha: CotangentFrame<NdArrayBackend<F, I>>,
    ) -> Vec<Option<CotangentFrame<NdArrayBackend<F, I>>>> {
        vec![state.clone().map(|omega| exp_vjp(omega, alpha))]
    }

    fn hvp(
        &self,
        vjp_state: &Self::VjpState,
        hvp_state: &Self::HvpState,
        alpha: CotangentFrame<NdArrayBackend<F, I>>,
    ) -> Vec<Option<CotangentFrame<NdArrayBackend<F, I>>>> {
        vec![
            vjp_state
                .as_ref()
                .zip(hvp_state.as_ref())
                .map(|(omega, v)| exp_hvp(omega.clone(), v.clone(), alpha)),
        ]
    }
}

pub fn exp<B>(omega: Point<B>) -> Point<B>
where
    B: Backend,
    B::FloatElem: Float + One,
{
    // unpack data from inputs
    let [batch_size, three] = omega.dims();
    if three != 3 {
        panic!("exp requires a tensor of shape `[b, 3]`");
    }

    // unpack tensor data
    let device = omega.device();
    let omega_data = omega.into_data();
    let omega_slice = omega_data.as_slice::<B::FloatElem>().unwrap();

    // Allocate output memory
    let mut out_vec = Vec::with_capacity(batch_size * 9);

    // run batch-by-batch
    for batch in omega_slice.chunks_exact(3) {
        let w_x = batch[0];
        let w_y = batch[1];
        let w_z = batch[2];
        let th_sq = w_x * w_x + w_y * w_y + w_z * w_z;
        let ExpJacobianCoeffs { c1, c2, .. } = exp_jacobian_coeffs(th_sq);
        let o = <B::FloatElem as One>::one();
        out_vec.push(o - c2 * (w_y * w_y + w_z * w_z));
        out_vec.push(-c1 * w_z + c2 * w_x * w_y);
        out_vec.push(c1 * w_y + c2 * w_x * w_z);
        out_vec.push(c1 * w_z + c2 * w_x * w_y);
        out_vec.push(o - c2 * (w_x * w_x + w_z * w_z));
        out_vec.push(-c1 * w_x + c2 * w_y * w_z);
        out_vec.push(-c1 * w_y + c2 * w_x * w_z);
        out_vec.push(c1 * w_x + c2 * w_y * w_z);
        out_vec.push(o - c2 * (w_x * w_x + w_y * w_y));
    }

    // pack data
    Tensor::from_data(TensorData::new(out_vec, [batch_size, 9]), &device)
}

pub fn exp_jvp<B>(omega: Point<B>, v: TangentFrame<B>) -> TangentFrame<B>
where
    B: Backend,
    B::FloatElem: Float + One,
{
    // unpack data from inputs
    let [batch_size, three] = omega.dims();
    let [obatch_size, othree, frame_size] = v.dims();
    if batch_size != obatch_size || three != 3 || othree != 3 {
        panic!("exp_jvp requires tensors of shapes `[b, 3]` and `[b, 3, f]`");
    }

    // unpack tensor data
    let device = omega.device();
    let omega_data = omega.into_data();
    let omega_slice = omega_data.as_slice::<B::FloatElem>().unwrap();
    let v_data = v.into_data();
    let v_slice = v_data.as_slice::<B::FloatElem>().unwrap();

    // Allocate output memory
    let mut out_vec = Vec::with_capacity(batch_size * 3 * frame_size);

    let batch_iter = omega_slice
        .chunks_exact(3)
        .zip(v_slice.chunks_exact(3 * frame_size));

    for (omega_batch, v_batch) in batch_iter {
        let w_x = omega_batch[0];
        let w_y = omega_batch[1];
        let w_z = omega_batch[2];

        // c2(|ω|), c3(|ω|)
        let th_sq = w_x * w_x + w_y * w_y + w_z * w_z;
        let ExpJacobianCoeffs { c2, c3, .. } = exp_jacobian_coeffs(th_sq);

        // `v_slice` indexes `[b, 3, f]` from the right, so each chunk of size `3 * f` will
        // correspond one batch, with `x` data of length `f`, then `y` of length `f`, then `z` of
        // length `f`; the proceeding lines split these into separate arrays.
        let (v_x, rest) = v_batch.split_at(frame_size);
        let (v_y, v_z) = rest.split_at(frame_size);

        // fill the `x` data first, then the `y`, then the `z`.
        let v_iter = v_x.iter().zip(v_y).zip(v_z);
        out_vec.extend(v_iter.clone().map(|((&v_x, &v_y), &v_z)| {
            let (wxv_x, wxv_y, wxv_z) = cross_product(w_x, w_y, w_z, v_x, v_y, v_z);
            let wxwxv_x = w_y * wxv_z - w_z * wxv_y;
            v_x - c2 * wxv_x + c3 * wxwxv_x
        }));
        out_vec.extend(v_iter.clone().map(|((&v_x, &v_y), &v_z)| {
            let (wxv_x, wxv_y, wxv_z) = cross_product(w_x, w_y, w_z, v_x, v_y, v_z);
            let wxwxv_y = w_z * wxv_x - w_x * wxv_z;
            v_y - c2 * wxv_y + c3 * wxwxv_y
        }));
        out_vec.extend(v_iter.map(|((&v_x, &v_y), &v_z)| {
            let (wxv_x, wxv_y, wxv_z) = cross_product(w_x, w_y, w_z, v_x, v_y, v_z);
            let wxwxv_z = w_x * wxv_y - w_y * wxv_x;
            v_z - c2 * wxv_z + c3 * wxwxv_z
        }));
    }

    // pack data
    Tensor::from_data(
        TensorData::new(out_vec, [batch_size, 3, frame_size]),
        &device,
    )
}

pub fn exp_vjp<B>(omega: Point<B>, alpha: CotangentFrame<B>) -> CotangentFrame<B>
where
    B: Backend,
    B::FloatElem: Float + One,
{
    // unpack data from inputs
    let [batch_size, three] = omega.dims();
    let [obatch_size, othree, frame_size] = alpha.dims();
    if batch_size != obatch_size || three != 3 || othree != 3 {
        panic!("exp_vjp requires tensors of shapes `[b, 3]` and `[b, 3, f]`");
    }

    // unpack tensor data
    let device = omega.device();
    let omega_data = omega.into_data();
    let omega_slice = omega_data.as_slice::<B::FloatElem>().unwrap();
    let a_data = alpha.into_data();
    let a_slice = a_data.as_slice::<B::FloatElem>().unwrap();

    // Allocate output memory
    let mut out_vec = Vec::with_capacity(batch_size * 3 * frame_size);

    let batch_iter = omega_slice
        .chunks_exact(3)
        .zip(a_slice.chunks_exact(3 * frame_size));

    for (omega_batch, a_batch) in batch_iter {
        let w_x = omega_batch[0];
        let w_y = omega_batch[1];
        let w_z = omega_batch[2];

        // c2(|ω|), c3(|ω|)
        let th_sq = w_x * w_x + w_y * w_y + w_z * w_z;
        let ExpJacobianCoeffs { c2, c3, .. } = exp_jacobian_coeffs(th_sq);

        // `a_slice` indexes `[b, 3, f]` from the right, so each chunk of size `3 * f` will
        // correspond one batch, with `x` data of length `f`, then `y` of length `f`, then `z` of
        // length `f`; the proceeding lines split these into separate arrays.
        let (a_x, rest) = a_batch.split_at(frame_size);
        let (a_y, a_z) = rest.split_at(frame_size);

        // fill the `x` data first, then the `y`, then the `z`.
        let a_iter = a_x.iter().zip(a_y).zip(a_z);
        out_vec.extend(a_iter.clone().map(|((&a_x, &a_y), &a_z)| {
            let (wxa_x, wxa_y, wxa_z) = cross_product(w_x, w_y, w_z, a_x, a_y, a_z);
            let wxwxa_x = w_y * wxa_z - w_z * wxa_y;
            a_x + c2 * wxa_x + c3 * wxwxa_x
        }));
        out_vec.extend(a_iter.clone().map(|((&a_x, &a_y), &a_z)| {
            let (wxa_x, wxa_y, wxa_z) = cross_product(w_x, w_y, w_z, a_x, a_y, a_z);
            let wxwxa_y = w_z * wxa_x - w_x * wxa_z;
            a_y + c2 * wxa_y + c3 * wxwxa_y
        }));
        out_vec.extend(a_iter.map(|((&a_x, &a_y), &a_z)| {
            let (wxa_x, wxa_y, wxa_z) = cross_product(w_x, w_y, w_z, a_x, a_y, a_z);
            let wxwxa_z = w_x * wxa_y - w_y * wxa_x;
            a_z + c2 * wxa_z + c3 * wxwxa_z
        }));
    }

    // pack data
    Tensor::from_data(
        TensorData::new(out_vec, [batch_size, 3, frame_size]),
        &device,
    )
}

pub fn exp_hvp<B>(
    omega: Point<B>,
    v: TangentVector<B>,
    alpha: CotangentFrame<B>,
) -> CotangentFrame<B>
where
    B: Backend,
    B::FloatElem: Float + One,
{
    // unpack data from inputs
    let [batch_size, three] = omega.dims();
    let [obatch_size, othree] = v.dims();
    let [oobatch_size, oothree, frame_size] = alpha.dims();
    if batch_size != obatch_size
        || batch_size != oobatch_size
        || three != 3
        || othree != 3
        || oothree != 3
    {
        panic!("exp_vjp requires tensors of shapes `[b, 3]`, `[b, 3]`, and `[b, 3, f]`");
    }

    // unpack tensor data
    let device = omega.device();
    let omega_data = omega.into_data();
    let omega_slice = omega_data.as_slice::<B::FloatElem>().unwrap();
    let v_data = v.into_data();
    let v_slice = v_data.as_slice::<B::FloatElem>().unwrap();
    let a_data = alpha.into_data();
    let a_slice = a_data.as_slice::<B::FloatElem>().unwrap();

    // Allocate output memory
    let mut out_vec = Vec::with_capacity(batch_size * 3 * frame_size);

    let batch_iter = omega_slice
        .chunks_exact(3)
        .zip(v_slice.chunks_exact(3))
        .zip(a_slice.chunks_exact(3 * frame_size));

    for ((omega_batch, v_batch), a_batch) in batch_iter {
        let w_x = omega_batch[0];
        let w_y = omega_batch[1];
        let w_z = omega_batch[2];

        let v_x = v_batch[0];
        let v_y = v_batch[1];
        let v_z = v_batch[2];

        // c2(|ω|), c3(|ω|), c2'(|ω|), c3'(|ω|)
        let th_sq = w_x * w_x + w_y * w_y + w_z * w_z;
        let ExpJacobianCoeffs {
            c2, c3, c2_p, c3_p, ..
        } = exp_jacobian_coeffs(th_sq);

        // `a_slice` indexes `[b, 3, f]` from the right, so each chunk of size `3 * f` will
        // correspond one batch, with `x` data of length `f`, then `y` of length `f`, then `z` of
        // length `f`; the proceeding lines split these into separate arrays.
        let (a_x, rest) = a_batch.split_at(frame_size);
        let (a_y, a_z) = rest.split_at(frame_size);

        // fill the `x` data first, then the `y`, then the `z`.
        let a_iter = a_x.iter().zip(a_y).zip(a_z);
        out_vec.extend(a_iter.clone().map(|((&a_x, &a_y), &a_z)| {
            let (wxa_x, wxa_y, wxa_z) = cross_product(w_x, w_y, w_z, a_x, a_y, a_z);
            let (vxa_x, vxa_y, vxa_z) = cross_product(v_x, v_y, v_z, a_x, a_y, a_z);
            let vxwxa_x = v_y * wxa_z - v_z * wxa_y;
            let wxvxa_x = w_y * vxa_z - w_z * vxa_y;
            let wxwxa_x = w_y * wxa_z - w_z * wxa_y;
            let w_dot_v = w_x * v_x + w_y * v_y + w_z * v_z;
            c2 * vxa_x + c3 * (vxwxa_x + wxvxa_x) + w_dot_v * (c2_p * wxa_x + c3_p * wxwxa_x)
        }));
        out_vec.extend(a_iter.clone().map(|((&a_x, &a_y), &a_z)| {
            let (wxa_x, wxa_y, wxa_z) = cross_product(w_x, w_y, w_z, a_x, a_y, a_z);
            let (vxa_x, vxa_y, vxa_z) = cross_product(v_x, v_y, v_z, a_x, a_y, a_z);
            let vxwxa_y = v_z * wxa_x - v_x * wxa_z;
            let wxvxa_y = w_z * vxa_x - w_x * vxa_z;
            let wxwxa_y = w_z * wxa_x - w_x * wxa_z;
            let w_dot_v = w_x * v_x + w_y * v_y + w_z * v_z;
            c2 * vxa_y + c3 * (vxwxa_y + wxvxa_y) + w_dot_v * (c2_p * wxa_y + c3_p * wxwxa_y)
        }));
        out_vec.extend(a_iter.map(|((&a_x, &a_y), &a_z)| {
            let (wxa_x, wxa_y, wxa_z) = cross_product(w_x, w_y, w_z, a_x, a_y, a_z);
            let (vxa_x, vxa_y, vxa_z) = cross_product(v_x, v_y, v_z, a_x, a_y, a_z);
            let vxwxa_z = v_x * wxa_y - v_y * wxa_x;
            let wxvxa_z = w_x * vxa_y - w_y * vxa_x;
            let wxwxa_z = w_x * wxa_y - w_y * wxa_x;
            let w_dot_v = w_x * v_x + w_y * v_y + w_z * v_z;
            c2 * vxa_z + c3 * (vxwxa_z + wxvxa_z) + w_dot_v * (c2_p * wxa_z + c3_p * wxwxa_z)
        }));
    }

    // pack data
    Tensor::from_data(
        TensorData::new(out_vec, [batch_size, 3, frame_size]),
        &device,
    )
}

// SO(3) has a closed form for the Lie-group exponential map we use as our retraction operation.
// ```
// Exp(ω) = I + (sin|ω|/|ω|) ω^ + ((1 - cos|ω|)/|ω|²) (ω^)²
// ```
// The JVP/VJP of this operation boils down to the calculation of the right-Jacobian of the
// function,
// ```
// J(ω) = I - ((1 - cos|ω|)/|ω|²) ω^ + ((|ω| - sin|ω|)/|ω|³) (ω^)²
// ```
// the unique map which satisfies the following asymptotic.
// ```
// Exp(ω + δ) = Exp(ω) Exp( J(ω)⋅δ + o(δ) )
// ```
// The HVP of this operation requires calculating the sensitivity of `J(ω)` in a direction `v`,
// ```
// DvJ(ω)α =
//     ((1 - cos |ω|) / |ω|²) (v × α)
//     + ((|ω| - sin |ω|) / |ω|³) (v × (ω × α) + ω × (v × α))
//     + (v⊤ ω) (
//         ((|ω| sin |ω| - 2 (1 - cos |ω|)) / |ω|⁴) (ω × α)
//         + ((|ω| (1 - cos |ω|) - 3 (|ω| - sin |ω|)) / |ω|⁵) (ω × (ω × α)))
// ```
// `exp_jacobian_coeffs` computes the scalar coefficients showing up in each of these expressions,
// dependent on θ=|ω|.
// ```
// c1 = sin θ / θ
// c2 = (1 - cos θ) / θ²
// c3 = (θ - sin θ) / θ³
// c2_p = (θ sin θ - 2 (1 - cos θ)) / θ⁴
// c3_p = (θ (1 - cos θ) - 3 (θ - sin θ)) / |θ|⁵
// ```
// ## Evaluating them without losing precision
//
// The closed forms above cancel catastrophically as θ → 0. `θ - sin θ` loses about seven
// significant digits at θ = 1e-3, and `c2_p` / `c3_p` cancel at fourth and fifth order. We
// therefore evaluate a truncated Taylor series below a cutoff and the closed forms above it. Each
// series is in θ², so the input is `θ²` rather than θ (which also makes the branch cheaper).
// ```
// c1   =  1    - θ²/6    + θ⁴/120   - θ⁶/5040    + θ⁸/362880
// c2   =  1/2  - θ²/24   + θ⁴/720   - θ⁶/40320   + θ⁸/3628800
// c3   =  1/6  - θ²/120  + θ⁴/5040  - θ⁶/362880  + θ⁸/39916800
// c2_p = -1/12 + θ²/180  - θ⁴/6720  + θ⁶/453600  - θ⁸/47900160
// c3_p = -1/60 + θ²/1260 - θ⁴/60480 + θ⁶/4989600 - θ⁸/622702080
// ```
// Truncation error is `O(θ¹⁰/11!)`, so the cutoff balancing truncation against round-off is
// `θ² = (11!·ε)^(1/5)`, which is 1.366 for `f32` (θ ≈ 1.17) and 0.0245 for `f64` (θ ≈ 0.157).
// Above those the closed forms have retained enough digits to be the better choice.
//
// A single cutoff cannot work for both widths: the previous `θ² > 1e-6` is right for `f64` and
// leaves `f32` evaluating the closed forms exactly where they are numerically dead. Worst-case
// relative error over θ ∈ (0, π], before and after:
// ```
//                c1        c2        c3       c2_p      c3_p
//   f32 before   6.6e-5    5.7e-2    3.3e-1   6.7e+5    1.1e+7
//   f32 after    6.6e-5    1.6e-7    2.0e-7   9.1e-7    4.6e-6
//   f64 before   1.7e-7    8.3e-8    5.0e-8   1.0e-3    1.7e-2
//   f64 after    3.0e-14   4.2e-15   1.9e-14  1.9e-12   2.6e-11
// ```
// `c1`'s `f32` figure is dominated by the accuracy of `sin θ` near θ = π and is unchanged.
/// The scalar coefficients of `J(ω)` and of its sensitivity, all functions of `θ = |ω|` alone.
///
/// Returned as a struct rather than a tuple because callers want different subsets of it — `exp`
/// needs `c1`/`c2`, the JVP and VJP need `c2`/`c3`, and only the HVP needs the primed pair. Field
/// access keeps each call site self-documenting and makes a mix-up a compile error.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExpJacobianCoeffs<F> {
    /// `sin θ / θ`
    pub c1: F,
    /// `(1 - cos θ) / θ²`
    pub c2: F,
    /// `(θ - sin θ) / θ³`
    pub c3: F,
    /// `c2'(θ) / θ`
    pub c2_p: F,
    /// `c3'(θ) / θ`
    pub c3_p: F,
}

#[inline(always)]
pub fn exp_jacobian_coeffs<F: Float>(theta_sq: F) -> ExpJacobianCoeffs<F> {
    let k = |x: f64| F::from(x).unwrap();

    // `(11!·ε)^(1/5)`, precomputed rather than raised to a fractional power on every call.
    // Anything coarser than `f64` takes the wider Taylor region, which is the safe direction.
    let cutoff_sq = if F::epsilon() > k(1e-10) {
        k(1.3661)
    } else {
        k(0.024513)
    };

    if theta_sq > cutoff_sq {
        let theta = theta_sq.sqrt();
        let sin_th = theta.sin();
        let cos_th = theta.cos();
        let th_sq = theta_sq;
        let th_cubed = th_sq * theta;
        let th_quad = th_sq * th_sq;
        let th_fifth = th_quad * theta;

        ExpJacobianCoeffs {
            c1: sin_th / theta,
            c2: (k(1.0) - cos_th) / th_sq,
            c3: (theta - sin_th) / th_cubed,
            c2_p: (theta * sin_th - k(2.0) * (k(1.0) - cos_th)) / th_quad,
            c3_p: (theta * (k(1.0) - cos_th) - k(3.0) * (theta - sin_th)) / th_fifth,
        }
    } else {
        // Horner in θ², highest power first.
        let horner = |a: [f64; 5]| {
            (((k(a[4]) * theta_sq + k(a[3])) * theta_sq + k(a[2])) * theta_sq + k(a[1])) * theta_sq
                + k(a[0])
        };
        ExpJacobianCoeffs {
            c1: horner([1.0, -1.0 / 6.0, 1.0 / 120.0, -1.0 / 5040.0, 1.0 / 362880.0]),
            c2: horner([
                0.5,
                -1.0 / 24.0,
                1.0 / 720.0,
                -1.0 / 40320.0,
                1.0 / 3628800.0,
            ]),
            c3: horner([
                1.0 / 6.0,
                -1.0 / 120.0,
                1.0 / 5040.0,
                -1.0 / 362880.0,
                1.0 / 39916800.0,
            ]),
            c2_p: horner([
                -1.0 / 12.0,
                1.0 / 180.0,
                -1.0 / 6720.0,
                1.0 / 453600.0,
                -1.0 / 47900160.0,
            ]),
            c3_p: horner([
                -1.0 / 60.0,
                1.0 / 1260.0,
                -1.0 / 60480.0,
                1.0 / 4989600.0,
                -1.0 / 622702080.0,
            ]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ExpJacobianCoeffs, exp_jacobian_coeffs};

    // The natural size of each coefficient (its value at θ = 0). Errors are measured against
    // `|want| + scale` rather than `|want|`: `c1 = sin θ / θ` passes through zero at θ = π, where a
    // purely relative error is unbounded and says nothing about the accuracy downstream.
    const SCALES: [f64; 5] = [1.0, 0.5, 1.0 / 6.0, 1.0 / 12.0, 1.0 / 60.0];

    fn as_array<F>(k: ExpJacobianCoeffs<F>) -> [F; 5] {
        [k.c1, k.c2, k.c3, k.c2_p, k.c3_p]
    }

    // A sweep of θ over `(0, π]`: decades near the origin where the closed forms cancel, both
    // branch cutoffs, and a uniform sweep over the rest.
    fn thetas() -> Vec<f64> {
        let mut out: Vec<f64> = (-9..=0)
            .flat_map(|e| [1.0, 2.0, 5.0].map(|m| m * 10f64.powi(e)))
            .collect();
        for cutoff_sq in [0.024513f64, 1.3661f64] {
            out.extend([1.0 - 1e-6, 1.0, 1.0 + 1e-6].map(|d| (cutoff_sq * d).sqrt()));
        }
        out.extend((1..=64).map(|i| i as f64 * std::f64::consts::PI / 64.0));
        out
    }

    // `f64` is accurate to ~1e-11 everywhere (see the note above `exp_jacobian_coeffs`), so it serves as
    // the reference for `f32`. What this guards against is a coefficient evaluated on the wrong
    // side of its cutoff: the `f64` cutoff is `θ² = 0.0245` and the `f32` one is `θ² = 1.366`, and
    // sharing one cutoff leaves `f32` on the closed forms where they are numerically dead — that
    // was worth five orders of magnitude on `c2_p` and seven on `c3_p`.
    #[test]
    fn f32_coeffs_track_f64_across_the_branch_cutoff() {
        for theta in thetas() {
            let want = as_array(exp_jacobian_coeffs(theta * theta));
            let got = as_array(exp_jacobian_coeffs((theta * theta) as f32));

            for (i, (&want, &got)) in want.iter().zip(got.iter()).enumerate() {
                let error = ((got as f64) - want).abs() / (want.abs() + SCALES[i]);
                assert!(
                    error < 1e-6,
                    "θ={theta}: coefficient {i} is {got} (f32) vs {want} (f64), \
                     scaled error {error:e} exceeds 1e-6",
                );
            }
        }
    }

    // Truncation and round-off are balanced at the cutoff, so neither branch should be visibly
    // better there. A jump means the cutoff has drifted out of the series' accurate range — which
    // is the one failure mode the `f32` sweep above cannot see, having no finer reference to
    // compare `f64` against.
    #[test]
    fn the_two_f64_branches_agree_at_the_cutoff() {
        let cutoff_sq = 0.024513f64;
        let below = as_array(exp_jacobian_coeffs(cutoff_sq * (1.0 - 1e-9)));
        let above = as_array(exp_jacobian_coeffs(cutoff_sq * (1.0 + 1e-9)));

        for (i, (&below, &above)) in below.iter().zip(above.iter()).enumerate() {
            let error = (above - below).abs() / (below.abs() + SCALES[i]);
            assert!(
                error < 1e-10,
                "θ²={cutoff_sq}: coefficient {i} jumps from {below} (Taylor) to {above} \
                 (closed form), scaled error {error:e} exceeds 1e-10",
            );
        }
    }
}
