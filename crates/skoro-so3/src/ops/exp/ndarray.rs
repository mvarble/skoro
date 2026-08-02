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
            DiffModeState::Hvp(tvecs) => {
                let [v] = tvecs.try_into().unwrap();
                DiffOperationForward::hvp(
                    exp(omega.clone()),
                    v.clone()
                        .map(|v| exp_jvp(omega.clone(), v.unsqueeze_dim(2)).squeeze_dim(2)),
                    if v.is_some() { Some(omega) } else { None },
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
        let (c1, c2, _, _, _) = get_so3_coeffs(th_sq);
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
        let (_, c2, c3, _, _) = get_so3_coeffs(th_sq);

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
        let (_, c2, c3, _, _) = get_so3_coeffs(th_sq);

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
        let (_, c2, c3, c2_p, c3_p) = get_so3_coeffs(th_sq);

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
// This function will compute the scalar coefficients showing up in each of these expressions,
// dependent on θ=|ω|.
// ```
// c1 = sin θ / θ
// c2 = (1 - cos θ) / θ²
// c3 = (θ - sin θ) / θ³
// c2_p = (θ sin θ - 2 (1 - cos θ)) / θ⁴
// c3_p = (θ (1 - cos θ) - 3 (θ - sin θ)) / |θ|⁵
// ```
// For numerical stability with small θ (<1e-3), we use Taylor-series approximations.
// ```
// c1 = 1 - 1/6 θ² + O(θ⁴) ≈ 1
// c2 = 1/2 - 1/24 θ² + O(θ⁴) ≈ 1/2
// c3 = 1/6 - 1/120  θ² + O(θ⁴) ≈ 1/6
// c2_p = -1/12 + O(θ2) ≈ -1/12
// c2_p = -1/60 + O(θ2) ≈ -1/60
// ```
//
// **Note:** The input is `θ²`, not `θ` for faster short-circuiting.
#[inline(always)]
pub fn get_so3_coeffs<F: Float>(theta_sq: F) -> (F, F, F, F, F) {
    // Default to Taylor series expansions at theta = 0
    let mut c1 = F::from(1.0).unwrap();
    let mut c2 = F::from(0.5).unwrap();
    let mut c3 = F::from(1.0 / 6.0).unwrap();
    let mut c2_p = F::from(-1.0 / 12.0).unwrap();
    let mut c3_p = F::from(-1.0 / 60.0).unwrap();

    // For non-small theta, evaluate actual trig functions.
    if theta_sq > F::from(1e-6).unwrap() {
        let theta = theta_sq.sqrt();
        let sin_th = theta.sin();
        let cos_th = theta.cos();
        let th_sq = theta_sq;
        let th_cubed = th_sq * theta;
        let th_quad = th_sq * th_sq;
        let th_fifth = th_quad * theta;

        c1 = sin_th / theta;
        c2 = (F::from(1.0).unwrap() - cos_th) / th_sq;
        c3 = (theta - sin_th) / th_cubed;
        c2_p =
            (theta * sin_th - F::from(2.0).unwrap() * (F::from(1.0).unwrap() - cos_th)) / th_quad;
        c3_p = (theta * (F::from(1.0).unwrap() - cos_th)
            - F::from(3.0).unwrap() * (theta - sin_th))
            / th_fifth;
    }

    (c1, c2, c3, c2_p, c3_p)
}
