use burn::Tensor;
use num_traits::{Float, FromPrimitive, One};
use skoro_backend::{
    Backend, CotangentFrame, NdArrayBackend, NdArrayFloat, NdArrayInt, NdArrayTensor, Point,
    SharedArray, TangentFrame, TangentVector, TensorData,
};
use skoro_linalg::vector::cross_product::ndarray::cross_product_impl as cross_product;
use skoro_operation::{
    Input,
    diff::{DiffModeState, DiffOperation, DiffOperationForward},
};

use super::SO3Log;

impl<F, I> DiffOperation<NdArrayBackend<F, I>> for SO3Log
where
    F: NdArrayFloat + Float,
    I: NdArrayInt,
    NdArrayTensor: From<SharedArray<F>> + From<SharedArray<I>>,
{
    type VjpState = Option<Point<NdArrayBackend<F, I>>>;
    type HvpState = Option<Point<NdArrayBackend<F, I>>>;

    fn forward(
        &self,
        inputs: Vec<Input<NdArrayBackend<F, I>>>,
        diff_state: DiffModeState<NdArrayBackend<F, I>>,
    ) -> DiffOperationForward<NdArrayBackend<F, I>, Self::VjpState, Self::HvpState> {
        let inputs: Vec<Point<NdArrayBackend<F, I>>> =
            inputs.into_iter().map(Input::into_float).collect();
        let [r] = inputs.try_into().unwrap();
        let omega = log(r);
        match diff_state {
            DiffModeState::ForwardOnly => DiffOperationForward::forward_only(omega),
            DiffModeState::Jvp(tvecs) => {
                let [dr] = tvecs.try_into().unwrap();
                if let Some(dr) = dr {
                    DiffOperationForward::jvp(omega.clone(), Some(log_jvp(omega, dr)))
                } else {
                    DiffOperationForward::jvp(omega, None)
                }
            }
            DiffModeState::Vjp(diffs) => {
                let [diff] = diffs.try_into().unwrap();
                if diff {
                    DiffOperationForward::vjp(omega.clone(), Some(omega))
                } else {
                    DiffOperationForward::vjp(omega, None)
                }
            }
            DiffModeState::Hvp(tvecs) => {
                let [dr] = tvecs.try_into().unwrap();
                if let Some(dr) = dr {
                    let eta = log_jvp(omega.clone(), dr.unsqueeze_dim(2)).squeeze_dim(2);
                    DiffOperationForward::hvp(
                        omega.clone(),
                        Some(eta.clone()),
                        Some(omega),
                        Some(eta),
                    )
                } else {
                    DiffOperationForward::hvp(omega, None, None, None)
                }
            }
        }
    }

    fn vjp(
        &self,
        vjp_state: &Self::VjpState,
        alpha: CotangentFrame<NdArrayBackend<F, I>>,
    ) -> Vec<Option<CotangentFrame<NdArrayBackend<F, I>>>> {
        if let Some(omega) = vjp_state {
            vec![Some(log_vjp(omega.clone(), alpha))]
        } else {
            vec![None]
        }
    }

    fn hvp(
        &self,
        vjp_state: &Self::VjpState,
        hvp_state: &Self::HvpState,
        alpha: CotangentFrame<NdArrayBackend<F, I>>,
    ) -> Vec<Option<CotangentFrame<NdArrayBackend<F, I>>>> {
        if let Some((omega, eta)) = vjp_state.as_ref().zip(hvp_state.as_ref()) {
            vec![Some(log_hvp(omega.clone(), eta.clone(), alpha))]
        } else {
            vec![None]
        }
    }
}

pub fn log<B>(r: Point<B>) -> Point<B>
where
    B: Backend,
    B::FloatElem: Float + One + FromPrimitive,
{
    // unpack data from inputs
    let [batch_size, nine] = r.dims();
    if nine != 9 {
        panic!("log requires a tensor of shape `[b, 9]`");
    }

    // unpack tensor data
    let device = r.device();
    let r_data = r.into_data();
    let r_slice = r_data.as_slice::<B::FloatElem>().unwrap();

    // Allocate output memory
    let mut omega_vec = Vec::with_capacity(batch_size * 3);

    let one = <B::FloatElem as One>::one();
    let half = <B::FloatElem as FromPrimitive>::from_f32(0.5).unwrap();
    let eps = <B::FloatElem as FromPrimitive>::from_f32(1e-6).unwrap();
    let twelve = <B::FloatElem as FromPrimitive>::from_f32(12.0).unwrap();
    let neg_thresh = <B::FloatElem as FromPrimitive>::from_f32(-0.999).unwrap();
    let two = <B::FloatElem as FromPrimitive>::from_f32(2.0).unwrap();
    let zero = <B::FloatElem as FromPrimitive>::from_f32(0.0).unwrap();

    for batch in r_slice.chunks_exact(9) {
        let r11 = batch[0];
        let r12 = batch[1];
        let r13 = batch[2];
        let r21 = batch[3];
        let r22 = batch[4];
        let r23 = batch[5];
        let r31 = batch[6];
        let r32 = batch[7];
        let r33 = batch[8];

        let r_trace = r11 + r22 + r33;
        let mut cos_theta = (r_trace - one) * half;
        if cos_theta > one {
            cos_theta = one;
        }
        if cos_theta < -one {
            cos_theta = -one;
        }

        let theta = cos_theta.acos();
        let t_sq = theta * theta;
        let a_x = r32 - r23;
        let a_y = r13 - r31;
        let a_z = r21 - r12;

        let (w_x, w_y, w_z) = if t_sq < eps {
            // Taylor series for theta / (2 * sin(theta)) near 0: 1/2 + theta^2 / 12
            let mult = half + t_sq / twelve;
            (a_x * mult, a_y * mult, a_z * mult)
        } else if cos_theta < neg_thresh {
            // Robust evaluation for theta near pi
            let one_minus_cos = one - cos_theta;

            let s11 = r11 - cos_theta;
            let s22 = r22 - cos_theta;
            let s33 = r33 - cos_theta;

            let (u_x, u_y, u_z) = if s11 > s22 && s11 > s33 {
                let u_x = (s11 / one_minus_cos).sqrt();
                let denom = two * one_minus_cos * u_x;
                let u_y = (r12 + r21) / denom;
                let u_z = (r13 + r31) / denom;
                (u_x, u_y, u_z)
            } else if s22 > s33 {
                let u_y = (s22 / one_minus_cos).sqrt();
                let denom = two * one_minus_cos * u_y;
                let u_x = (r12 + r21) / denom;
                let u_z = (r23 + r32) / denom;
                (u_x, u_y, u_z)
            } else {
                let u_z = (s33 / one_minus_cos).sqrt();
                let denom = two * one_minus_cos * u_z;
                let u_x = (r13 + r31) / denom;
                let u_y = (r23 + r32) / denom;
                (u_x, u_y, u_z)
            };

            // Align the sign of u with a = (R - R^T)
            let dot = u_x * a_x + u_y * a_y + u_z * a_z;
            if dot < zero {
                (-theta * u_x, -theta * u_y, -theta * u_z)
            } else {
                (theta * u_x, theta * u_y, theta * u_z)
            }
        } else {
            // Standard evaluation inside safe region
            let mult = theta / (two * theta.sin());
            (a_x * mult, a_y * mult, a_z * mult)
        };

        omega_vec.push(w_x);
        omega_vec.push(w_y);
        omega_vec.push(w_z);
    }

    Tensor::from_data(TensorData::new(omega_vec, [batch_size, 3]), &device)
}

pub fn log_jvp<B>(omega: Point<B>, v: TangentFrame<B>) -> TangentFrame<B>
where
    B: Backend,
    B::FloatElem: Float + One + FromPrimitive,
{
    // unpack data from inputs
    let [batch_size, three] = omega.dims();
    let [obatch_size, othree, frame_size] = v.dims();
    if batch_size != obatch_size || three != 3 || othree != 3 {
        panic!("log_jvp requires tensors of shapes `[b, 3]`, and `[b, 3, f]`");
    }

    // unpack tensor data
    let device = omega.device();
    let omega_data = omega.into_data();
    let omega_slice = omega_data.as_slice::<B::FloatElem>().unwrap();
    let v_data = v.into_data();
    let v_slice = v_data.as_slice::<B::FloatElem>().unwrap();

    // Allocate output memory
    let mut out_vec = Vec::with_capacity(batch_size * 3 * frame_size);
    let half = <B::FloatElem as FromPrimitive>::from_f32(0.5).unwrap();

    let batch_iter = omega_slice
        .chunks_exact(3)
        .zip(v_slice.chunks_exact(3 * frame_size));

    for (omega_batch, v_batch) in batch_iter {
        let w_x = omega_batch[0];
        let w_y = omega_batch[1];
        let w_z = omega_batch[2];
        let th_sq = w_x * w_x + w_y * w_y + w_z * w_z;

        // c(|ω|)
        let (c, _) = get_so3_coeffs(th_sq);

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
            v_x + half * wxv_x + c * wxwxv_x
        }));
        out_vec.extend(v_iter.clone().map(|((&v_x, &v_y), &v_z)| {
            let (wxv_x, wxv_y, wxv_z) = cross_product(w_x, w_y, w_z, v_x, v_y, v_z);
            let wxwxv_y = w_z * wxv_x - w_x * wxv_z;
            v_y + half * wxv_y + c * wxwxv_y
        }));
        out_vec.extend(v_iter.map(|((&v_x, &v_y), &v_z)| {
            let (wxv_x, wxv_y, wxv_z) = cross_product(w_x, w_y, w_z, v_x, v_y, v_z);
            let wxwxv_z = w_x * wxv_y - w_y * wxv_x;
            v_z + half * wxv_z + c * wxwxv_z
        }));
    }

    // pack data
    Tensor::from_data(
        TensorData::new(out_vec, [batch_size, 3, frame_size]),
        &device,
    )
}

pub fn log_vjp<B>(omega: Point<B>, alpha: CotangentFrame<B>) -> CotangentFrame<B>
where
    B: Backend,
    B::FloatElem: Float + One + FromPrimitive,
{
    // unpack data from inputs
    let [batch_size, three] = omega.dims();
    let [obatch_size, othree, frame_size] = alpha.dims();
    if batch_size != obatch_size || three != 3 || othree != 3 {
        panic!("log_vjp requires tensors of shapes `[b, 3]`, and `[b, 3, f]`");
    }

    // unpack tensor data
    let device = omega.device();
    let omega_data = omega.into_data();
    let omega_slice = omega_data.as_slice::<B::FloatElem>().unwrap();
    let a_data = alpha.into_data();
    let a_slice = a_data.as_slice::<B::FloatElem>().unwrap();

    // Allocate output memory
    let mut out_vec = Vec::with_capacity(batch_size * 3 * frame_size);
    let half = <B::FloatElem as FromPrimitive>::from_f32(0.5).unwrap();

    let batch_iter = omega_slice
        .chunks_exact(3)
        .zip(a_slice.chunks_exact(3 * frame_size));

    for (omega_batch, a_batch) in batch_iter {
        let w_x = omega_batch[0];
        let w_y = omega_batch[1];
        let w_z = omega_batch[2];
        let th_sq = w_x * w_x + w_y * w_y + w_z * w_z;

        // c(|ω|)
        let (c, _) = get_so3_coeffs(th_sq);

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
            a_x - half * wxa_x + c * wxwxa_x
        }));
        out_vec.extend(a_iter.clone().map(|((&a_x, &a_y), &a_z)| {
            let (wxa_x, wxa_y, wxa_z) = cross_product(w_x, w_y, w_z, a_x, a_y, a_z);
            let wxwxa_y = w_z * wxa_x - w_x * wxa_z;
            a_y - half * wxa_y + c * wxwxa_y
        }));
        out_vec.extend(a_iter.map(|((&a_x, &a_y), &a_z)| {
            let (wxa_x, wxa_y, wxa_z) = cross_product(w_x, w_y, w_z, a_x, a_y, a_z);
            let wxwxa_z = w_x * wxa_y - w_y * wxa_x;
            a_z - half * wxa_z + c * wxwxa_z
        }));
    }

    // pack data
    Tensor::from_data(
        TensorData::new(out_vec, [batch_size, 3, frame_size]),
        &device,
    )
}

pub fn log_hvp<B>(
    omega: Point<B>,
    eta: TangentVector<B>,
    alpha: CotangentFrame<B>,
) -> CotangentFrame<B>
where
    B: Backend,
    B::FloatElem: Float + One + FromPrimitive,
{
    // unpack data from inputs
    let [batch_size, three] = omega.dims();
    let [obatch_size, othree] = eta.dims();
    let [oobatch_size, oothree, frame_size] = alpha.dims();
    if batch_size != obatch_size
        || batch_size != oobatch_size
        || three != 3
        || othree != 3
        || oothree != 3
    {
        panic!("log_hvp requires tensors of shapes `[b, 3]`, `[b, 3]`, and `[b, 3, f]`");
    }

    // unpack tensor data
    let device = omega.device();
    let omega_data = omega.into_data();
    let omega_slice = omega_data.as_slice::<B::FloatElem>().unwrap();
    let e_data = eta.into_data();
    let e_slice = e_data.as_slice::<B::FloatElem>().unwrap();
    let a_data = alpha.into_data();
    let a_slice = a_data.as_slice::<B::FloatElem>().unwrap();

    // Allocate output memory
    let mut out_vec = Vec::with_capacity(batch_size * 3 * frame_size);
    let half = <B::FloatElem as FromPrimitive>::from_f32(0.5).unwrap();

    let batch_iter = omega_slice
        .chunks_exact(3)
        .zip(e_slice.chunks_exact(3))
        .zip(a_slice.chunks_exact(3 * frame_size));

    for ((omega_batch, e_batch), a_batch) in batch_iter {
        let w_x = omega_batch[0];
        let w_y = omega_batch[1];
        let w_z = omega_batch[2];
        let th_sq = w_x * w_x + w_y * w_y + w_z * w_z;

        let e_x = e_batch[0];
        let e_y = e_batch[1];
        let e_z = e_batch[2];

        // c(|ω|), d(|ω|)
        let (c, d) = get_so3_coeffs(th_sq);
        let wdote = w_x * e_x + w_y * e_y + w_z * e_z;

        // `a_slice` indexes `[b, 3, f]` from the right, so each chunk of size `3 * f` will
        // correspond one batch, with `x` data of length `f`, then `y` of length `f`, then `z` of
        // length `f`; the proceeding lines split these into separate arrays.
        let (a_x, rest) = a_batch.split_at(frame_size);
        let (a_y, a_z) = rest.split_at(frame_size);

        // fill the `x` data first, then the `y`, then the `z`.
        let a_iter = a_x.iter().zip(a_y).zip(a_z);

        out_vec.extend(a_iter.clone().map(|((&a_x, &a_y), &a_z)| {
            let (exa_x, exa_y, exa_z) = cross_product(e_x, e_y, e_z, a_x, a_y, a_z);
            let (_, wxa_y, wxa_z) = cross_product(w_x, w_y, w_z, a_x, a_y, a_z);
            let wxwxa_x = w_y * wxa_z - w_z * wxa_y;
            let exwxa_x = e_y * wxa_z - e_z * wxa_y;
            let wxexa_x = w_y * exa_z - w_z * exa_y;

            -half * exa_x + c * (exwxa_x + wxexa_x) + d * wdote * wxwxa_x
        }));

        out_vec.extend(a_iter.clone().map(|((&a_x, &a_y), &a_z)| {
            let (exa_x, exa_y, exa_z) = cross_product(e_x, e_y, e_z, a_x, a_y, a_z);
            let (wxa_x, _, wxa_z) = cross_product(w_x, w_y, w_z, a_x, a_y, a_z);
            let wxwxa_y = w_z * wxa_x - w_x * wxa_z;
            let exwxa_y = e_z * wxa_x - e_x * wxa_z;
            let wxexa_y = w_z * exa_x - w_x * exa_z;

            -half * exa_y + c * (exwxa_y + wxexa_y) + d * wdote * wxwxa_y
        }));

        out_vec.extend(a_iter.map(|((&a_x, &a_y), &a_z)| {
            let (exa_x, exa_y, exa_z) = cross_product(e_x, e_y, e_z, a_x, a_y, a_z);
            let (wxa_x, wxa_y, _) = cross_product(w_x, w_y, w_z, a_x, a_y, a_z);
            let wxwxa_z = w_x * wxa_y - w_y * wxa_x;
            let exwxa_z = e_x * wxa_y - e_y * wxa_x;
            let wxexa_z = w_x * exa_y - w_y * exa_x;

            -half * exa_z + c * (exwxa_z + wxexa_z) + d * wdote * wxwxa_z
        }));
    }

    // pack data
    Tensor::from_data(
        TensorData::new(out_vec, [batch_size, 3, frame_size]),
        &device,
    )
}

// SO(3) has a closed form for the Lie-group logarithm map has the following right-Jacobian which we
// use in our JVP/VJP operations.
// ```
// J(ω) = I + (1/2) ω^ + (1/|ω|²) (1 - (|ω|/2) cot(|ω|/2))) (ω^)²
// ```
// This function returns the coefficients
// ```
// c(|ω|)      = (1/|ω|²) (1 - (|ω|/2) cot(|ω|/2)))
// c'(|ω|)/|ω| = (1/|ω|⁴) ((|ω|/2) cot(|ω|/2) + (|ω|/2)² csc²(|ω|/2) - 2)
// ```
// as well as a numerically stable approximation for small values.
// ```
// c(|ω|) = 1/12 + 1/360 |ω| + O(|ω|³)
// c'(|ω|)/|ω| = 1/360 + O(|ω|²)
// ```
#[inline(always)]
pub fn get_so3_coeffs<F: Float>(theta_sq: F) -> (F, F) {
    // Default to Taylor series expansions at theta = 0
    let mut c = F::from(1.0 / 12.0).unwrap();
    let mut d = F::from(1.0 / 360.0).unwrap();

    // For non-small theta, evaluate actual trig function.
    if theta_sq > F::from(1e-6).unwrap() {
        let one = F::from(1.0).unwrap();
        let half = F::from(0.5).unwrap();
        let theta = theta_sq.sqrt();
        let theta_fourth = theta_sq * theta_sq;
        let halftheta = half * theta;
        let cot_halfth = one / halftheta.tan();
        let halfth_cot_halfth = halftheta * cot_halfth;
        let csc2 = one + cot_halfth * cot_halfth;
        c = (one / theta_sq) * (one - halfth_cot_halfth);
        d = (one / theta_fourth)
            * (halfth_cot_halfth + halftheta * halftheta * csc2 - F::from(2.0).unwrap());
    }

    (c, d)
}
