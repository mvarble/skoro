use num_traits::{Float, FromPrimitive, One};
use skoro_backend::{
    Backend, CotangentFrame, NdArrayBackend, NdArrayFloat, NdArrayInt, NdArrayTensor, Point,
    SharedArray, TangentFrame, TangentVector,
};
use skoro_backend::{Element, host};
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
            inputs.into_iter().map(Input::unwrap_float).collect();
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
            DiffModeState::Hvp {
                diffs,
                tangents: tvecs,
            } => {
                let [dr] = tvecs.try_into().unwrap();
                let eta = dr.map(|dr| log_jvp(omega.clone(), dr.unsqueeze_dim(2)).squeeze_dim(2));
                let vjp_state = diffs[0].then(|| omega.clone());
                DiffOperationForward::hvp(omega, eta.clone(), vjp_state, eta)
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

pub fn log<B: Backend>(r: Point<B>) -> Point<B> {
    if host::is_f64(&r) {
        log_typed::<B, f64>(r)
    } else {
        log_typed::<B, f32>(r)
    }
}

fn log_typed<B, F>(r: Point<B>) -> Point<B>
where
    B: Backend,
    F: Float + One + FromPrimitive + Element,
{
    // unpack data from inputs
    let [batch_size, nine] = r.dims();
    if nine != 9 {
        panic!("log requires a tensor of shape `[b, 9]`");
    }

    // unpack tensor data
    let device = r.device();
    let dtype = r.dtype();
    let r_slice: Vec<F> = host::read(r);

    // Allocate output memory
    let mut omega_vec = Vec::with_capacity(batch_size * 3);

    let one = <F as One>::one();
    let half = <F as FromPrimitive>::from_f32(0.5).unwrap();
    let eps = <F as FromPrimitive>::from_f32(1e-6).unwrap();
    let twelve = <F as FromPrimitive>::from_f32(12.0).unwrap();
    let neg_thresh = <F as FromPrimitive>::from_f32(-0.999).unwrap();
    let two = <F as FromPrimitive>::from_f32(2.0).unwrap();
    let zero = <F as FromPrimitive>::from_f32(0.0).unwrap();

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

    host::write(omega_vec, [batch_size, 3], &device, dtype)
}

pub fn log_jvp<B: Backend>(omega: Point<B>, v: TangentFrame<B>) -> TangentFrame<B> {
    if host::is_f64(&omega) {
        log_jvp_typed::<B, f64>(omega, v)
    } else {
        log_jvp_typed::<B, f32>(omega, v)
    }
}

fn log_jvp_typed<B, F>(omega: Point<B>, v: TangentFrame<B>) -> TangentFrame<B>
where
    B: Backend,
    F: Float + One + FromPrimitive + Element,
{
    // unpack data from inputs
    let [batch_size, three] = omega.dims();
    let [obatch_size, othree, frame_size] = v.dims();
    if batch_size != obatch_size || three != 3 || othree != 3 {
        panic!("log_jvp requires tensors of shapes `[b, 3]`, and `[b, 3, f]`");
    }

    // unpack tensor data
    let device = omega.device();
    let dtype = omega.dtype();
    let omega_slice: Vec<F> = host::read(omega);
    let v_slice: Vec<F> = host::read(v);

    // Allocate output memory
    let mut out_vec = Vec::with_capacity(batch_size * 3 * frame_size);
    let half = <F as FromPrimitive>::from_f32(0.5).unwrap();

    let batch_iter = omega_slice
        .chunks_exact(3)
        .zip(v_slice.chunks_exact(3 * frame_size));

    for (omega_batch, v_batch) in batch_iter {
        let w_x = omega_batch[0];
        let w_y = omega_batch[1];
        let w_z = omega_batch[2];
        let th_sq = w_x * w_x + w_y * w_y + w_z * w_z;

        // c(|ω|)
        let LogJacobianCoeffs { c, .. } = log_jacobian_coeffs(th_sq);

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
    host::write(out_vec, [batch_size, 3, frame_size], &device, dtype)
}

pub fn log_vjp<B: Backend>(omega: Point<B>, alpha: CotangentFrame<B>) -> CotangentFrame<B> {
    if host::is_f64(&omega) {
        log_vjp_typed::<B, f64>(omega, alpha)
    } else {
        log_vjp_typed::<B, f32>(omega, alpha)
    }
}

fn log_vjp_typed<B, F>(omega: Point<B>, alpha: CotangentFrame<B>) -> CotangentFrame<B>
where
    B: Backend,
    F: Float + One + FromPrimitive + Element,
{
    // unpack data from inputs
    let [batch_size, three] = omega.dims();
    let [obatch_size, othree, frame_size] = alpha.dims();
    if batch_size != obatch_size || three != 3 || othree != 3 {
        panic!("log_vjp requires tensors of shapes `[b, 3]`, and `[b, 3, f]`");
    }

    // unpack tensor data
    let device = omega.device();
    let dtype = omega.dtype();
    let omega_slice: Vec<F> = host::read(omega);
    let a_slice: Vec<F> = host::read(alpha);

    // Allocate output memory
    let mut out_vec = Vec::with_capacity(batch_size * 3 * frame_size);
    let half = <F as FromPrimitive>::from_f32(0.5).unwrap();

    let batch_iter = omega_slice
        .chunks_exact(3)
        .zip(a_slice.chunks_exact(3 * frame_size));

    for (omega_batch, a_batch) in batch_iter {
        let w_x = omega_batch[0];
        let w_y = omega_batch[1];
        let w_z = omega_batch[2];
        let th_sq = w_x * w_x + w_y * w_y + w_z * w_z;

        // c(|ω|)
        let LogJacobianCoeffs { c, .. } = log_jacobian_coeffs(th_sq);

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
    host::write(out_vec, [batch_size, 3, frame_size], &device, dtype)
}

pub fn log_hvp<B: Backend>(
    omega: Point<B>,
    eta: TangentVector<B>,
    alpha: CotangentFrame<B>,
) -> CotangentFrame<B> {
    if host::is_f64(&omega) {
        log_hvp_typed::<B, f64>(omega, eta, alpha)
    } else {
        log_hvp_typed::<B, f32>(omega, eta, alpha)
    }
}

fn log_hvp_typed<B, F>(
    omega: Point<B>,
    eta: TangentVector<B>,
    alpha: CotangentFrame<B>,
) -> CotangentFrame<B>
where
    B: Backend,
    F: Float + One + FromPrimitive + Element,
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
    let dtype = omega.dtype();
    let omega_slice: Vec<F> = host::read(omega);
    let e_slice: Vec<F> = host::read(eta);
    let a_slice: Vec<F> = host::read(alpha);

    // Allocate output memory
    let mut out_vec = Vec::with_capacity(batch_size * 3 * frame_size);
    let half = <F as FromPrimitive>::from_f32(0.5).unwrap();

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

        // c(|ω|), c'(|ω|)/|ω|
        let LogJacobianCoeffs { c, c_p } = log_jacobian_coeffs(th_sq);
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

            -half * exa_x + c * (exwxa_x + wxexa_x) + c_p * wdote * wxwxa_x
        }));

        out_vec.extend(a_iter.clone().map(|((&a_x, &a_y), &a_z)| {
            let (exa_x, exa_y, exa_z) = cross_product(e_x, e_y, e_z, a_x, a_y, a_z);
            let (wxa_x, _, wxa_z) = cross_product(w_x, w_y, w_z, a_x, a_y, a_z);
            let wxwxa_y = w_z * wxa_x - w_x * wxa_z;
            let exwxa_y = e_z * wxa_x - e_x * wxa_z;
            let wxexa_y = w_z * exa_x - w_x * exa_z;

            -half * exa_y + c * (exwxa_y + wxexa_y) + c_p * wdote * wxwxa_y
        }));

        out_vec.extend(a_iter.map(|((&a_x, &a_y), &a_z)| {
            let (exa_x, exa_y, exa_z) = cross_product(e_x, e_y, e_z, a_x, a_y, a_z);
            let (wxa_x, wxa_y, _) = cross_product(w_x, w_y, w_z, a_x, a_y, a_z);
            let wxwxa_z = w_x * wxa_y - w_y * wxa_x;
            let exwxa_z = e_x * wxa_y - e_y * wxa_x;
            let wxexa_z = w_x * exa_y - w_y * exa_x;

            -half * exa_z + c * (exwxa_z + wxexa_z) + c_p * wdote * wxwxa_z
        }));
    }

    // pack data
    host::write(out_vec, [batch_size, 3, frame_size], &device, dtype)
}

// SO(3) has a closed form for the Lie-group logarithm map has the following right-Jacobian which we
// use in our JVP/VJP operations.
// ```
// J(ω) = I + (1/2) ω^ + (1/|ω|²) (1 - (|ω|/2) cot(|ω|/2))) (ω^)²
// ```
// This function returns the coefficients
// ```
// c    = (1/θ²) (1 - (θ/2) cot(θ/2))
// c_p  = c'(θ)/θ = (1/θ⁴) ((θ/2) cot(θ/2) + (θ/2)² csc²(θ/2) - 2)
// ```
// ## Evaluating them without losing precision
//
// Both closed forms cancel as θ → 0. `c` cancels at second order, `c_p` at fourth: its numerator is
// a difference of terms of size ~2 that lands at `c_p·θ⁴`, so it loses a factor of `2/(c_p θ⁴)` in
// significance — about 1e5 at θ = 1e-2. Below a cutoff we therefore evaluate a truncated Taylor
// series, in θ² so that the input is `θ²` rather than θ.
// ```
// c   = 1/12  + θ²/720  + θ⁴/30240  + θ⁶/1209600 + θ⁸/47900160
//             + 691θ¹⁰/1307674368000 + θ¹²/74724249600 + 3617θ¹⁴/10670622842880000
// c_p = 1/360 + θ²/7560 + θ⁴/201600  + θ⁶/5987520 + 691θ⁸/130767436800
//             + θ¹⁰/6227020800 + 3617θ¹²/762187345920000 + 43867θ¹⁴/319318388573184000
// ```
// The cutoff is where the two branches cross, measured rather than derived — these series converge
// only as `(θ/2π)²` per term, so the balance point sits far higher than for `exp`'s. Worst-case
// relative error over θ ∈ (0, π], sweeping the cutoff at eight terms:
// ```
//   f32   θ² > 1.3    c 5.4e-7   c_p 5.6e-5
//         θ² > 5.0    c 1.6e-7   c_p 1.9e-6     <- chosen
//         θ² > 9.9    c 1.0e-5   c_p 1.0e-4
//   f64   θ² > 0.05   c 1.8e-14  c_p 6.0e-11
//         θ² > 0.3    c 5.5e-15  c_p 9.5e-13    <- chosen
// ```
// A single cutoff cannot serve both widths, and the previous code used one (`θ² > 1e-6`, with a
// zeroth-order Taylor branch) calibrated for neither. Before and after:
// ```
//                  c         c_p
//   f32 before     9.5e-2    8.6e+3
//   f32 after      1.6e-7    1.9e-6
//   f64 before     1.7e-8    8.8e-3
//   f64 after      5.5e-15   9.5e-13
// ```
// Note that `f64` was broken too: `c_p` was wrong by 8% just above the old cutoff. Both feed
// `log_hvp`, and `c` also feeds `log_jvp`/`log_vjp`, so this reached every SO(3) and SE(3)
// inverse-retraction derivative.

/// The scalar coefficients of `J(ω)` for the logarithm, both functions of `θ = |ω|` alone.
///
/// Returned as a struct rather than a tuple because the JVP and VJP need only `c` while the HVP
/// needs both. Field access keeps each call site self-documenting and makes a mix-up a compile
/// error. Note this is a *different* Jacobian from [`super::super::exp::ndarray::ExpJacobianCoeffs`]
/// — that one is `Exp`'s, this one is `Log`'s.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogJacobianCoeffs<F> {
    /// `(1 - (θ/2) cot(θ/2)) / θ²`
    pub c: F,
    /// `c'(θ) / θ`
    pub c_p: F,
}

#[inline(always)]
pub fn log_jacobian_coeffs<F: Float>(theta_sq: F) -> LogJacobianCoeffs<F> {
    let k = |x: f64| F::from(x).unwrap();

    // Measured crossing points; anything coarser than `f64` takes the wider Taylor region, which is
    // the safe direction.
    let cutoff_sq = if F::epsilon() > k(1e-10) {
        k(5.0)
    } else {
        k(0.3)
    };

    if theta_sq > cutoff_sq {
        let one = k(1.0);
        let theta = theta_sq.sqrt();
        let theta_fourth = theta_sq * theta_sq;
        let halftheta = k(0.5) * theta;
        let cot_halfth = one / halftheta.tan();
        let halfth_cot_halfth = halftheta * cot_halfth;
        let csc2 = one + cot_halfth * cot_halfth;

        LogJacobianCoeffs {
            c: (one / theta_sq) * (one - halfth_cot_halfth),
            c_p: (one / theta_fourth) * (halfth_cot_halfth + halftheta * halftheta * csc2 - k(2.0)),
        }
    } else {
        // Horner in θ², highest power first.
        let horner = |a: [f64; 8]| {
            a.iter()
                .rev()
                .fold(k(0.0), |acc, &coeff| acc * theta_sq + k(coeff))
        };
        LogJacobianCoeffs {
            c: horner([
                1.0 / 12.0,
                1.0 / 720.0,
                1.0 / 30240.0,
                1.0 / 1209600.0,
                1.0 / 47900160.0,
                691.0 / 1307674368000.0,
                1.0 / 74724249600.0,
                3617.0 / 10670622842880000.0,
            ]),
            c_p: horner([
                1.0 / 360.0,
                1.0 / 7560.0,
                1.0 / 201600.0,
                1.0 / 5987520.0,
                691.0 / 130767436800.0,
                1.0 / 6227020800.0,
                3617.0 / 762187345920000.0,
                43867.0 / 319318388573184000.0,
            ]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{LogJacobianCoeffs, log_jacobian_coeffs};

    /// The natural size of each coefficient (its value at θ = 0). Errors are measured against
    /// `|want| + scale` rather than `|want|` so a coefficient passing near zero cannot produce an
    /// unbounded relative error.
    const SCALES: [f64; 2] = [1.0 / 12.0, 1.0 / 360.0];

    fn as_array<F>(k: LogJacobianCoeffs<F>) -> [F; 2] {
        [k.c, k.c_p]
    }

    /// Decades near the origin, both branch cutoffs (`θ² = 5.0` for `f32`, `0.3` for `f64`), and a
    /// uniform sweep over the rest of `(0, π]`.
    fn thetas() -> Vec<f64> {
        let mut out: Vec<f64> = (-9..=0)
            .flat_map(|e| [1.0, 2.0, 5.0].map(|m| m * 10f64.powi(e)))
            .collect();
        for cutoff_sq in [0.3f64, 5.0] {
            out.extend([1.0 - 1e-6, 1.0, 1.0 + 1e-6].map(|d| (cutoff_sq * d).sqrt()));
        }
        out.extend((1..=64).map(|i| i as f64 * std::f64::consts::PI / 64.0));
        out.retain(|&t| t <= std::f64::consts::PI);
        out
    }

    /// `f64` is accurate to ~1e-12 everywhere (see the note above `log_jacobian_coeffs`), so it
    /// serves as the reference for `f32`.
    ///
    /// This is the only test that sees the cutoff choice directly. The `log_hvp` reference table in
    /// `tests/common/coefficients.rs` catches the historical bug — an `f64`-calibrated cutoff with
    /// a zeroth-order Taylor branch — but not a swap of the two current cutoffs, because `c_p`
    /// contributes only ~1% of `log_hvp` over the band where they differ.
    #[test]
    fn f32_coeffs_track_f64_across_the_branch_cutoff() {
        for theta in thetas() {
            let want = as_array(log_jacobian_coeffs(theta * theta));
            let got = as_array(log_jacobian_coeffs((theta * theta) as f32));

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

    /// Truncation and round-off are balanced at the cutoff, so neither branch should be visibly
    /// better there. A jump means the cutoff has drifted out of the series' accurate range — the
    /// one failure mode the `f32` sweep above cannot see, having no finer reference for `f64`.
    #[test]
    fn the_two_f64_branches_agree_at_the_cutoff() {
        let cutoff_sq = 0.3f64;
        let below = as_array(log_jacobian_coeffs(cutoff_sq * (1.0 - 1e-9)));
        let above = as_array(log_jacobian_coeffs(cutoff_sq * (1.0 + 1e-9)));

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
