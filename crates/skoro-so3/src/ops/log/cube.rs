use cubecl::prelude::*;

use burn::tensor::Shape;
use cubecl::{CubeCount, CubeDim};
use skoro_backend::{
    CotangentFrame, CubeBackend, CubeFloat, CubeInt, CubeRuntime, CubeTensor, Point, TangentFrame,
    TangentVector, ToCubeTensor,
};
use skoro_linalg::vector::cross_product::cube::kernel::cross_product_impl as cross_product;
use skoro_operation::{
    Input,
    diff::{DiffModeState, DiffOperation, DiffOperationForward},
};

use super::SO3Log;

impl<R: CubeRuntime, F: CubeFloat, I: CubeInt> DiffOperation<CubeBackend<R, F, I>> for SO3Log {
    type VjpState = Option<Point<CubeBackend<R, F, I>>>;
    type HvpState = Option<Point<CubeBackend<R, F, I>>>;

    fn forward(
        &self,
        inputs: Vec<Input<CubeBackend<R, F, I>>>,
        diff_state: DiffModeState<CubeBackend<R, F, I>>,
    ) -> DiffOperationForward<CubeBackend<R, F, I>, Self::VjpState, Self::HvpState> {
        let inputs: Vec<Point<CubeBackend<R, F, I>>> =
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
        alpha: CotangentFrame<CubeBackend<R, F, I>>,
    ) -> Vec<Option<CotangentFrame<CubeBackend<R, F, I>>>> {
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
        alpha: CotangentFrame<CubeBackend<R, F, I>>,
    ) -> Vec<Option<CotangentFrame<CubeBackend<R, F, I>>>> {
        if let Some((omega, eta)) = vjp_state.as_ref().zip(hvp_state.as_ref()) {
            vec![Some(log_hvp(omega.clone(), eta.clone(), alpha))]
        } else {
            vec![None]
        }
    }
}

pub fn log<R: CubeRuntime, F: CubeFloat, I: CubeInt>(
    r: Point<CubeBackend<R, F, I>>,
) -> Point<CubeBackend<R, F, I>> {
    // unpack data from inputs
    let [batch_size, nine] = r.dims();
    if nine != 9 {
        panic!("log requires a tensor of shape `[b, 9]`");
    }

    // cube inputs
    let device = r.device();
    let r_prim = r.to_cube_tensor();
    let client = r_prim.client.clone();

    // allocate output handles
    let omega_prim = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::new([batch_size, 3]),
        client.empty(3 * batch_size * core::mem::size_of::<F>()),
        F::dtype(),
    );

    // launch kernel
    let threads_per_block = 256;
    let num_blocks = (batch_size as u32 + threads_per_block - 1).div_ceil(threads_per_block);
    let cube_dim = CubeDim::new_1d(threads_per_block);
    let cubes_count = CubeCount::Static(num_blocks, 1, 1);
    log_kernel::launch::<F, R>(
        &client,
        cubes_count,
        cube_dim,
        r_prim.into_tensor_arg(),
        omega_prim.clone().into_tensor_arg(),
    );

    // pack data
    Point::from_primitive(burn::tensor::TensorPrimitive::Float(omega_prim))
}

pub fn log_jvp<R: CubeRuntime, F: CubeFloat, I: CubeInt>(
    omega: Point<CubeBackend<R, F, I>>,
    v: TangentFrame<CubeBackend<R, F, I>>,
) -> TangentFrame<CubeBackend<R, F, I>> {
    // unpack data from inputs
    let [batch_size, three] = omega.dims();
    let [obatch_size, othree, frame_size] = v.dims();
    if batch_size != obatch_size || three != 3 || othree != 3 {
        panic!("log_jvp requires tensors of shapes `[b, 3]`, and `[b, 3, f]`");
    }

    // cube inputs
    let device = omega.device();
    let omega_prim = omega.to_cube_tensor();
    let v_prim = v.to_cube_tensor();
    let client = omega_prim.client.clone();

    // allocate output handle
    let out_prim = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::new([batch_size, 3, frame_size]),
        client.empty(3 * batch_size * frame_size * core::mem::size_of::<F>()),
        F::dtype(),
    );

    // launch kernel
    let threads_x = 16;
    let threads_y = 16;
    let num_blocks_x = (batch_size as u32 + threads_x - 1).div_ceil(threads_x);
    let num_blocks_y = (frame_size as u32 + threads_y - 1).div_ceil(threads_y);
    let cube_dim = CubeDim::new_2d(threads_x, threads_y);
    let cubes_count = CubeCount::Static(num_blocks_x, num_blocks_y, 1);
    log_jvp_kernel::launch::<F, R>(
        &client,
        cubes_count,
        cube_dim,
        omega_prim.into_tensor_arg(),
        v_prim.into_tensor_arg(),
        out_prim.clone().into_tensor_arg(),
    );

    // pack data
    TangentFrame::from_primitive(burn::tensor::TensorPrimitive::Float(out_prim))
}

pub fn log_vjp<R: CubeRuntime, F: CubeFloat, I: CubeInt>(
    omega: Point<CubeBackend<R, F, I>>,
    alpha: CotangentFrame<CubeBackend<R, F, I>>,
) -> CotangentFrame<CubeBackend<R, F, I>> {
    // unpack data from inputs
    let [batch_size, three] = omega.dims();
    let [obatch_size, othree, frame_size] = alpha.dims();
    if batch_size != obatch_size || three != 3 || othree != 3 {
        panic!("log_vjp requires tensors of shapes `[b, 3]`, and `[b, 3, f]`");
    }

    // cube inputs
    let device = omega.device();
    let omega_prim = omega.to_cube_tensor();
    let a_prim = alpha.to_cube_tensor();
    let client = omega_prim.client.clone();

    // allocate output handle
    let out_prim = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::new([batch_size, 3, frame_size]),
        client.empty(3 * batch_size * frame_size * core::mem::size_of::<F>()),
        F::dtype(),
    );

    // launch kernel
    let threads_x = 16;
    let threads_y = 16;
    let num_blocks_x = (batch_size as u32 + threads_x - 1).div_ceil(threads_x);
    let num_blocks_y = (frame_size as u32 + threads_y - 1).div_ceil(threads_y);
    let cube_dim = CubeDim::new_2d(threads_x, threads_y);
    let cubes_count = CubeCount::Static(num_blocks_x, num_blocks_y, 1);
    log_vjp_kernel::launch::<F, R>(
        &client,
        cubes_count,
        cube_dim,
        omega_prim.into_tensor_arg(),
        a_prim.into_tensor_arg(),
        out_prim.clone().into_tensor_arg(),
    );

    // pack data
    TangentFrame::from_primitive(burn::tensor::TensorPrimitive::Float(out_prim))
}

pub fn log_hvp<R: CubeRuntime, F: CubeFloat, I: CubeInt>(
    omega: Point<CubeBackend<R, F, I>>,
    v: TangentVector<CubeBackend<R, F, I>>,
    alpha: CotangentFrame<CubeBackend<R, F, I>>,
) -> CotangentFrame<CubeBackend<R, F, I>> {
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
        panic!("log_hvp requires tensors of shapes `[b, 3]`, `[b, 3]`, and `[b, 3, f]`");
    }

    // cube inputs
    let device = omega.device();
    let omega_prim = omega.to_cube_tensor();
    let v_prim = v.to_cube_tensor();
    let a_prim = alpha.to_cube_tensor();
    let client = omega_prim.client.clone();

    // allocate output handle
    let out_prim = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::new([batch_size, 3, frame_size]),
        client.empty(3 * batch_size * frame_size * core::mem::size_of::<F>()),
        F::dtype(),
    );

    // launch kernel
    let threads_x = 16;
    let threads_y = 16;
    let num_blocks_x = (batch_size as u32 + threads_x - 1).div_ceil(threads_x);
    let num_blocks_y = (frame_size as u32 + threads_y - 1).div_ceil(threads_y);
    let cube_dim = CubeDim::new_2d(threads_x, threads_y);
    let cubes_count = CubeCount::Static(num_blocks_x, num_blocks_y, 1);
    log_hvp_kernel::launch::<F, R>(
        &client,
        cubes_count,
        cube_dim,
        omega_prim.into_tensor_arg(),
        v_prim.into_tensor_arg(),
        a_prim.into_tensor_arg(),
        out_prim.clone().into_tensor_arg(),
    );

    // pack data
    TangentFrame::from_primitive(burn::tensor::TensorPrimitive::Float(out_prim))
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
// Both closed forms cancel as θ → 0 — `c_p` at fourth order — so below a cutoff we use a truncated
// Taylor series in θ² instead. See the extended note on the `ndarray` twin of this function
// (`crates/skoro-so3/src/ops/log/ndarray.rs`) for the derivation and the measured error table; the
// series and the cutoffs are identical.
// ```
// c   = 1/12  + θ²/720  + θ⁴/30240  + θ⁶/1209600 + θ⁸/47900160
//             + 691θ¹⁰/1307674368000 + θ¹²/74724249600 + 3617θ¹⁴/10670622842880000
// c_p = 1/360 + θ²/7560 + θ⁴/201600  + θ⁶/5987520 + 691θ⁸/130767436800
//             + θ¹⁰/6227020800 + 3617θ¹²/762187345920000 + 43867θ¹⁴/319318388573184000
// ```
// This matters most on GPU: the device `tan` is less accurate than libm's, which widens the dead
// zone the closed forms leave around θ = 0.
//
// **Note:** the constants below go through `F::new`, which takes an `f32`, so on an `f64` runtime
// they carry only `f32` precision. That is a limitation of the `Float` frontend, not of the series;
// the `f32` path this kernel actually runs on is unaffected.
//
// This is a hand transcription of the `ndarray` function, so it can drift from it silently.
// `coefficients::log_hvp_tracks_the_f64_reference` pins it against the frozen `f64` table in
// `crates/skoro-so3/tests/common/coefficients.rs`, across both cutoffs.

/// The scalar coefficients of `J(ω)` for the logarithm, both functions of `θ = |ω|` alone.
///
/// The `ndarray` twin of this type carries the same fields and the same meanings. Note this is a
/// *different* Jacobian from `exp::cube::ExpJacobianCoeffs` — that one is `Exp`'s, this one is
/// `Log`'s.
#[derive(CubeType)]
pub struct LogJacobianCoeffs<F: Float> {
    /// `(1 - (θ/2) cot(θ/2)) / θ²`
    pub c: F,
    /// `c'(θ) / θ`
    pub c_p: F,
}

// `#[cube]` has no `if` expression, so the coefficients have to be declared before the branch and
// assigned inside it; the placeholder initializers are dead by construction.
#[allow(unused_assignments)]
#[cube]
fn log_jacobian_coeffs<F: Float>(theta_sq: F) -> LogJacobianCoeffs<F> {
    // Measured crossing points of the two branches. Anything coarser than `f64` takes the wider
    // Taylor region, which is the safe direction.
    let mut cutoff_sq = F::new(0.3);
    if F::EPSILON > F::new(1e-10) {
        cutoff_sq = F::new(5.0);
    }

    let mut c = F::new(0.0);
    let mut c_p = F::new(0.0);

    if theta_sq > cutoff_sq {
        let one = F::new(1.0);
        let theta = theta_sq.sqrt();
        let theta_fourth = theta_sq * theta_sq;
        let halftheta = F::new(0.5) * theta;
        let cot_halfth = one / halftheta.tan();
        let halfth_cot_halfth = halftheta * cot_halfth;
        let csc2 = one + cot_halfth * cot_halfth;

        c = (one / theta_sq) * (one - halfth_cot_halfth);
        c_p =
            (one / theta_fourth) * (halfth_cot_halfth + halftheta * halftheta * csc2 - F::new(2.0));
    } else {
        // Horner in θ², highest power first.
        let t = theta_sq;
        c = ((((((F::new(3617.0 / 10670622842880000.0) * t + F::new(1.0 / 74724249600.0)) * t
            + F::new(691.0 / 1307674368000.0))
            * t
            + F::new(1.0 / 47900160.0))
            * t
            + F::new(1.0 / 1209600.0))
            * t
            + F::new(1.0 / 30240.0))
            * t
            + F::new(1.0 / 720.0))
            * t
            + F::new(1.0 / 12.0);
        c_p = ((((((F::new(43867.0 / 319318388573184000.0) * t
            + F::new(3617.0 / 762187345920000.0))
            * t
            + F::new(1.0 / 6227020800.0))
            * t
            + F::new(691.0 / 130767436800.0))
            * t
            + F::new(1.0 / 5987520.0))
            * t
            + F::new(1.0 / 201600.0))
            * t
            + F::new(1.0 / 7560.0))
            * t
            + F::new(1.0 / 360.0);
    }

    LogJacobianCoeffs::<F> { c, c_p }
}

// Calculate the SO3 logarithm, returning both the vectors and their square norms.
// ```
// r ↦  Log(ω)
// ```
#[cube(launch)]
fn log_kernel<F: Float>(r: &Tensor<F>, omega: &mut Tensor<F>) {
    // correctly index
    let b = ABSOLUTE_POS_X as usize;
    if b >= r.shape(0) {
        terminate!();
    }

    // unpack the components
    let r_sb = r.stride(0);
    let r_sd = r.stride(1);
    let r11 = r[b * r_sb];
    let r12 = r[b * r_sb + r_sd];
    let r13 = r[b * r_sb + 2 * r_sd];
    let r21 = r[b * r_sb + 3 * r_sd];
    let r22 = r[b * r_sb + 4 * r_sd];
    let r23 = r[b * r_sb + 5 * r_sd];
    let r31 = r[b * r_sb + 6 * r_sd];
    let r32 = r[b * r_sb + 7 * r_sd];
    let r33 = r[b * r_sb + 8 * r_sd];

    // calculate terms which show up in logarithm
    let r_trace = r11 + r22 + r33;
    let one = F::new(1.0);
    let half = F::new(0.5);
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

    // allocate the outputs and fill them conditionally for numerical stability
    let eps = F::new(1e-6);
    let (w_x, w_y, w_z) = if t_sq < eps {
        // Taylor series for theta / (2 * sin(theta)) near 0: 1/2 + theta^2 / 12
        let mult = F::new(0.5) + t_sq / F::new(12.0);
        (a_x * mult, a_y * mult, a_z * mult)
    } else if cos_theta < F::new(-0.999) {
        // Robust evaluation for theta near pi
        let one_minus_cos = one - cos_theta;

        let s11 = r11 - cos_theta;
        let s22 = r22 - cos_theta;
        let s33 = r33 - cos_theta;

        let (u_x, u_y, u_z) = if s11 > s22 && s11 > s33 {
            let u_x = (s11 / one_minus_cos).sqrt();
            let denom = F::new(2.0) * one_minus_cos * u_x;
            let u_y = (r12 + r21) / denom;
            let u_z = (r13 + r31) / denom;
            (u_x, u_y, u_z)
        } else if s22 > s33 {
            let u_y = (s22 / one_minus_cos).sqrt();
            let denom = F::new(2.0) * one_minus_cos * u_y;
            let u_x = (r12 + r21) / denom;
            let u_z = (r23 + r32) / denom;
            (u_x, u_y, u_z)
        } else {
            let u_z = (s33 / one_minus_cos).sqrt();
            let denom = F::new(2.0) * one_minus_cos * u_z;
            let u_x = (r13 + r31) / denom;
            let u_y = (r23 + r32) / denom;
            (u_x, u_y, u_z)
        };

        // Align the sign of u with a = (R - R^T)
        let dot = u_x * a_x + u_y * a_y + u_z * a_z;
        if dot < F::new(0.0) {
            (-theta * u_x, -theta * u_y, -theta * u_z)
        } else {
            (theta * u_x, theta * u_y, theta * u_z)
        }
    } else {
        // Standard evaluation inside safe region
        let mult = theta / (F::new(2.0) * theta.sin());
        (a_x * mult, a_y * mult, a_z * mult)
    };

    // pack the components
    let t_sb = omega.stride(0);
    let t_sd = omega.stride(1);
    omega[b * t_sb] = w_x;
    omega[b * t_sb + t_sd] = w_y;
    omega[b * t_sb + 2 * t_sd] = w_z;
}

// Calculate the JVP of the SO3 logarithm.
// ```
// (ω, v) ↦ J(ω)⋅v
// ```
// where `J(ω)` is our right-Jacobian of the Riemannian logarithm map.
// ```
// J(ω) = I + (1/2) ω^ + (1/|ω|²) (1 - (|ω|/2) cot(|ω|/2))) (ω^)²
// ```
// This calculation simplifies like so.
// ```
// J(ω)⋅v = v + (1/2) (ω × v) + c(|ω|) (ω × (ω × v))
// ```
#[cube(launch)]
fn log_jvp_kernel<F: Float>(omega: &Tensor<F>, v: &Tensor<F>, out: &mut Tensor<F>) {
    // correctly index
    let b = ABSOLUTE_POS_X as usize;
    let f = ABSOLUTE_POS_Y as usize;
    if b >= omega.shape(0) || f >= v.shape(2) {
        terminate!();
    }

    // unpack the components
    let w_sb = omega.stride(0);
    let w_sd = omega.stride(1);
    let w_x = omega[b * w_sb];
    let w_y = omega[b * w_sb + w_sd];
    let w_z = omega[b * w_sb + 2 * w_sd];

    let v_sb = v.stride(0);
    let v_sd = v.stride(1);
    let v_sf = v.stride(2);
    let v_x = v[b * v_sb + f * v_sf];
    let v_y = v[b * v_sb + v_sd + f * v_sf];
    let v_z = v[b * v_sb + 2 * v_sd + f * v_sf];

    // |ω|²
    let theta_sq = w_x * w_x + w_y * w_y + w_z * w_z;

    // c(|ω|)
    let c = log_jacobian_coeffs::<F>(theta_sq).c;

    // ω × v
    let (wxv_x, wxv_y, wxv_z) = cross_product::<F>(w_x, w_y, w_z, v_x, v_y, v_z);

    // ω × (ω × v)
    let (wxwxv_x, wxwxv_y, wxwxv_z) = cross_product::<F>(w_x, w_y, w_z, wxv_x, wxv_y, wxv_z);

    // pack the components
    let o_sb = out.stride(0);
    let o_sd = out.stride(1);
    let o_sf = out.stride(2);
    let half = F::new(0.5);
    out[b * o_sb + f * o_sf] = v_x + half * wxv_x + c * wxwxv_x;
    out[b * o_sb + o_sd + f * o_sf] = v_y + half * wxv_y + c * wxwxv_y;
    out[b * o_sb + 2 * o_sd + f * o_sf] = v_z + half * wxv_z + c * wxwxv_z;
}

// Calculate the VJP of the SO3 logarithm.
// ```
// (ω, α) ↦ J(ω)⊤⋅α
// ```
// where `J(ω)` is our right-Jacobian of the Riemannian logarithm map.
// ```
// J(ω) = I + (1/2) ω^ + (1/|ω|²) (1 - (|ω|/2) cot(|ω|/2))) (ω^)²
// ```
// This calculation simplifies like so.
// ```
// J(ω)⊤⋅α = α - (1/2) (ω × α) + c(|ω|) (ω × (ω × α))
// ```
#[cube(launch)]
fn log_vjp_kernel<F: Float>(omega: &Tensor<F>, v: &Tensor<F>, out: &mut Tensor<F>) {
    // correctly index
    let b = ABSOLUTE_POS_X as usize;
    let f = ABSOLUTE_POS_Y as usize;
    if b >= omega.shape(0) || f >= v.shape(2) {
        terminate!();
    }

    // unpack the components
    let w_sb = omega.stride(0);
    let w_sd = omega.stride(1);
    let w_x = omega[b * w_sb];
    let w_y = omega[b * w_sb + w_sd];
    let w_z = omega[b * w_sb + 2 * w_sd];

    let v_sb = v.stride(0);
    let v_sd = v.stride(1);
    let v_sf = v.stride(2);
    let v_x = v[b * v_sb + f * v_sf];
    let v_y = v[b * v_sb + v_sd + f * v_sf];
    let v_z = v[b * v_sb + 2 * v_sd + f * v_sf];

    // |ω|²
    let theta_sq = w_x * w_x + w_y * w_y + w_z * w_z;

    // c(|ω|)
    let c = log_jacobian_coeffs::<F>(theta_sq).c;

    // ω × v
    let (wxv_x, wxv_y, wxv_z) = cross_product::<F>(w_x, w_y, w_z, v_x, v_y, v_z);

    // ω × (ω × v)
    let (wxwxv_x, wxwxv_y, wxwxv_z) = cross_product::<F>(w_x, w_y, w_z, wxv_x, wxv_y, wxv_z);

    // pack the components
    let o_sb = out.stride(0);
    let o_sd = out.stride(1);
    let o_sf = out.stride(2);
    let half = F::new(0.5);
    out[b * o_sb + f * o_sf] = v_x - half * wxv_x + c * wxwxv_x;
    out[b * o_sb + o_sd + f * o_sf] = v_y - half * wxv_y + c * wxwxv_y;
    out[b * o_sb + 2 * o_sd + f * o_sf] = v_z - half * wxv_z + c * wxwxv_z;
}

// Calculate the HVP of the SO3 logarithm. In other words, we want the sensitivity to the following
// operation.
// ```
// J(R) = I + (1/2) ω^ + (1/|ω|²) (1 - (|ω|/2) cot(|ω|/2))) (ω^)²
// ```
// ```
// DvJ(R)⊤α
//  = -0.5 (η × α)
//      + c(|ω|) ( η × (ω × α) + ω × (η × α) )
//      + c_p(|ω|) (ω⊤ η) (ω × (ω × α))
// ```
// where `η = J(R)v`.
#[cube(launch)]
fn log_hvp_kernel<F: Float>(
    omega: &Tensor<F>,
    eta: &Tensor<F>,
    alpha: &Tensor<F>,
    out: &mut Tensor<F>,
) {
    // correctly index
    let b = ABSOLUTE_POS_X as usize;
    let f = ABSOLUTE_POS_Y as usize;
    if b >= omega.shape(0) || f >= alpha.shape(2) {
        terminate!();
    }

    // unpack the components
    let w_sb = omega.stride(0);
    let w_sd = omega.stride(1);
    let w_x = omega[b * w_sb];
    let w_y = omega[b * w_sb + w_sd];
    let w_z = omega[b * w_sb + 2 * w_sd];

    let e_sb = eta.stride(0);
    let e_sd = eta.stride(1);
    let e_x = eta[b * e_sb];
    let e_y = eta[b * e_sb + e_sd];
    let e_z = eta[b * e_sb + 2 * e_sd];

    let a_sb = alpha.stride(0);
    let a_sd = alpha.stride(1);
    let a_sf = alpha.stride(2);
    let a_x = alpha[b * a_sb + f * a_sf];
    let a_y = alpha[b * a_sb + a_sd + f * a_sf];
    let a_z = alpha[b * a_sb + 2 * a_sd + f * a_sf];

    // |ω|²
    let theta_sq = w_x * w_x + w_y * w_y + w_z * w_z;

    // c(|ω|), c'(|ω|)/|ω|
    let k = log_jacobian_coeffs::<F>(theta_sq);
    let c = k.c;
    let c_p = k.c_p;

    // η × α
    let (exa_x, exa_y, exa_z) = cross_product::<F>(e_x, e_y, e_z, a_x, a_y, a_z);

    // ω × α
    let (wxa_x, wxa_y, wxa_z) = cross_product::<F>(w_x, w_y, w_z, a_x, a_y, a_z);

    // ω × (ω × α)
    let (wxwxa_x, wxwxa_y, wxwxa_z) = cross_product::<F>(w_x, w_y, w_z, wxa_x, wxa_y, wxa_z);

    //  η × (ω × α)
    let (exwxa_x, exwxa_y, exwxa_z) = cross_product::<F>(e_x, e_y, e_z, wxa_x, wxa_y, wxa_z);

    //  ω × (η × α)
    let (wxexa_x, wxexa_y, wxexa_z) = cross_product::<F>(w_x, w_y, w_z, exa_x, exa_y, exa_z);

    // ω⊤ η
    let half = F::new(0.5);
    let wdote = w_x * e_x + w_y * e_y + w_z * e_z;

    // pack the components
    let o_sb = out.stride(0);
    let o_sd = out.stride(1);
    let o_sf = out.stride(2);
    out[b * o_sb + f * o_sf] = -half * exa_x + c * (exwxa_x + wxexa_x) + c_p * wdote * wxwxa_x;
    out[b * o_sb + o_sd + f * o_sf] =
        -half * exa_y + c * (exwxa_y + wxexa_y) + c_p * wdote * wxwxa_y;
    out[b * o_sb + 2 * o_sd + f * o_sf] =
        -half * exa_z + c * (exwxa_z + wxexa_z) + c_p * wdote * wxwxa_z;
}
