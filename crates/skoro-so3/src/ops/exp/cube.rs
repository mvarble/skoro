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

use super::SO3Exp;

impl<R: CubeRuntime, F: CubeFloat, I: CubeInt> DiffOperation<CubeBackend<R, F, I>> for SO3Exp {
    type VjpState = Option<Point<CubeBackend<R, F, I>>>;
    type HvpState = Option<TangentVector<CubeBackend<R, F, I>>>;

    fn forward(
        &self,
        inputs: Vec<Input<CubeBackend<R, F, I>>>,
        diff_state: DiffModeState<CubeBackend<R, F, I>>,
    ) -> DiffOperationForward<CubeBackend<R, F, I>, Self::VjpState, Self::HvpState> {
        let inputs: Vec<Point<CubeBackend<R, F, I>>> =
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
        alpha: CotangentFrame<CubeBackend<R, F, I>>,
    ) -> Vec<Option<CotangentFrame<CubeBackend<R, F, I>>>> {
        vec![state.clone().map(|omega| exp_vjp(omega, alpha))]
    }

    fn hvp(
        &self,
        vjp_state: &Self::VjpState,
        hvp_state: &Self::HvpState,
        alpha: CotangentFrame<CubeBackend<R, F, I>>,
    ) -> Vec<Option<CotangentFrame<CubeBackend<R, F, I>>>> {
        vec![
            vjp_state
                .as_ref()
                .zip(hvp_state.as_ref())
                .map(|(omega, v)| exp_hvp(omega.clone(), v.clone(), alpha)),
        ]
    }
}

pub fn exp<R: CubeRuntime, F: CubeFloat, I: CubeInt>(
    omega: Point<CubeBackend<R, F, I>>,
) -> Point<CubeBackend<R, F, I>> {
    // unpack data from inputs
    let [batch_size, three] = omega.dims();
    if three != 3 {
        panic!("exp requires a tensor of shape `[b, 3]`");
    }

    // cube inputs
    let device = omega.device();
    let omega_prim = omega.to_cube_tensor();
    let client = omega_prim.client.clone();

    // allocate output handle
    let out_prim = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::new([batch_size, 9]),
        client.empty(9 * batch_size * core::mem::size_of::<F>()),
        F::dtype(),
    );

    // launch kernel
    let threads_per_block = 256;
    let num_blocks = (batch_size as u32 + threads_per_block - 1).div_ceil(threads_per_block);
    let cube_dim = CubeDim::new_1d(threads_per_block);
    let cubes_count = CubeCount::Static(num_blocks, 1, 1);
    exp_kernel::launch::<F, R>(
        &client,
        cubes_count,
        cube_dim,
        omega_prim.into_tensor_arg(),
        out_prim.clone().into_tensor_arg(),
    );

    // pack data
    Point::from_primitive(burn::tensor::TensorPrimitive::Float(out_prim))
}

pub fn exp_jvp<R: CubeRuntime, F: CubeFloat, I: CubeInt>(
    omega: Point<CubeBackend<R, F, I>>,
    v: TangentFrame<CubeBackend<R, F, I>>,
) -> TangentFrame<CubeBackend<R, F, I>> {
    // unpack data from inputs
    let [batch_size, three] = omega.dims();
    let [obatch_size, othree, frame_size] = v.dims();
    if batch_size != obatch_size || three != 3 || othree != 3 {
        panic!("exp_jvp requires tensors of shapes `[b, 3]` and `[b, 3, f]`");
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
    exp_jvp_kernel::launch::<F, R>(
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

pub fn exp_vjp<R: CubeRuntime, F: CubeFloat, I: CubeInt>(
    omega: Point<CubeBackend<R, F, I>>,
    alpha: CotangentFrame<CubeBackend<R, F, I>>,
) -> CotangentFrame<CubeBackend<R, F, I>> {
    // unpack data from inputs
    let [batch_size, three] = omega.dims();
    let [obatch_size, othree, frame_size] = alpha.dims();
    if batch_size != obatch_size || three != 3 || othree != 3 {
        panic!("exp_vjp requires tensors of shapes `[b, 3]` and `[b, 3, f]`");
    }

    // cube inputs
    let device = omega.device();
    let omega_prim = omega.to_cube_tensor();
    let alpha_prim = alpha.to_cube_tensor();
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
    exp_vjp_kernel::launch::<F, R>(
        &client,
        cubes_count,
        cube_dim,
        omega_prim.into_tensor_arg(),
        alpha_prim.into_tensor_arg(),
        out_prim.clone().into_tensor_arg(),
    );

    // pack data
    TangentFrame::from_primitive(burn::tensor::TensorPrimitive::Float(out_prim))
}

pub fn exp_hvp<R: CubeRuntime, F: CubeFloat, I: CubeInt>(
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
        panic!("exp_vjp requires tensors of shapes `[b, 3]`, `[b, 3]`, and `[b, 3, f]`");
    }

    // cube inputs
    let device = omega.device();
    let omega_prim = omega.to_cube_tensor();
    let v_prim = v.to_cube_tensor();
    let alpha_prim = alpha.to_cube_tensor();
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
    exp_hvp_kernel::launch::<F, R>(
        &client,
        cubes_count,
        cube_dim,
        omega_prim.into_tensor_arg(),
        v_prim.into_tensor_arg(),
        alpha_prim.into_tensor_arg(),
        out_prim.clone().into_tensor_arg(),
    );

    // pack data
    TangentFrame::from_primitive(burn::tensor::TensorPrimitive::Float(out_prim))
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
// This CubeCL kernel will compute the scalar coefficients showing up in each of these expressions,
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
#[cube]
fn get_so3_coeffs<F: Float>(theta_sq: F) -> (F, F, F, F, F) {
    // Default to Taylor series expansions at theta = 0
    let mut c1 = F::new(1.0);
    let mut c2 = F::new(0.5);
    let mut c3 = F::new(1.0 / 6.0);
    let mut c2_p = F::new(-1.0 / 12.0);
    let mut c3_p = F::new(-1.0 / 60.0);

    // For non-small theta, evaluate actual trig functions.
    if theta_sq > F::new(1e-6) {
        let theta = theta_sq.sqrt();
        let sin_th = theta.sin();
        let cos_th = theta.cos();
        let th_sq = theta_sq;
        let th_cubed = th_sq * theta;
        let th_quad = th_sq * th_sq;
        let th_fifth = th_quad * theta;

        c1 = sin_th / theta;
        c2 = (F::new(1.0) - cos_th) / th_sq;
        c3 = (theta - sin_th) / th_cubed;
        c2_p = (theta * sin_th - F::new(2.0) * (F::new(1.0) - cos_th)) / th_quad;
        c3_p = (theta * (F::new(1.0) - cos_th) - F::new(3.0) * (theta - sin_th)) / th_fifth;
    }

    (c1, c2, c3, c2_p, c3_p)
}

// Calculate the SO3 exponential.
// ```
// ω ↦  Exp(ω) = I + c1(|ω|) ω^ + c2(|ω|) (ω^)²
// ```
#[cube(launch)]
fn exp_kernel<F: Float>(omega: &Tensor<F>, out: &mut Tensor<F>) {
    // correctly index
    let b = ABSOLUTE_POS_X as usize;
    if b >= omega.shape(0) {
        terminate!();
    }

    // unpack the components
    let w_sb = omega.stride(0);
    let w_sd = omega.stride(1);
    let w_x = omega[b * w_sb];
    let w_y = omega[b * w_sb + w_sd];
    let w_z = omega[b * w_sb + 2 * w_sd];

    // c1(|ω|), c2(|ω|)
    let th_sq = w_x * w_x + w_y * w_y + w_z * w_z;
    let (c1, c2, _, _, _) = get_so3_coeffs::<F>(th_sq);

    // pack the components
    let o_sb = out.stride(0);
    let o_sd = out.stride(1);
    let o = F::new(1.0);
    out[b * o_sb] = o - c2 * (w_y * w_y + w_z * w_z);
    out[b * o_sb + o_sd] = -c1 * w_z + c2 * w_x * w_y;
    out[b * o_sb + 2 * o_sd] = c1 * w_y + c2 * w_x * w_z;
    out[b * o_sb + 3 * o_sd] = c1 * w_z + c2 * w_x * w_y;
    out[b * o_sb + 4 * o_sd] = o - c2 * (w_x * w_x + w_z * w_z);
    out[b * o_sb + 5 * o_sd] = -c1 * w_x + c2 * w_y * w_z;
    out[b * o_sb + 6 * o_sd] = -c1 * w_y + c2 * w_x * w_z;
    out[b * o_sb + 7 * o_sd] = c1 * w_x + c2 * w_y * w_z;
    out[b * o_sb + 8 * o_sd] = o - c2 * (w_x * w_x + w_y * w_y);
}

// Calculate the JVP of the SO3 exponential.
// ```
// (ω, v) ↦ J(ω)⋅v
// ```
// where `J(ω)` is our right-Jacobian of the Riemannian exponential retraction.
// ```
// J(ω) = I - ((1 - cos|ω|)/|ω|²) ω^ + ((|ω| - sin|ω|)/|ω|³) (ω^)²
// ```
// This calculation simplifies like so.
// ```
// J(ω)⋅v = v - c2(|ω|) (ω × v) + c3(|ω|) (ω × (ω × v))
// ```
#[cube(launch)]
fn exp_jvp_kernel<F: Float>(omega: &Tensor<F>, v: &Tensor<F>, out: &mut Tensor<F>) {
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

    // c2(|ω|), c3(|ω|)
    let th_sq = w_x * w_x + w_y * w_y + w_z * w_z;
    let (_, c2, c3, _, _) = get_so3_coeffs::<F>(th_sq);

    // ω × v
    let (wxv_x, wxv_y, wxv_z) = cross_product::<F>(w_x, w_y, w_z, v_x, v_y, v_z);

    // ω × (ω × v)
    let (wxwxv_x, wxwxv_y, wxwxv_z) = cross_product::<F>(w_x, w_y, w_z, wxv_x, wxv_y, wxv_z);

    // pack the components
    let o_sb = out.stride(0);
    let o_sd = out.stride(1);
    let o_sf = out.stride(2);
    out[b * o_sb + f * o_sf] = v_x - c2 * wxv_x + c3 * wxwxv_x;
    out[b * o_sb + o_sd + f * o_sf] = v_y - c2 * wxv_y + c3 * wxwxv_y;
    out[b * o_sb + 2 * o_sd + f * o_sf] = v_z - c2 * wxv_z + c3 * wxwxv_z;
}

// Calculate the VJP of the SO3 exponential.
// ```
// (ω, α) ↦ J(ω)⊤⋅α
// ```
// where `J(ω)` is our right-Jacobian of the Riemannian exponential retraction.
// ```
// J(ω) = I - ((1 - cos|ω|)/|ω|²) ω^ + ((|ω| - sin|ω|)/|ω|³) (ω^)²
// ```
// This calculation simplifies like so.
// ```
// J(ω)⊤⋅α = α + c2(|ω|) (ω × v) + c3(|ω|) (ω × (ω × v))
// ```
#[cube(launch)]
fn exp_vjp_kernel<F: Float>(omega: &Tensor<F>, alpha: &Tensor<F>, out: &mut Tensor<F>) {
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
    let a_sb = alpha.stride(0);
    let a_sd = alpha.stride(1);
    let a_sf = alpha.stride(2);
    let a_x = alpha[b * a_sb + f * a_sf];
    let a_y = alpha[b * a_sb + a_sd + f * a_sf];
    let a_z = alpha[b * a_sb + 2 * a_sd + f * a_sf];

    // c2(|ω|), c3(|ω|)
    let th_sq = w_x * w_x + w_y * w_y + w_z * w_z;
    let (_, c2, c3, _, _) = get_so3_coeffs::<F>(th_sq);

    // ω × α
    let (wxa_x, wxa_y, wxa_z) = cross_product::<F>(w_x, w_y, w_z, a_x, a_y, a_z);

    // ω × (ω × α)
    let (wxwxa_x, wxwxa_y, wxwxa_z) = cross_product::<F>(w_x, w_y, w_z, wxa_x, wxa_y, wxa_z);

    // pack the components
    let o_sb = out.stride(0);
    let o_sd = out.stride(1);
    let o_sf = out.stride(2);
    out[b * o_sb + f * o_sf] = a_x + c2 * wxa_x + c3 * wxwxa_x;
    out[b * o_sb + o_sd + f * o_sf] = a_y + c2 * wxa_y + c3 * wxwxa_y;
    out[b * o_sb + 2 * o_sd + f * o_sf] = a_z + c2 * wxa_z + c3 * wxwxa_z;
}

// Calculate the HVP of the SO3 exponential.
// ```
// (ω, v, α) ↦  DvJ(ω)⋅α
// ```
// where `J(ω)` is our right-Jacobian of the Riemannian exponential retraction,
// ```
// J(ω) = I - ((1 - cos|ω|)/|ω|²) ω^ + ((|ω| - sin|ω|)/|ω|³) (ω^)²
// ```
// and `DvJ(ω)` is our sensitivity of it in the direction of `v`.
// ```
// DvJ(ω)α
//     = ((1 - cos |ω|) / |ω|²) (v × α)
//         + ((|ω| - sin |ω|) / |ω|³) (v × (ω × α) + ω × (v × α))
//         + (v⊤ ω) (
//             ((|ω| sin |ω| - 2 (1 - cos |ω|)) / |ω|⁴) (ω × α)
//             + ((|ω| (1 - cos |ω|) - 3 (|ω| - sin |ω|)) / |ω|⁵) (ω × (ω × α)))
//     = c2(|ω|) (v × α)
//         + c3(|ω|) (v × (ω × α) + ω × (v × α))
//         +  (v⊤ ω) (c2'(|ω|) (ω × α) + c3'(|ω|) (ω × (ω × α)))
//
// ```
#[cube(launch)]
fn exp_hvp_kernel<F: Float>(
    omega: &Tensor<F>,
    v: &Tensor<F>,
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
    let v_sb = v.stride(0);
    let v_sd = v.stride(1);
    let v_x = v[b * v_sb];
    let v_y = v[b * v_sb + v_sd];
    let v_z = v[b * v_sb + 2 * v_sd];
    let a_sb = alpha.stride(0);
    let a_sd = alpha.stride(1);
    let a_sf = alpha.stride(2);
    let a_x = alpha[b * a_sb + f * a_sf];
    let a_y = alpha[b * a_sb + a_sd + f * a_sf];
    let a_z = alpha[b * a_sb + 2 * a_sd + f * a_sf];

    // c2(|ω|), c3(|ω|), c2'(|ω|), c3'(|ω|)
    let th_sq = w_x * w_x + w_y * w_y + w_z * w_z;
    let (_, c2, c3, c2_p, c3_p) = get_so3_coeffs::<F>(th_sq);

    // ω × a
    let (wxa_x, wxa_y, wxa_z) = cross_product::<F>(w_x, w_y, w_z, a_x, a_y, a_z);

    // v × α
    let (vxa_x, vxa_y, vxa_z) = cross_product::<F>(v_x, v_y, v_z, a_x, a_y, a_z);

    // v × (ω × α)
    let (vxwxa_x, vxwxa_y, vxwxa_z) = cross_product::<F>(v_x, v_y, v_z, wxa_x, wxa_y, wxa_z);

    // ω × (v × α)
    let (wxvxa_x, wxvxa_y, wxvxa_z) = cross_product::<F>(w_x, w_y, w_z, vxa_x, vxa_y, vxa_z);

    // ω × (ω × α)
    let (wxwxa_x, wxwxa_y, wxwxa_z) = cross_product::<F>(w_x, w_y, w_z, wxa_x, wxa_y, wxa_z);

    // ω⊤ v
    let w_dot_v = w_x * v_x + w_y * v_y + w_z * v_z;

    // pack components
    let o_sb = out.stride(0);
    let o_sd = out.stride(1);
    let o_sf = out.stride(2);
    out[b * o_sb + f * o_sf] =
        c2 * vxa_x + c3 * (vxwxa_x + wxvxa_x) + w_dot_v * (c2_p * wxa_x + c3_p * wxwxa_x);
    out[b * o_sb + o_sd + f * o_sf] =
        c2 * vxa_y + c3 * (vxwxa_y + wxvxa_y) + w_dot_v * (c2_p * wxa_y + c3_p * wxwxa_y);
    out[b * o_sb + 2 * o_sd + f * o_sf] =
        c2 * vxa_z + c3 * (vxwxa_z + wxvxa_z) + w_dot_v * (c2_p * wxa_z + c3_p * wxwxa_z);
}
