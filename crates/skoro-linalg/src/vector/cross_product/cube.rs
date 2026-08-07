use burn::tensor::Shape;
use cubecl::prelude::*;
use skoro_backend::{
    CotangentFrame, CubeBackend, CubeFloat, CubeInt, CubeRuntime, CubeTensor, Point, TangentVector,
    ToCubeTensor,
};
use skoro_operation::{
    Input,
    diff::{DiffModeState, DiffOperation, DiffOperationForward},
    sum_opt_tensors,
};

use super::CrossProduct;

impl<R: CubeRuntime, F: CubeFloat, I: CubeInt> DiffOperation<CubeBackend<R, F, I>>
    for CrossProduct
{
    type VjpState = [Option<Point<CubeBackend<R, F, I>>>; 2];
    /// `du` then `dv`, each present only when that input was seeded with a tangent.
    type HvpState = [Option<TangentVector<CubeBackend<R, F, I>>>; 2];

    fn forward(
        &self,
        inputs: Vec<Input<CubeBackend<R, F, I>>>,
        diff_state: DiffModeState<CubeBackend<R, F, I>>,
    ) -> DiffOperationForward<CubeBackend<R, F, I>, Self::VjpState, Self::HvpState> {
        let inputs: Vec<Point<CubeBackend<R, F, I>>> =
            inputs.into_iter().map(Input::into_float).collect();
        let [u, v] = inputs.try_into().unwrap();
        match diff_state {
            DiffModeState::ForwardOnly => DiffOperationForward::forward_only(cross_product(u, v)),
            DiffModeState::Jvp(tvecs) => {
                let [du, dv] = tvecs.try_into().unwrap();
                DiffOperationForward::jvp(
                    cross_product(u.clone(), v.clone()),
                    sum_opt_tensors([
                        du.map(|du| -left_broadcast_cross_product(v, du)),
                        dv.map(|dv| left_broadcast_cross_product(u, dv)),
                    ]),
                )
            }
            DiffModeState::Vjp(diffs) => {
                let [udiff, vdiff] = diffs.try_into().unwrap();
                DiffOperationForward::vjp(
                    cross_product(u.clone(), v.clone()),
                    [
                        if udiff { Some(v) } else { None },
                        if vdiff { Some(u) } else { None },
                    ],
                )
            }
            DiffModeState::Hvp {
                diffs,
                tangents: tvecs,
            } => {
                let [du, dv] = tvecs.try_into().unwrap();
                let [udiff, vdiff] = diffs.try_into().unwrap();
                DiffOperationForward::hvp(
                    cross_product(u.clone(), v.clone()),
                    sum_opt_tensors([
                        du.as_ref().map(|du| cross_product(du.clone(), v.clone())),
                        dv.as_ref().map(|dv| cross_product(u.clone(), dv.clone())),
                    ]),
                    [
                        if udiff { Some(v) } else { None },
                        if vdiff { Some(u) } else { None },
                    ],
                    [du, dv],
                )
            }
        }
    }

    fn vjp(
        &self,
        [v, u]: &Self::VjpState,
        alpha: CotangentFrame<CubeBackend<R, F, I>>,
    ) -> Vec<Option<CotangentFrame<CubeBackend<R, F, I>>>> {
        match (v, u) {
            (Some(v), Some(u)) => vec![
                Some(left_broadcast_cross_product(v.clone(), alpha.clone())),
                Some(-left_broadcast_cross_product(u.clone(), alpha)),
            ],
            (Some(v), None) => vec![Some(left_broadcast_cross_product(v.clone(), alpha)), None],
            (None, Some(u)) => vec![None, Some(-left_broadcast_cross_product(u.clone(), alpha))],
            (None, None) => vec![None, None],
        }
    }

    fn hvp(
        &self,
        vjp_state: &Self::VjpState,
        hvp_state: &Self::HvpState,
        alpha: CotangentFrame<CubeBackend<R, F, I>>,
    ) -> Vec<Option<CotangentFrame<CubeBackend<R, F, I>>>> {
        let [udiff, vdiff] = vjp_state;
        let [du, dv] = hvp_state;
        vec![
            udiff
                .as_ref()
                .and(dv.as_ref())
                .map(|dv| left_broadcast_cross_product(dv.clone(), alpha.clone())),
            vdiff
                .as_ref()
                .and(du.as_ref())
                .map(|du| -left_broadcast_cross_product(du.clone(), alpha.clone())),
        ]
    }
}

pub mod kernel {
    use cubecl::prelude::*;

    #[cube]
    pub fn cross_product_impl<F: Float>(
        a_x: F,
        a_y: F,
        a_z: F,
        b_x: F,
        b_y: F,
        b_z: F,
    ) -> (F, F, F) {
        (
            a_y * b_z - a_z * b_y,
            a_z * b_x - a_x * b_z,
            a_x * b_y - a_y * b_x,
        )
    }

    #[cube(launch)]
    pub fn cross_product<F: Float>(u: &Tensor<F>, v: &Tensor<F>, out: &mut Tensor<F>) {
        // correctly index
        let b = ABSOLUTE_POS_X as usize;
        if b >= u.shape(0) {
            terminate!();
        }

        // unpack the components
        let u_sb = u.stride(0);
        let u_sd = u.stride(1);
        let u_x = u[b * u_sb];
        let u_y = u[b * u_sb + u_sd];
        let u_z = u[b * u_sb + 2 * u_sd];

        let v_sb = v.stride(0);
        let v_sd = v.stride(1);
        let v_x = v[b * v_sb];
        let v_y = v[b * v_sb + v_sd];
        let v_z = v[b * v_sb + 2 * v_sd];

        // fill the output
        let (o_x, o_y, o_z) = cross_product_impl::<F>(u_x, u_y, u_z, v_x, v_y, v_z);
        let o_sb = out.stride(0);
        let o_sd = out.stride(1);
        out[b * o_sb] = o_x;
        out[b * o_sb + o_sd] = o_y;
        out[b * o_sb + 2 * o_sd] = o_z;
    }

    #[cube(launch)]
    pub fn left_broadcast_cross_product<F: Float>(
        u: &Tensor<F>,
        v: &Tensor<F>,
        out: &mut Tensor<F>,
    ) {
        // correctly index
        let b = ABSOLUTE_POS_X as usize;
        let f = ABSOLUTE_POS_Y as usize;
        if b >= u.shape(0) || f >= v.shape(2) {
            terminate!();
        }

        // unpack the components
        let u_sb = u.stride(0);
        let u_sd = u.stride(1);
        let u_x = u[b * u_sb];
        let u_y = u[b * u_sb + u_sd];
        let u_z = u[b * u_sb + 2 * u_sd];

        // `v` is a frame `[b, 3, f]`; this thread owns frame column `f`, so every read has to be
        // offset by it. Dropping `f * v_sf` makes every column a copy of column 0.
        let v_sb = v.stride(0);
        let v_sd = v.stride(1);
        let v_sf = v.stride(2);
        let v_x = v[b * v_sb + f * v_sf];
        let v_y = v[b * v_sb + v_sd + f * v_sf];
        let v_z = v[b * v_sb + 2 * v_sd + f * v_sf];

        // fill the output
        let (o_x, o_y, o_z) = cross_product_impl::<F>(u_x, u_y, u_z, v_x, v_y, v_z);
        let o_sb = out.stride(0);
        let o_sd = out.stride(1);
        let o_sf = out.stride(2);
        out[b * o_sb + f * o_sf] = o_x;
        out[b * o_sb + o_sd + f * o_sf] = o_y;
        out[b * o_sb + 2 * o_sd + f * o_sf] = o_z;
    }
}

pub fn cross_product<R: CubeRuntime, F: CubeFloat, I: CubeInt>(
    u: Point<CubeBackend<R, F, I>>,
    v: Point<CubeBackend<R, F, I>>,
) -> Point<CubeBackend<R, F, I>> {
    // unpack data from inputs
    let [batch_size, three] = u.dims();
    let [obatch_size, othree] = v.dims();
    if batch_size != obatch_size || three != 3 || othree != 3 {
        panic!("cross_product requires input tensors of shape `[b, 3]`");
    }

    // cube inputs
    let device = u.device();
    let u_prim = u.to_cube_tensor();
    let v_prim = v.to_cube_tensor();
    let client = u_prim.client.clone();

    // allocate output handle
    let out_prim = CubeTensor::new_contiguous(
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
    kernel::cross_product::launch::<F, R>(
        &client,
        cubes_count,
        cube_dim,
        u_prim.into_tensor_arg(),
        v_prim.into_tensor_arg(),
        out_prim.clone().into_tensor_arg(),
    );

    // pack data
    Point::from_primitive(burn::tensor::TensorPrimitive::Float(out_prim))
}

pub fn left_broadcast_cross_product<R: CubeRuntime, F: CubeFloat, I: CubeInt>(
    u: Point<CubeBackend<R, F, I>>,
    v: CotangentFrame<CubeBackend<R, F, I>>,
) -> CotangentFrame<CubeBackend<R, F, I>> {
    // unpack data from inputs
    let [batch_size, three] = u.dims();
    let [obatch_size, othree, frame_size] = v.dims();
    if batch_size != obatch_size || three != 3 || othree != 3 {
        panic!("cross_product requires input tensors of shape `[b, 3]`");
    }

    // cube inputs
    let device = u.device();
    let u_prim = u.to_cube_tensor();
    let v_prim = v.to_cube_tensor();
    let client = u_prim.client.clone();

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
    kernel::left_broadcast_cross_product::launch::<F, R>(
        &client,
        cubes_count,
        cube_dim,
        u_prim.into_tensor_arg(),
        v_prim.into_tensor_arg(),
        out_prim.clone().into_tensor_arg(),
    );

    // pack data
    CotangentFrame::from_primitive(burn::tensor::TensorPrimitive::Float(out_prim))
}
