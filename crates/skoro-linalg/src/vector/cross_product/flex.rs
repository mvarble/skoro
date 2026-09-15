use num_traits::Float;
use skoro_backend::{Backend, CotangentFrame, Flex, Point, TangentVector};
use skoro_backend::{Element, host};
use skoro_operation::{
    Input,
    diff::{DiffModeState, DiffOperation, DiffOperationForward},
    sum_opt_tensors,
};

use super::CrossProduct;

impl DiffOperation<Flex> for CrossProduct {
    type VjpState = [Option<Point<Flex>>; 2];
    type HvpState = [Option<TangentVector<Flex>>; 2];

    fn forward(
        &self,
        inputs: Vec<Input<Flex>>,
        diff_state: DiffModeState<Flex>,
    ) -> DiffOperationForward<Flex, Self::VjpState, Self::HvpState> {
        let inputs: Vec<Point<Flex>> = inputs.into_iter().map(Input::unwrap_float).collect();
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
        alpha: CotangentFrame<Flex>,
    ) -> Vec<Option<CotangentFrame<Flex>>> {
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
        alpha: CotangentFrame<Flex>,
    ) -> Vec<Option<CotangentFrame<Flex>>> {
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

// Cross-product of two 3-vectors `a` and `b`.
#[inline(always)]
pub fn cross_product_impl<F: Float>(a_x: F, a_y: F, a_z: F, b_x: F, b_y: F, b_z: F) -> (F, F, F) {
    (
        a_y * b_z - a_z * b_y,
        a_z * b_x - a_x * b_z,
        a_x * b_y - a_y * b_x,
    )
}

pub fn cross_product<B: Backend>(u: Point<B>, v: Point<B>) -> Point<B> {
    if host::is_f64(&u) {
        cross_product_typed::<B, f64>(u, v)
    } else {
        cross_product_typed::<B, f32>(u, v)
    }
}

fn cross_product_typed<B, F>(u: Point<B>, v: Point<B>) -> Point<B>
where
    B: Backend,
    F: Float + Element,
{
    // unpack data from inputs
    let [batch_size, three] = u.dims();
    let [obatch_size, othree] = v.dims();
    if batch_size != obatch_size || three != 3 || othree != 3 {
        panic!("cross_product requires tensors of shape `[b, 3]`");
    }

    // unpack tensor data
    let device = u.device();
    let dtype = u.dtype();
    let u_slice: Vec<F> = host::read(u);
    let v_slice: Vec<F> = host::read(v);

    // Allocate output memory
    let mut out_vec = Vec::with_capacity(batch_size * 3);

    // run batch-by-batch
    let uv_iter = u_slice.chunks_exact(3).zip(v_slice.chunks_exact(3));
    for (u_batch, v_batch) in uv_iter {
        let u_x = u_batch[0];
        let u_y = u_batch[1];
        let u_z = u_batch[2];
        let v_x = v_batch[0];
        let v_y = v_batch[1];
        let v_z = v_batch[2];
        let (o_x, o_y, o_z) = cross_product_impl(u_x, u_y, u_z, v_x, v_y, v_z);
        out_vec.push(o_x);
        out_vec.push(o_y);
        out_vec.push(o_z);
    }

    // pack data
    host::write(out_vec, [batch_size, 3], &device, dtype)
}

pub fn left_broadcast_cross_product<B: Backend>(
    u: Point<B>,
    alpha: CotangentFrame<B>,
) -> CotangentFrame<B> {
    if host::is_f64(&u) {
        left_broadcast_cross_product_typed::<B, f64>(u, alpha)
    } else {
        left_broadcast_cross_product_typed::<B, f32>(u, alpha)
    }
}

fn left_broadcast_cross_product_typed<B, F>(
    u: Point<B>,
    alpha: CotangentFrame<B>,
) -> CotangentFrame<B>
where
    B: Backend,
    F: Float + Element,
{
    // unpack data from inputs
    let [batch_size, three] = u.dims();
    let [obatch_size, othree, frame_size] = alpha.dims();
    if batch_size != obatch_size || three != 3 || othree != 3 {
        panic!("cross_product requires tensors of shape `[b, 3]`");
    }

    // unpack tensor data
    let device = u.device();
    let dtype = u.dtype();
    let u_slice: Vec<F> = host::read(u);
    let a_slice: Vec<F> = host::read(alpha);

    // Allocate output memory
    let mut out_vec = Vec::with_capacity(batch_size * 3 * frame_size);

    // run batch-by-batch
    let ua_iter = u_slice
        .chunks_exact(3)
        .zip(a_slice.chunks_exact(3 * frame_size));
    for (u_batch, a_batch) in ua_iter {
        let u_x = u_batch[0];
        let u_y = u_batch[1];
        let u_z = u_batch[2];

        // `a_slice` indexes `[b, 3, f]` from the right, so each chunk of size `3 * f` will
        // correspond one batch, with `x` data of length `f`, then `y` of length `f`, then `z` of
        // length `f`; the proceeding lines split these into separate arrays.
        let (a_x, rest) = a_batch.split_at(frame_size);
        let (a_y, a_z) = rest.split_at(frame_size);

        // fill the `x` data first, then the `y`, then the `z`.
        let a_iter = a_x.iter().zip(a_y).zip(a_z);
        out_vec.extend(
            a_iter
                .clone()
                .map(|((_, &a_y), &a_z)| u_y * a_z - u_z * a_y),
        );
        out_vec.extend(
            a_iter
                .clone()
                .map(|((&a_x, _), &a_z)| u_z * a_x - u_x * a_z),
        );
        out_vec.extend(a_iter.map(|((&a_x, &a_y), _)| u_x * a_y - u_y * a_x));
    }

    // pack data
    host::write(out_vec, [batch_size, 3, frame_size], &device, dtype)
}
