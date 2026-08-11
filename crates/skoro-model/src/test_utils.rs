use skoro_backend::{Backend, ElementConversion, Tensor, TensorData};
use skoro_graph::quantity::ContinuousQuantity;
use skoro_operation::{Input, ValueMap};
use skoro_riemann::DynManifold;

use crate::{
    hash_map_builders::{cotangent_frames, tangent_frames, tangents},
    q::Q,
};

pub fn tensor<B: Backend, const D: usize>(
    data: impl Into<TensorData>,
    device: &B::Device,
) -> Tensor<B, D> {
    Tensor::from_data(data.into().convert::<B::FloatElem>(), device)
}

pub fn into_vec<B: Backend, const D: usize>(t: Tensor<B, D>) -> Vec<B::FloatElem> {
    let data = t.into_data().convert::<B::FloatElem>();
    data.to_vec().unwrap()
}

pub fn into_ndarray<B: Backend, const D: usize>(t: Tensor<B, D>) -> ndarray::ArrayD<B::FloatElem> {
    let dims = t.dims();
    ndarray::Array1::from_vec(into_vec(t))
        .into_shape_with_order(ndarray::IxDyn(&dims))
        .unwrap()
}

fn identity_matrix<B: Backend>(dim: usize) -> Tensor<B, 3> {
    let mut tdata = vec![B::FloatElem::from_elem(0.0f64); dim * dim];
    for i in 0..dim {
        tdata[i * (dim + 1)] = B::FloatElem::from_elem(1.0f64);
    }
    Tensor::from_data(
        TensorData::new(tdata, [1, dim, dim]).convert::<B::FloatElem>(),
        &B::Device::default(),
    )
}

pub fn test_jvp_numerically<B, InQty, OutQty>(
    inqty: &Q<B, InQty>,
    outqty: &Q<B, OutQty>,
    mut point: ValueMap<B>,
    delta: B::FloatElem,
    epsilon: <B::FloatElem as approx::AbsDiffEq>::Epsilon,
) where
    B: Backend,
    InQty: ContinuousQuantity<B>,
    OutQty: ContinuousQuantity<B>,
    B::FloatElem: approx::AbsDiffEq<B::FloatElem, Epsilon = B::FloatElem> + Copy,
{
    // build batched identity matrix [b, n, n]
    let intype = inqty.get_quantity_type();
    let outtype = outqty.get_quantity_type();
    let indim = intype.dimension();
    let outdim = outtype.dimension();
    let batches = point
        .get(&inqty.id)
        .expect("test_jvp_numerically should have input quantity ponit")
        .dims()[0];
    let ident = identity_matrix(indim).repeat_dim(0, batches);

    // evaluate JVP where the relevant input quantity has [b, n, n] identity tangent frame
    let ts = tangent_frames().with_quantity(inqty, ident.clone());
    let mut eval = outqty
        .evaluate_jvp(&point, &ts)
        .expect("test_jvp_numerically should be able to evaluate output quantity");
    let y0 = eval.value;
    let dydx = eval.tangents.remove(&outqty.id).unwrap();

    // get numerical comparison by calculating `Rinv(f(x), f(R(x, δ di))) / δ` for each component
    // vector `di`
    //
    // Compare each one to JVP output multiplied by `di`.
    let dx = ident.mul_scalar(delta);
    let x0 = point
        .remove(&inqty.id)
        .expect("test_jvp_numerically should have input quantity in value-map.")
        .into_float();
    for i in 0..indim {
        let vi = dx.clone().slice_dim(2, i).squeeze_dim(2);
        let xi = intype.retract(x0.clone(), vi.clone());
        let _ = point.insert(inqty.id, Input::Float(xi));
        let yi = outqty
            .evaluate(&point)
            .expect("test_jvp_numerically should be able to evaluate output quantity");
        let dydxvi_approx = outtype.inverse_retract(y0.clone(), yi).div_scalar(delta);
        let dydxvi_approx = ndarray::Array1::<B::FloatElem>::from_vec(into_vec(dydxvi_approx))
            .into_shape_with_order([batches, outdim])
            .unwrap();
        let dydxvi: Tensor<B, 2> = dydx.clone().slice_dim(2, i).squeeze_dim(2);
        let dydxvi = ndarray::Array1::<B::FloatElem>::from_vec(into_vec(dydxvi))
            .into_shape_with_order([batches, outdim])
            .unwrap();

        approx::assert_abs_diff_eq!(dydxvi, dydxvi_approx, epsilon = epsilon);
    }
}

pub fn test_vjp_numerically<B, InQty, OutQty>(
    inqty: &Q<B, InQty>,
    outqty: &Q<B, OutQty>,
    mut point: ValueMap<B>,
    delta: B::FloatElem,
    epsilon: <B::FloatElem as approx::AbsDiffEq>::Epsilon,
) where
    B: Backend,
    InQty: ContinuousQuantity<B>,
    OutQty: ContinuousQuantity<B>,
    B::FloatElem: approx::AbsDiffEq<B::FloatElem, Epsilon = B::FloatElem> + Copy,
{
    // unpack shapes
    let intype = inqty.get_quantity_type();
    let outtype = outqty.get_quantity_type();
    let indim = intype.dimension();
    let outdim = outtype.dimension();
    let batches = point
        .get(&inqty.id)
        .expect("test_vjp_numerically should have input quantity ponit")
        .dims()[0];

    // evaluate VJP where the output quantity has [b, m, m] identity cotangent frame
    let cots =
        cotangent_frames().with_quantity(outqty, identity_matrix(outdim).repeat_dim(0, batches));
    let mut eval = outqty
        .evaluate_vjp(&point, &cots)
        .expect("test_vjp_numerically should be able to evaluate output quantity");
    let y0 = eval.value;
    let dydxt = eval.cotangents.remove(&inqty.id).unwrap();

    // get numerical comparison by calculating `Rinv(f(x), f(R(x, δ di))) / δ` for each component
    // vector `di`
    //
    // Compare each one to VJP output multiplied by `di`.
    let dx = identity_matrix(indim)
        .repeat_dim(0, batches)
        .mul_scalar(delta);
    let x0 = point
        .remove(&inqty.id)
        .expect("test_vjp_numerically should have input quantity in value-map.")
        .into_float();
    for i in 0..indim {
        let vi = dx.clone().slice_dim(2, i).squeeze_dim(2);
        let xi = intype.retract(x0.clone(), vi.clone());
        let _ = point.insert(inqty.id, Input::Float(xi));
        let yi = outqty
            .evaluate(&point)
            .expect("test_vjp_numerically should be able to evaluate output quantity");
        let dydxvi_approx = outtype.inverse_retract(y0.clone(), yi).div_scalar(delta);
        let dydxvi_approx = ndarray::Array1::<B::FloatElem>::from_vec(into_vec(dydxvi_approx))
            .into_shape_with_order([batches, outdim])
            .unwrap();
        let vitdydxt = dydxt.clone().slice_dim(1, i);
        let vitdydxt = ndarray::Array1::<B::FloatElem>::from_vec(into_vec(vitdydxt))
            .into_shape_with_order([batches, outdim])
            .unwrap();

        approx::assert_abs_diff_eq!(vitdydxt, dydxvi_approx, epsilon = epsilon);
    }
}

/// Numerically check the Hessian-vector product of `outqty` with respect to `inqty`, seeding the
/// order-1 tangent on `inqty` itself.
///
/// This only ever exercises the *diagonal* block `∂²/∂inqty²`. Use
/// [`test_hvp_cross_numerically`] to reach the mixed blocks of a multi-input operation.
pub fn test_hvp_numerically<B, InQty, OutQty>(
    inqty: &Q<B, InQty>,
    outqty: &Q<B, OutQty>,
    point: ValueMap<B>,
    delta: B::FloatElem,
    epsilon: <B::FloatElem as approx::AbsDiffEq>::Epsilon,
) where
    B: Backend,
    InQty: ContinuousQuantity<B>,
    OutQty: ContinuousQuantity<B>,
    B::FloatElem: approx::AbsDiffEq<B::FloatElem, Epsilon = B::FloatElem> + Copy,
{
    test_hvp_cross_numerically(inqty, inqty, outqty, point, delta, epsilon)
}

/// Numerically check the mixed Hessian-vector product block `∂²outqty / ∂readqty ∂seedqty`.
///
/// The order-1 tangent is seeded on (and the numerical perturbation applied to) `seedqty`, while
/// the order-2 cotangent that gets compared is the one accumulated at `readqty`. Passing the same
/// quantity for both recovers [`test_hvp_numerically`].
pub fn test_hvp_cross_numerically<B, SeedQty, ReadQty, OutQty>(
    seedqty: &Q<B, SeedQty>,
    readqty: &Q<B, ReadQty>,
    outqty: &Q<B, OutQty>,
    point: ValueMap<B>,
    delta: B::FloatElem,
    epsilon: <B::FloatElem as approx::AbsDiffEq>::Epsilon,
) where
    B: Backend,
    SeedQty: ContinuousQuantity<B>,
    ReadQty: ContinuousQuantity<B>,
    OutQty: ContinuousQuantity<B>,
    B::FloatElem: approx::AbsDiffEq<B::FloatElem, Epsilon = B::FloatElem> + Copy,
{
    let inqty = seedqty;
    // build batched identity matrix [b, m, m]
    let intype = inqty.get_quantity_type();
    let outtype = outqty.get_quantity_type();
    let indim = intype.dimension();
    let outdim = outtype.dimension();
    let readdim = readqty.get_quantity_type().dimension();
    let batches = point
        .get(&inqty.id)
        .expect("test_hvp_numerically should have input quantity ponit")
        .dims()[0];
    let ident = identity_matrix(outdim).repeat_dim(0, batches);

    // iteratively evaluate HVP where the input quantity has component tangent vector
    // and output quantity has [b, m, m] identity order-1 cotangent frame
    // and [b, m, m] zero order-2 cotangent frame.
    let mut other_point = point.clone();
    let x0 = other_point
        .remove(&inqty.id)
        .expect("test_vjp_numerically should have input quantity in value-map.")
        .into_float();
    let v = identity_matrix(indim)
        .repeat_dim(0, batches)
        .mul_scalar(delta);
    let cots1 = cotangent_frames().with_quantity(outqty, ident.clone());
    let cots2 = cotangent_frames().with_tensor(
        outqty.id,
        Tensor::zeros([batches, outdim, outdim], &B::Device::default()),
    );
    for i in 0..indim {
        // evaluate VJP and HVP
        let mut t = vec![B::FloatElem::from_elem(0.0); indim];
        t[i] = B::FloatElem::from_elem(1.0);
        let t = Tensor::from_data(TensorData::new(t, [1, indim]), &B::Device::default())
            .repeat_dim(0, batches);
        let ts = tangents().with_quantity(inqty, t.clone());
        let mut eval = outqty
            .evaluate_hvp(&point, &ts, &cots1, &cots2)
            .expect("test_hvp_numerically should be able to evaluate output quantity");
        let vjp0 = eval.ord1_cotangents.remove(&readqty.id).unwrap();
        let hvp = eval.ord2_cotangents.remove(&readqty.id).unwrap();
        let hvp = ndarray::Array1::<B::FloatElem>::from_vec(into_vec(hvp))
            .into_shape_with_order([batches, readdim, outdim])
            .unwrap();

        // evaluate VJP at perturbed point
        let vi = v.clone().slice_dim(2, i).squeeze_dim(2);
        let xi = intype.retract(x0.clone(), vi.clone());
        let _ = other_point.insert(inqty.id, Input::Float(xi));
        let mut eval = outqty
            .evaluate_vjp(&other_point, &cots1)
            .expect("test_hvp_numerically should be able to evaluate output quantity");
        let vjp1 = eval.cotangents.remove(&readqty.id).unwrap();

        // compare numerical approximation
        let hvp_approx = (vjp1 - vjp0).div_scalar(delta);
        let hvp_approx = ndarray::Array1::<B::FloatElem>::from_vec(into_vec(hvp_approx))
            .into_shape_with_order([batches, readdim, outdim])
            .unwrap();

        approx::assert_abs_diff_eq!(hvp, hvp_approx, epsilon = epsilon);
    }
}
