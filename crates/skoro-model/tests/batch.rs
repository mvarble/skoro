//! How the batch axis behaves when not every quantity carries one.
//!
//! The compiler **normalizes rather than broadcasts**, and the two halves are asymmetric:
//! a [`ConstantValue`] is rank-1 unbatched storage and gets `expand`ed to the running batch
//! size, while a `Parameter` or `Sampler` value is *checked* against it. That asymmetry is
//! what makes "one dataset, N different starting points" expressible — a multi-start
//! optimizer supplies N rows of free variables against constants stored once.
//!
//! Nothing in the workspace exercised this before. `tests/jacobian.rs` runs at batch 2, but
//! every input there is a `Parameter`, so the expansion path never executed.

use skoro_backend::{NdArrayBackend, Tensor};
use skoro_core::error::SkoroError;
use skoro_model::{
    ModelClientOps, TangentLayout,
    hash_map_builders::{point_map, points},
    jacobian::jacobian_vjp,
    model::ModelClient,
    q::Q,
};
use skoro_riemann::euclidean::{EuclideanD, VectorD};
use skoro_riemann::real::Real;

type B = NdArrayBackend<f64, i32>;

const BATCH: usize = 3;

fn tensor(rows: [[f64; 2]; BATCH]) -> Tensor<B, 2> {
    let flat: Vec<f64> = rows.into_iter().flatten().collect();
    Tensor::<B, 1>::from_floats(flat.as_slice(), &Default::default()).reshape([BATCH, 2])
}

fn into_vec(t: Tensor<B, 2>) -> Vec<f64> {
    t.into_data().to_vec().unwrap()
}

/// A constant stored once is applied to every row.
///
/// If it were *broadcast* rather than expanded the answer would be the same here; what this
/// pins is that the shapes reconcile at all, and that the same constant reaches each row
/// rather than only the first.
#[test]
fn a_constant_is_expanded_to_the_running_batch() {
    let model = ModelClient::<B>::default();
    let offset = model.add_constant(EuclideanD(2), &VectorD::new(&[10.0, -20.0]));
    let x: Q<B, EuclideanD> = model.add_parameter(EuclideanD(2));
    let y = (x.clone() - offset.clone()).unwrap();

    let values = points()
        .with_quantity(&x, tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        .build();

    assert_eq!(
        into_vec(model.evaluate(&values, &y).unwrap()),
        vec![
            1.0 - 10.0,
            2.0 + 20.0,
            3.0 - 10.0,
            4.0 + 20.0,
            5.0 - 10.0,
            6.0 + 20.0,
        ],
        "every row sees the same constant"
    );
}

/// The Jacobian path agrees: a constant contributes no column and perturbs no row.
///
/// This is the combination a multi-start optimizer actually runs — `jacobian_vjp` at batch
/// N over a model that holds constants — and it is the reason the expansion above has to
/// work on the differentiated path too, not only the forward one.
#[test]
fn the_jacobian_is_per_row_when_the_model_holds_constants() {
    let model = ModelClient::<B>::default();
    let scale = model.add_constant(Real, &2.5f64);
    let x: Q<B, EuclideanD> = model.add_parameter(EuclideanD(2));
    // `∂y/∂x` is `2.5 · I` on every row — the same on each, but reached through the
    // expanded constant.
    let y = (x.clone() * scale.clone()).unwrap();

    let values = points()
        .with_quantity(&x, tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        .build();
    let free = model.tangent_layout(&[x.id()]).unwrap();
    let rows = model.tangent_layout(&[y.id()]).unwrap();

    let j = jacobian_vjp(&model, &values, &free, &rows).unwrap();
    assert_eq!(j.dims(), [BATCH, 2, 2]);

    let j: Vec<f64> = j.into_data().to_vec().unwrap();
    for row in 0..BATCH {
        let block = &j[row * 4..(row + 1) * 4];
        assert_eq!(block, &[2.5, 0.0, 0.0, 2.5], "row {row}");
    }
}

/// Parameters are checked, not expanded: supplying one at batch 1 alongside another at
/// batch N is an error rather than a silent broadcast.
///
/// This is the caveat a multi-start caller has to know — every parameter in the map has to
/// be replicated to N rows, free or not. Asserting the variant rather than the id is
/// deliberate: the running batch size is inferred from an arbitrary `HashMap` entry, so
/// *which* quantity gets blamed is not stable across runs.
#[test]
fn a_parameter_at_the_wrong_batch_is_rejected() {
    let model = ModelClient::<B>::default();
    let x: Q<B, EuclideanD> = model.add_parameter(EuclideanD(2));
    let z: Q<B, EuclideanD> = model.add_parameter(EuclideanD(2));
    let y = (x.clone() - z.clone()).unwrap();

    let single =
        Tensor::<B, 1>::from_floats([7.0, 8.0].as_slice(), &Default::default()).reshape([1, 2]);
    let values = points()
        .with_quantity(&x, tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        .with_quantity(&z, single)
        .build();

    assert!(matches!(
        model.evaluate(&values, &y),
        Err(SkoroError::BatchSizeMismatch { .. })
    ));
}

/// A layout is a description of manifold dimensions and carries no batch, so the same one
/// serves every batch size.
#[test]
fn a_layout_is_independent_of_the_batch() {
    let model = ModelClient::<B>::default();
    let x: Q<B, EuclideanD> = model.add_parameter(EuclideanD(2));
    let layout: TangentLayout = model.tangent_layout(&[x.id()]).unwrap();

    assert_eq!(layout.total(), 2);
    assert_eq!(
        layout.identity_frames::<B>(BATCH, &Default::default())[&x.id()].dims(),
        [BATCH, 2, 2]
    );
}

/// `retract_flat` refuses a delta whose batch disagrees with the point.
///
/// This used to be the one *silently wrong* path in the area. `unflatten_vector` checks only
/// the width, so a mismatch reached the manifold, where a flat one retracts by `Addition` —
/// which broadcasts and returns a plausible wrong answer — while `SO3`/`SE3` take their
/// batch from the point and panic inside a reshape. Neither is actionable.
#[test]
fn retract_flat_rejects_a_delta_at_the_wrong_batch() {
    let model = ModelClient::<B>::default();
    let x: Q<B, EuclideanD> = model.add_parameter(EuclideanD(2));
    let layout = model.tangent_layout(&[x.id()]).unwrap();

    let points = point_map()
        .with_quantity(&x, tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        .build();

    let single =
        Tensor::<B, 1>::from_floats([0.5, 0.5].as_slice(), &Default::default()).reshape([1, 2]);
    assert_eq!(
        points.retract_flat(&layout, single).map(|p| p.len()),
        Err(SkoroError::BatchSizeMismatch {
            id: x.id(),
            expected: 1,
            found: BATCH,
        })
    );

    // The matching batch still works, and steps every row by its own tangent.
    let delta = tensor([[0.5, 0.5], [1.0, 1.0], [2.0, 2.0]]);
    let stepped = points.retract_flat(&layout, delta).unwrap();
    let (value, _) = stepped.get(&x.id()).unwrap();
    assert_eq!(
        into_vec(value.as_float().clone()),
        vec![1.5, 2.5, 4.0, 5.0, 7.0, 8.0]
    );
}
