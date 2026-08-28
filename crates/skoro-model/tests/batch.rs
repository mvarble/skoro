//! How the batch axis behaves when not every quantity carries one.
//!
//! The compiler **normalizes rather than broadcasts**, and which half a value falls in is a
//! property of the value: a `QuantityValue::Unbatched` is rank-1 storage and gets `expand`ed to
//! the running batch size, while a `QuantityValue::Batched`, a `Parameter`, or a `Sampler` value
//! is *checked* against it. That asymmetry is what makes "one dataset, N different starting
//! points" expressible — a multi-start optimizer supplies N rows of free variables against
//! constants stored once — while `add_batched_constant` covers the opposite case, where each of
//! the N problems carries its own constant.
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

/// A constant stored once reaches every row: the compiler **expands** it to the running batch
/// rather than broadcasting, on the differentiated path as much as the forward one. A layout
/// describes manifold dimensions and carries no batch, so the same one serves every batch size.
mod expansion {
    use super::*;

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
}

/// `add_batched_constant` is the opposite case: each row of the batch carries its own value, which
/// is what makes N independent problems expressible. It is still a constant, so it contributes no
/// column to the Jacobian and no cotangent.
mod batch_pinning {
    use super::*;

    /// A batch-pinned constant gives every row its own value.
    ///
    /// The counterpart to `a_constant_is_expanded_to_the_running_batch`: there, one value reaches
    /// each row; here, each row gets a different one. A batch is a set of *independent* problems, and
    /// without this there is no way to give them independent data.
    #[test]
    fn a_batched_constant_gives_each_row_its_own_value() {
        let model = ModelClient::<B>::default();
        let offsets = model
            .add_batched_constant(
                EuclideanD(2),
                &[
                    VectorD::new(&[10.0, -20.0]),
                    VectorD::new(&[0.0, 0.0]),
                    VectorD::new(&[-1.0, 1.0]),
                ],
            )
            .unwrap();
        let x: Q<B, EuclideanD> = model.add_parameter(EuclideanD(2));
        let y = (x.clone() - offsets).unwrap();

        let values = points()
            .with_quantity(&x, tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
            .build();

        assert_eq!(
            into_vec(model.evaluate(&values, &y).unwrap()),
            vec![
                1.0 - 10.0,
                2.0 + 20.0,
                3.0 - 0.0,
                4.0 - 0.0,
                5.0 + 1.0,
                6.0 - 1.0,
            ]
        );
    }

    /// A batch-pinned constant is still a constant: it carries no derivative.
    ///
    /// It reaches the pass through the same `CompiledStep::Constant` as any other, so
    /// `is_differentiable_parent` never sees it as a free variable and no cotangent is collected for
    /// it. What the Jacobian sees is `∂y/∂x = I` on every row, undisturbed by the per-row offsets.
    #[test]
    fn a_batched_constant_carries_no_derivative() {
        let model = ModelClient::<B>::default();
        let offsets = model
            .add_batched_constant(
                EuclideanD(2),
                &[
                    VectorD::new(&[10.0, -20.0]),
                    VectorD::new(&[0.0, 0.0]),
                    VectorD::new(&[-1.0, 1.0]),
                ],
            )
            .unwrap();
        let x: Q<B, EuclideanD> = model.add_parameter(EuclideanD(2));
        let y = (x.clone() - offsets).unwrap();

        let values = points()
            .with_quantity(&x, tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
            .build();
        let columns = model.tangent_layout(&[x.id()]).unwrap();
        let rows = model.tangent_layout(&[y.id()]).unwrap();

        let jacobian = jacobian_vjp(&model, &values, &columns, &rows).unwrap();
        assert_eq!(jacobian.dims(), [BATCH, 2, 2]);
        assert_eq!(
            into_vec(jacobian.reshape([BATCH * 2, 2])),
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
        );
    }
}

/// What is checked rather than expanded: a parameter, a batch-pinned constant, or a `retract_flat`
/// delta whose batch disagrees. The batch-1 case is the one worth pinning -- a single row looks
/// broadcastable, and a rule where batch 1 silently widens is how a caller solves one problem N
/// times without being told.
mod rejections {
    use super::*;

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
                expected: 1,
                found: BATCH,
            })
        );

        // The matching batch still works, and steps every row by its own tangent.
        let delta = tensor([[0.5, 0.5], [1.0, 1.0], [2.0, 2.0]]);
        let stepped = points.retract_flat(&layout, delta).unwrap();
        let (value, _) = stepped.get(&x.id()).unwrap();
        assert_eq!(
            into_vec(value.unwrap_float_ref().clone()),
            vec![1.5, 2.5, 4.0, 5.0, 7.0, 8.0]
        );
    }

    /// A batch-pinned constant is *checked* against the running batch, in both directions.
    ///
    /// The batch-1 case is the one worth pinning: it is the tempting relaxation, since a single row
    /// looks broadcastable. Broadcasting is what `add_constant` already is, and a rule where batch 1
    /// silently widens but batch 5 errors is how a caller ends up solving one problem N times without
    /// being told.
    #[test]
    fn a_batched_constant_at_the_wrong_batch_is_rejected() {
        let model = ModelClient::<B>::default();
        let offsets = model
            .add_batched_constant(
                EuclideanD(2),
                &[
                    VectorD::new(&[10.0, -20.0]),
                    VectorD::new(&[0.0, 0.0]),
                    VectorD::new(&[-1.0, 1.0]),
                ],
            )
            .unwrap();
        let x: Q<B, EuclideanD> = model.add_parameter(EuclideanD(2));
        let y = (x.clone() - offsets.clone()).unwrap();

        let single =
            Tensor::<B, 1>::from_floats([7.0, 8.0].as_slice(), &Default::default()).reshape([1, 2]);
        assert_eq!(
            model
                .evaluate(&points().with_quantity(&x, single).build(), &y)
                .err(),
            Some(SkoroError::BatchSizeMismatch {
                expected: 1,
                found: BATCH,
            })
        );

        let five = Tensor::<B, 1>::from_floats(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0].as_slice(),
            &Default::default(),
        )
        .reshape([5, 2]);
        assert_eq!(
            model
                .evaluate(&points().with_quantity(&x, five).build(), &y)
                .err(),
            Some(SkoroError::BatchSizeMismatch {
                expected: 5,
                found: BATCH,
            })
        );
    }
}
