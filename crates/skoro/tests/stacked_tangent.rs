//! A joint residual over a **heterogeneous** Markov blanket, without a product manifold.
//!
//! This is the case `docs/skoro.md` §4.1 calls the blocker for GAP 2: *"if the blanket holds more
//! than one quantity, there is no way to express a joint over several manifolds"*. The claim these
//! tests make is that a product manifold was never needed — stacking the members' **tangents**
//! into one `EuclideanD` is enough, and the members keep their own ids, storage, and children.
//!
//! It lives in the umbrella crate because it is the only one that can see `SE3` and `Tril` at the
//! same time as `ModelClient`; the individual pieces are tested in their own crates.
//!
//! The load-bearing property is **dimension, not storage**. An `SE3` member contributes 6 columns
//! and never 12, so a blanket of `SE3` + `EuclideanD(3)` stacks to 9 rather than 15. Getting that
//! wrong yields a well-shaped residual that no shape check downstream would reject.

use approx::assert_abs_diff_eq;

use skoro_backend::{NdArrayBackend, Tensor, TensorData};
use skoro_dense::{dense::Dense, factor::whitening_from_precision};
use skoro_linalg::tril::Tril;
use skoro_model::{
    ModelBuilderOps, ModelClientOps, hash_map_builders::point_map, jacobian::jacobian_vjp,
    model::ModelClient, q::Q, test_utils::into_vec,
};
use skoro_normal::joint::{JointNormal, add_joint_factor};
use skoro_riemann::euclidean::{EuclideanD, VectorD};
use skoro_se3::quantity::{SE3, SE3Pose};
use skoro_so3::quantity::SO3Matrix;

type B = NdArrayBackend<f64, i32>;

/// A pose rotated about z by `theta` and translated to `t`.
fn pose(theta: f64, t: [f64; 3]) -> SE3Pose<f64> {
    let (c, s) = (theta.cos(), theta.sin());
    SE3Pose::new(SO3Matrix([c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0]), t)
}

/// A `[1, n]` tensor holding exactly these `f64`s.
///
/// Built through [`TensorData`] rather than `Tensor::from_floats`, which round-trips its input
/// through `f32` even when the backend's element type is `f64` -- `1.0 / 3.0` comes back as
/// `0.3333333432674408`. That costs about eight digits, which is invisible against a `1e-5`
/// tolerance but not against the exact agreement asserted below.
fn row(values: &[f64]) -> Tensor<B, 2> {
    let n = values.len();
    Tensor::<B, 1>::from_data(TensorData::new(values.to_vec(), [n]), &Default::default())
        .reshape([1, n])
}

fn pose_row(theta: f64, t: [f64; 3]) -> Tensor<B, 2> {
    let (c, s) = (theta.cos(), theta.sin());
    row(&[c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0, t[0], t[1], t[2]])
}

/// A 9x9 precision genuinely coupled across the `SE3`/Euclidean boundary.
///
/// Diagonally dominant so it stays positive definite, but with every off-diagonal populated: a
/// block-diagonal `Λ` would pass even if the two blocks were scored independently, which is the
/// thing a joint factor exists not to do.
fn coupled_precision() -> Dense {
    let mut lambda = vec![0.0; 81];
    for i in 0..9 {
        lambda[i * 9 + i] = 4.0 + i as f64 * 0.5;
        for j in (i + 1)..9 {
            let v = 0.3 / (1.0 + (j - i) as f64);
            lambda[i * 9 + j] = v;
            lambda[j * 9 + i] = v;
        }
    }
    Dense::from_row_major(9, 9, lambda).unwrap()
}

/// That a product manifold was never needed: stacking the members' *tangents* is enough. The joint
/// is sized by **dimension** -- an `SE3` member contributes 6 columns and never 12 -- the anchors
/// are frozen snapshots rather than references to the live nodes, a perturbation lands in its own
/// block with the sign the convention says, and at one member the whole construction reduces to an
/// ordinary `inverse_retract`.
mod stacking {
    use super::*;

    /// A two-member blanket spanning two different manifolds stacks by *dimension*.
    ///
    /// `SE3` is 6-dimensional and 12 components wide; `EuclideanD(3)` is 3 of each. The joint tangent
    /// is 9 wide. A stack that used storage size would be 15 and would still evaluate, still
    /// differentiate, and still produce a plausible Jacobian — with every column after the sixth
    /// meaning something other than what the caller thinks.
    #[test]
    fn a_heterogeneous_blanket_stacks_by_dimension() {
        let model = ModelClient::<B>::default();
        let pose_q: Q<B, SE3> = model.add_parameter(SE3);
        let point_q: Q<B, EuclideanD> = model.add_parameter(EuclideanD(3));

        let at = point_map()
            .with_quantity(&pose_q, pose_row(0.4, [1.5, -2.25, 0.75]))
            .with_quantity(&point_q, row(&[1.0, 2.0, 3.0]))
            .build();

        let joint = model
            .stacked_tangent(&at, &[pose_q.id(), point_q.id()])
            .unwrap();

        assert_eq!(
            joint.get_quantity_type::<skoro_core::kind::Continuous>().0,
            9,
            "SE3 contributes its dimension (6), not its storage size (12)"
        );
    }

    /// Evaluated *at* the anchors, every block of the joint tangent is zero.
    ///
    /// True on a curved manifold as much as a flat one — `x ⊖ x` is the zero tangent — so this is the
    /// cheapest check that the anchor really was frozen from the supplied point and paired with the
    /// right quantity. A mispaired anchor would give a large residual on both blocks.
    #[test]
    fn the_joint_tangent_vanishes_at_its_anchors() {
        let model = ModelClient::<B>::default();
        let pose_q: Q<B, SE3> = model.add_parameter(SE3);
        let point_q: Q<B, EuclideanD> = model.add_parameter(EuclideanD(3));

        let at = point_map()
            .with_quantity(&pose_q, pose_row(0.4, [1.5, -2.25, 0.75]))
            .with_quantity(&point_q, row(&[1.0, 2.0, 3.0]))
            .build();
        let joint = model
            .stacked_tangent(&at, &[pose_q.id(), point_q.id()])
            .unwrap();

        let residual = into_vec(model.evaluate(at.as_ref(), &joint).unwrap());
        assert_eq!(residual.len(), 9);
        for (i, r) in residual.iter().enumerate() {
            assert_abs_diff_eq!(*r, 0.0, epsilon = 1e-12);
            assert!(r.is_finite(), "component {i}");
        }
    }

    /// Moving one member moves only its own block, and in the direction the convention says.
    ///
    /// Two things are pinned here and both are silent when wrong. The **slot**: perturbing the
    /// Euclidean member must leave the first six components — the pose block — untouched. And the
    /// **sign**: `a.inverse_retract(b)` is `b ⊖ a`, and the anchor is the first parent, so the
    /// residual is `x ⊖ p`, the displacement *from* the linearization point. The opposite convention
    /// negates every residual, which a sum-of-squares objective would happily minimize to the same
    /// point while reporting the wrong gradient direction to anything that reads it.
    #[test]
    fn a_perturbation_lands_in_its_own_block_with_the_right_sign() {
        let model = ModelClient::<B>::default();
        let pose_q: Q<B, SE3> = model.add_parameter(SE3);
        let point_q: Q<B, EuclideanD> = model.add_parameter(EuclideanD(3));

        let at = point_map()
            .with_quantity(&pose_q, pose_row(0.4, [1.5, -2.25, 0.75]))
            .with_quantity(&point_q, row(&[1.0, 2.0, 3.0]))
            .build();
        let joint = model
            .stacked_tangent(&at, &[pose_q.id(), point_q.id()])
            .unwrap();

        // Same pose, Euclidean member moved by +0.5 in its second component.
        let moved = point_map()
            .with_quantity(&pose_q, pose_row(0.4, [1.5, -2.25, 0.75]))
            .with_quantity(&point_q, row(&[1.0, 2.5, 3.0]))
            .build();

        let residual = into_vec(model.evaluate(moved.as_ref(), &joint).unwrap());
        for r in residual.iter().take(6) {
            assert_abs_diff_eq!(*r, 0.0, epsilon = 1e-12);
        }
        assert_abs_diff_eq!(residual[6], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(residual[7], 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(residual[8], 0.0, epsilon = 1e-12);
    }

    /// The anchor is a snapshot: the factor does not follow the quantity it was built from.
    ///
    /// This is the whole reason the anchors are frozen constants rather than references to the live
    /// nodes. If they tracked, the residual would stay zero no matter where the quantity moved, and
    /// the factor would contribute nothing while appearing to work.
    #[test]
    fn the_anchor_does_not_follow_the_quantity() {
        let model = ModelClient::<B>::default();
        let point_q: Q<B, EuclideanD> = model.add_parameter(EuclideanD(3));

        let at = point_map()
            .with_quantity(&point_q, row(&[1.0, 2.0, 3.0]))
            .build();
        let joint = model.stacked_tangent(&at, &[point_q.id()]).unwrap();

        let elsewhere = point_map()
            .with_quantity(&point_q, row(&[4.0, 6.0, 8.0]))
            .build();
        assert_eq!(
            into_vec(model.evaluate(elsewhere.as_ref(), &joint).unwrap()),
            vec![3.0, 4.0, 5.0]
        );
    }

    /// A blanket of one reproduces exactly the residual an ordinary `inverse_retract` gives.
    ///
    /// The bridge to the path the Gaussian suite already proves: whatever the stacked construction
    /// does, at `k = 1` it must agree component-for-component with `anchor.inverse_retract(x)` on the
    /// same manifold — including on `SE3`, where that is a genuine `Log(p⁻¹x)` rather than a
    /// subtraction.
    #[test]
    fn a_single_member_blanket_matches_inverse_retract() {
        let model = ModelClient::<B>::default();
        let pose_q: Q<B, SE3> = model.add_parameter(SE3);
        let anchor = model.add_constant(SE3, &pose(0.4, [1.5, -2.25, 0.75]));
        let direct: Q<B, EuclideanD> = anchor.inverse_retract(&pose_q).unwrap();

        let at = point_map()
            .with_quantity(&pose_q, pose_row(0.4, [1.5, -2.25, 0.75]))
            .build();
        let joint = model.stacked_tangent(&at, &[pose_q.id()]).unwrap();

        // Evaluate both somewhere far from the anchor, where `Log` is not near-linear.
        let elsewhere = point_map()
            .with_quantity(&pose_q, pose_row(0.9, [2.0, -1.0, 1.25]))
            .build();

        let stacked = into_vec(model.evaluate(elsewhere.as_ref(), &joint).unwrap());
        let expected = into_vec(model.evaluate(elsewhere.as_ref(), &direct).unwrap());
        assert_eq!(stacked.len(), 6);
        for (a, b) in stacked.iter().zip(&expected) {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-12);
        }
    }
}

/// The Jacobian of the joint residual, against central differences taken **through the
/// retraction**: on `SE3`, `x ± ε eᵢ` and `retract_flat(x, ±ε eᵢ)` are different points and only
/// the second is what `J` differentiates with respect to. The block structure is what makes it
/// discriminating -- the pose block is genuinely curved while the Euclidean block is the identity,
/// so an implementation that slid the cotangent slices by one block is not confusable numerically.
mod derivatives {
    use super::*;

    /// The Jacobian of the joint residual, against central differences taken **through the
    /// retraction**.
    ///
    /// This is the check that the per-block cotangents come back in the right *columns*. The
    /// perturbation is `retract_flat(x, ±ε eᵢ)` and not `x ± ε eᵢ`, because on `SE3` those are
    /// different points and only the first is what `J` differentiates with respect to.
    ///
    /// The block structure is what makes it discriminating: `J` is 9×9, and the pose block is
    /// genuinely curved — `∂(x ⊖ p)/∂x` is `J⁻¹` of the log, not the identity — while the Euclidean
    /// block *is* the identity. An implementation that slid the cotangent slices by one block would
    /// put a curved 6×6 where an identity 3×3 belongs, and the two are not confusable numerically even
    /// though the shapes agree.
    #[test]
    fn the_joint_jacobian_matches_central_differences() {
        let model = ModelClient::<B>::default();
        let pose_q: Q<B, SE3> = model.add_parameter(SE3);
        let point_q: Q<B, EuclideanD> = model.add_parameter(EuclideanD(3));

        let at = point_map()
            .with_quantity(&pose_q, pose_row(0.4, [1.5, -2.25, 0.75]))
            .with_quantity(&point_q, row(&[1.0, 2.0, 3.0]))
            .build();
        let joint = model
            .stacked_tangent(&at, &[pose_q.id(), point_q.id()])
            .unwrap();

        // Evaluate away from the anchors, where the pose block is far from the identity.
        let points = point_map()
            .with_quantity(&pose_q, pose_row(0.9, [2.0, -1.0, 1.25]))
            .with_quantity(&point_q, row(&[4.0, 6.0, 8.0]))
            .build();

        let columns = model.tangent_layout(&[pose_q.id(), point_q.id()]).unwrap();
        let rows = model.tangent_layout(&[joint.id()]).unwrap();
        assert_eq!((rows.total(), columns.total()), (9, 9));

        let analytic = into_vec(
            jacobian_vjp(&model, points.as_ref(), &columns, &rows)
                .unwrap()
                .reshape([9, 9]),
        );

        // Column `j` by central difference along the `j`-th basis tangent, through `retract_flat`.
        let eps = 1e-6;
        for j in 0..9 {
            let mut plus = vec![0.0; 9];
            plus[j] = eps;
            let mut minus = vec![0.0; 9];
            minus[j] = -eps;

            let forward = into_vec(
                model
                    .evaluate(
                        points.retract_flat(&columns, row(&plus)).unwrap().as_ref(),
                        &joint,
                    )
                    .unwrap(),
            );
            let backward = into_vec(
                model
                    .evaluate(
                        points.retract_flat(&columns, row(&minus)).unwrap().as_ref(),
                        &joint,
                    )
                    .unwrap(),
            );

            for i in 0..9 {
                let numeric = (forward[i] - backward[i]) / (2.0 * eps);
                assert_abs_diff_eq!(analytic[i * 9 + j], numeric, epsilon = 1e-6);
            }
        }

        // The Euclidean block is exactly the identity, and it sits in the bottom-right corner —
        // not overlapping the pose block.
        for i in 6..9 {
            for j in 0..9 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(analytic[i * 9 + j], expected, epsilon = 1e-12);
            }
        }
    }
}

/// The end-to-end claim: a joint over `SE3 × ℝ³` is expressible and correct, with no product
/// manifold anywhere. The expected density is computed on the host from the precision the factor
/// was built with, so it checks the definition rather than restating what the graph did -- and the
/// factor's two trait methods have to agree with each other as well.
mod scoring {
    use super::*;

    /// The factor's density really is `−½ δᵀ Λ δ + n` for the `δ` that stacking produces.
    ///
    /// The end-to-end claim: a joint over `SE3 × ℝ³` is expressible and correct, with no product
    /// manifold anywhere. The expectation is computed on the host from the precision the factor was
    /// built with, so it checks the *definition* rather than restating what the graph did — and `δ`
    /// comes from the graph, so a stacking that produced the wrong tangent would not agree with it.
    #[test]
    fn a_joint_factor_scores_the_quadratic_form_it_was_given() {
        let model = ModelClient::<B>::default();
        let pose_q: Q<B, SE3> = model.add_parameter(SE3);
        let point_q: Q<B, EuclideanD> = model.add_parameter(EuclideanD(3));

        let at = point_map()
            .with_quantity(&pose_q, pose_row(0.4, [1.5, -2.25, 0.75]))
            .with_quantity(&point_q, row(&[1.0, 2.0, 3.0]))
            .build();

        let lambda = coupled_precision();
        let (packed, normalizer) = whitening_from_precision(&lambda).unwrap();
        let whitening = model.add_constant(Tril(9), &VectorD::new(&packed));

        let ids = vec![pose_q.id(), point_q.id()];
        let factor = JointNormal::new(ids.clone(), at.clone(), whitening, normalizer);
        let density = factor.log_density(&model).unwrap();

        // Score it away from the anchors, where `δ` is not near zero.
        let elsewhere = point_map()
            .with_quantity(&pose_q, pose_row(0.9, [2.0, -1.0, 1.25]))
            .with_quantity(&point_q, row(&[4.0, 6.0, 8.0]))
            .build();

        let delta_node = model.stacked_tangent(&at, &ids).unwrap();
        let delta = into_vec(model.evaluate(elsewhere.as_ref(), &delta_node).unwrap());
        let d = Dense::from_row_major(9, 1, delta).unwrap();
        let quadratic = d
            .transpose()
            .matmul(&lambda.matmul(&d).unwrap())
            .unwrap()
            .get(0, 0);

        let scored = into_vec(model.evaluate(elsewhere.as_ref(), &density).unwrap())[0];
        assert_abs_diff_eq!(scored, -0.5 * quadratic + normalizer, epsilon = 1e-9);
    }

    /// The factor's two trait methods agree: the density is `−½‖r‖² + n` for the residual it reports.
    ///
    /// `JointNormal` cannot check this — it is the same obligation `Distribution` and
    /// `WhitenedResidual` already place on `Normal`. `JointNormal` discharges it by building both from
    /// one expression; this is what pins that it actually did.
    #[test]
    fn the_residual_and_the_density_agree() {
        let model = ModelClient::<B>::default();
        let point_q: Q<B, EuclideanD> = model.add_parameter(EuclideanD(3));

        let at = point_map()
            .with_quantity(&point_q, row(&[1.0, 2.0, 3.0]))
            .build();
        let lambda =
            Dense::from_row_major(3, 3, vec![4.0, 1.0, 0.5, 1.0, 3.0, -0.75, 0.5, -0.75, 2.5])
                .unwrap();
        let (packed, normalizer) = whitening_from_precision(&lambda).unwrap();
        let whitening = model.add_constant(Tril(3), &VectorD::new(&packed));

        let factor = JointNormal::new(vec![point_q.id()], at, whitening, normalizer);
        let residual = factor.whitened_residual(&model).unwrap();
        let density = factor.log_density(&model).unwrap();

        let elsewhere = point_map()
            .with_quantity(&point_q, row(&[4.0, 6.0, 8.0]))
            .build();
        let r = into_vec(model.evaluate(elsewhere.as_ref(), &residual).unwrap());
        let scored = into_vec(model.evaluate(elsewhere.as_ref(), &density).unwrap())[0];

        let sum_of_squares: f64 = r.iter().map(|x| x * x).sum();
        assert_abs_diff_eq!(scored, -0.5 * sum_of_squares + normalizer, epsilon = 1e-12);
    }
}

/// A joint factor arrives as *one* row block: a 9-member joint is 9 rows of `J`, and the residual
/// node has to survive `cleanup()`, which it reaches by a route no single-quantity factor takes.
mod records {
    use super::*;

    /// The least-squares builder records a joint factor as one row block, and keeps it alive.
    ///
    /// `residual_layout` is what an optimizer takes as the rows of `J`, so a 9-member joint has to
    /// arrive as 9 rows — and the residual node has to survive `cleanup()`, which is the hazard
    /// `residual_layout` exists for and which a joint factor reaches by a new route.
    #[test]
    fn the_builder_records_a_joint_factor_as_one_row_block() {
        let model = ModelClient::<B>::default();
        let pose_q: Q<B, SE3> = model.add_parameter(SE3);
        let point_q: Q<B, EuclideanD> = model.add_parameter(EuclideanD(3));

        let at = point_map()
            .with_quantity(&pose_q, pose_row(0.4, [1.5, -2.25, 0.75]))
            .with_quantity(&point_q, row(&[1.0, 2.0, 3.0]))
            .build();
        let (packed, normalizer) = whitening_from_precision(&coupled_precision()).unwrap();
        let whitening = model.add_constant(Tril(9), &VectorD::new(&packed));

        let factor = JointNormal::new(
            vec![pose_q.id(), point_q.id()],
            at,
            whitening.clone(),
            normalizer,
        );
        let density = add_joint_factor(&model, &factor).unwrap();

        assert_eq!(ModelBuilderOps::factors(&model).len(), 1);
        assert_eq!(ModelBuilderOps::factors(&model)[0].density, density.id());
        assert_eq!(
            ModelBuilderOps::factors(&model)[0].normalizer_parents,
            vec![whitening.id()],
            "the whitening is the normalizer's only parent"
        );

        // Nine rows of `J`: six for the pose block, three for the Euclidean one.
        assert_eq!(model.residual_layout().unwrap().total(), 9);

        model.cleanup();
        assert_eq!(
            model.residual_layout().unwrap().total(),
            9,
            "cleanup must not collect the recorded residual"
        );
    }
}

/// Retiring a joint factor, where a plain downstream walk is not enough. The factor's body is
/// downstream of the quantities it scores, so a closure at one member reaches it -- but its
/// **anchors** are not, and neither is a surviving member's per-block tangent. Both halves fail
/// silently on their own: a stranded anchor is a leak nothing reports, and a dangling record is a
/// `NodeNotFound` at the *next* solve rather than at the removal that caused it.
mod retirement {
    use super::*;

    /// Retiring a scored quantity takes the whole joint factor and leaves nothing stranded.
    ///
    /// The teardown half of a sliding window, and the reason it is not just a downstream walk. A joint
    /// factor's body — the per-block tangents, the concatenation, `Wδ`, `‖·‖²`, `ℓ` — is all
    /// downstream of the quantities it scores, so a closure at one member reaches it. Its **anchors**
    /// are not, and neither is the surviving member's per-block tangent: those sit upstream, carry no
    /// `Q`, and would be stranded once per tick forever.
    ///
    /// Both halves are asserted here, because each fails silently on its own. A stranded anchor is a
    /// leak nothing reports, and a dangling record is a `NodeNotFound` at the *next* solve
    /// rather than at the removal that caused it.
    #[test]
    fn retiring_a_scored_quantity_takes_the_whole_joint_factor() {
        let model = ModelClient::<B>::default();
        let pose_q: Q<B, SE3> = model.add_parameter(SE3);
        let point_q: Q<B, EuclideanD> = model.add_parameter(EuclideanD(3));

        let at = point_map()
            .with_quantity(&pose_q, pose_row(0.4, [1.5, -2.25, 0.75]))
            .with_quantity(&point_q, row(&[1.0, 2.0, 3.0]))
            .build();
        let (packed, normalizer) = whitening_from_precision(&coupled_precision()).unwrap();
        let whitening = model.add_constant(Tril(9), &VectorD::new(&packed));

        let node_count = || model.server_call(|server| server.graph().node_count());
        let contains = |id| model.server_call(|server| server.graph().contains(id));

        // The pose, the point, and the whitening — everything the factor is built *from*.
        let baseline = node_count();

        let factor = JointNormal::new(
            vec![pose_q.id(), point_q.id()],
            at,
            whitening.clone(),
            normalizer,
        );
        let density = factor.log_density(&model).unwrap();
        model.mark_likelihood(&density).unwrap();
        assert!(node_count() > baseline);

        // Retire the pose, as a window retires its oldest state.
        model.remove_closure(&[pose_q.id()]).unwrap();
        drop(density);
        model.cleanup();

        assert_eq!(
            node_count(),
            baseline - 1,
            "the pose, and every node the factor built — anchors included"
        );
        assert!(contains(point_q.id()), "the scored survivor is intact");
        assert!(contains(whitening.id()), "so is the whitening");
    }

    /// A retirement that deletes a recorded factor drops its record with it.
    ///
    /// The record is not merely stale afterwards, it is unusable: `residual_layout` resolves every
    /// recorded id through `TangentLayout::from_ids`, so one naming a deleted node fails the *next*
    /// solve rather than the removal that caused it. Pruning can only ever shrink the registry, so
    /// none of what makes one unforgeable is touched.
    #[test]
    fn retiring_a_scored_quantity_prunes_the_registry() {
        let model = ModelClient::<B>::default();
        let pose_q: Q<B, SE3> = model.add_parameter(SE3);
        let point_q: Q<B, EuclideanD> = model.add_parameter(EuclideanD(3));

        let at = point_map()
            .with_quantity(&pose_q, pose_row(0.4, [1.5, -2.25, 0.75]))
            .with_quantity(&point_q, row(&[1.0, 2.0, 3.0]))
            .build();
        let (packed, normalizer) = whitening_from_precision(&coupled_precision()).unwrap();
        let whitening = model.add_constant(Tril(9), &VectorD::new(&packed));

        let factor = JointNormal::new(vec![pose_q.id(), point_q.id()], at, whitening, normalizer);
        add_joint_factor(&model, &factor).unwrap();
        assert_eq!(model.residual_layout().unwrap().total(), 9);

        model.remove_closure(&[pose_q.id()]).unwrap();
        model.cleanup();

        assert!(
            ModelBuilderOps::factors(&model).is_empty(),
            "a record naming a deleted node is not evidence"
        );
        assert_eq!(model.residual_layout().unwrap().total(), 0);
    }

    /// A record whose density is gone is dropped, even though its residual node survives.
    ///
    /// Reachable by an obvious call: `add_joint_factor` hands back the log-density, and rooting a
    /// removal there takes `ℓ` alone — nothing is downstream of it. Half a factor is not a factor. It
    /// contributes nothing to the log-density any more, so leaving the record would put rows in `J`
    /// for a term the objective does not contain, which is the one way this fails without erroring.
    ///
    /// The residual handle goes with the record rather than with the node: the builder holds it to
    /// keep a *certified* residual alive, so an uncertified one becomes collectable again.
    #[test]
    fn a_factor_whose_density_was_removed_is_uncertified() {
        let model = ModelClient::<B>::default();
        let pose_q: Q<B, SE3> = model.add_parameter(SE3);
        let point_q: Q<B, EuclideanD> = model.add_parameter(EuclideanD(3));

        let at = point_map()
            .with_quantity(&pose_q, pose_row(0.4, [1.5, -2.25, 0.75]))
            .with_quantity(&point_q, row(&[1.0, 2.0, 3.0]))
            .build();
        let (packed, normalizer) = whitening_from_precision(&coupled_precision()).unwrap();
        let whitening = model.add_constant(Tril(9), &VectorD::new(&packed));

        let factor = JointNormal::new(vec![pose_q.id(), point_q.id()], at, whitening, normalizer);
        let density = add_joint_factor(&model, &factor).unwrap();
        let residual = ModelBuilderOps::factors(&model)[0].residual;

        // Rooted at the density: `ℓ` alone goes, and with the record pruned the residual is
        // unpinned — so the same call's cascade reclaims it too.
        assert_eq!(model.remove_closure(&[density.id()]).unwrap().len(), 1);

        assert!(ModelBuilderOps::factors(&model).is_empty());
        assert_eq!(model.residual_layout().unwrap().total(), 0);
        assert!(
            model.tangent_layout(&[residual]).is_err(),
            "an uncertified residual is reclaimed in the same breath"
        );

        // The handle outlives the node; dropping it stays uneventful.
        drop(density);
        model.cleanup();
    }

    /// Removing a factor's density reclaims the whole factor body — anchors included.
    ///
    /// A log-density is a leaf, so the closure takes `ℓ` alone; everything that computed it —
    /// the whitening product, the squared norm, the concatenation, the per-block tangents, the
    /// frozen anchors — sits *upstream* of it and holds no handle. That is the leak the cascade
    /// in `cleanup()` exists for: the removal offers the removed node's parents, each collection
    /// offers the next layer up, and the body unwinds entirely. `private_nodes` alone never
    /// covered this: it reclaims anchors only when the *concatenation* dies inside a closure,
    /// and here the concatenation is upstream of the root.
    #[test]
    fn retiring_a_density_reclaims_the_whole_factor_body() {
        let model = ModelClient::<B>::default();
        let pose_q: Q<B, SE3> = model.add_parameter(SE3);
        let point_q: Q<B, EuclideanD> = model.add_parameter(EuclideanD(3));

        let at = point_map()
            .with_quantity(&pose_q, pose_row(0.4, [1.5, -2.25, 0.75]))
            .with_quantity(&point_q, row(&[1.0, 2.0, 3.0]))
            .build();
        let (packed, normalizer) = whitening_from_precision(&coupled_precision()).unwrap();
        let whitening = model.add_constant(Tril(9), &VectorD::new(&packed));

        let node_count = || model.server_call(|server| server.graph().node_count());
        let baseline = node_count();

        let factor = JointNormal::new(
            vec![pose_q.id(), point_q.id()],
            at,
            whitening.clone(),
            normalizer,
        );
        let density = factor.log_density(&model).unwrap();
        model.mark_likelihood(&density).unwrap();
        assert!(node_count() > baseline);

        model.remove_closure(&[density.id()]).unwrap();
        drop(density);
        model.cleanup();

        assert_eq!(
            node_count(),
            baseline,
            "ℓ goes in the closure; everything that computed it follows in the cascade"
        );
    }
}
