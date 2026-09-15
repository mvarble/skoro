//! What a Gaussian factor computes, and what it binds.
//!
//! Two things are at stake. First, that `whitened_residual` is really the `r` in
//! `−log p = ½‖r‖² + n(θ)` — if it is not, a least-squares optimizer minimizes something
//! unrelated to the model, silently. Second, that the factor binds honestly: the nodes under its
//! residual, location, whitening and normalizer-parent roles are the ones its density was built
//! from, and the density they hang off is the term the model sums.
//!
//! `SE3` is exercised deliberately. Its manifold dimension is 6 and its storage is 12, so a
//! residual sized by storage rather than by dimension is visible here and invisible on
//! `Real` or `EuclideanD`.

use skoro_backend::{NdArrayBackend, Point, TensorData};
use skoro_core::id::Id;
use skoro_linalg::tril::{Tril, TrilOps};
use skoro_linalg::vector::LinalgVectorOps;
use skoro_model::{
    ModelBuilderOps, ModelClientOps,
    hash_map_builders::points,
    least_squares::residual_layout,
    model::ModelClient,
    q::Q,
    roles::{ARGUMENT, LOCATION, NORMALIZER_PARENT, RESIDUAL, WHITENING},
};
use skoro_normal::{
    distribution::Normal, facade::constant_covariance_diagonal, structure::GaussianBlock,
};
use skoro_operation::{Input, ValueMap};
use skoro_pointwise::PointwiseOps;
use skoro_riemann::{
    euclidean::{EuclideanD, VectorD},
    real::Real,
};
use skoro_se3::quantity::{SE3, SE3Pose};
use skoro_so3::quantity::SO3Matrix;

type B = NdArrayBackend<f64, i32>;

fn point(values: &[f64]) -> Input<B> {
    Input::float(Point::<B>::from_data(
        TensorData::new(values.to_vec(), [1, values.len()]),
        &Default::default(),
    ))
}

fn scalar(t: skoro_backend::Tensor<B, 2>) -> f64 {
    t.into_data().to_vec::<f64>().unwrap()[0]
}

fn vector(t: skoro_backend::Tensor<B, 2>) -> Vec<f64> {
    t.into_data().to_vec::<f64>().unwrap()
}

/// That `whitened_residual` really is the `r` in `−log p = ½‖r‖² + n(θ)`, over `Real`, `EuclideanD`
/// and `SE3`. The sum of squares is taken **on the host** from the evaluated residual, so the only
/// thing shared between the two sides is the residual node itself. `SE3` is deliberate: dimension
/// 6, storage 12, so a residual sized by storage is visible here and invisible on the flat manifolds.
mod densities {
    use super::*;

    /// `log p(x) = constant − ½‖r(x)‖² + log|L⁻¹|`, with `‖r‖²` summed **on the host** from the
    /// evaluated residual rather than through the graph's `norm_squared`.
    ///
    /// Taking the sum of squares outside the graph is what stops this from being a restatement
    /// of `log_density`'s own body: the only thing shared between the two sides is the residual
    /// node itself, which is exactly the thing under test.
    fn assert_density_is_the_sum_of_squares<M>(
        client: &ModelClient<B>,
        distr: &Normal<B, M>,
        x: &Q<B, M>,
        points: ValueMap<B>,
    ) where
        M: skoro_graph::quantity::ContinuousQuantity<B>,
    {
        let residual = distr.whitened_residual(x).unwrap();
        let log_det = distr
            .std_dev()
            .invert()
            .unwrap()
            .diag()
            .unwrap()
            .abs()
            .unwrap()
            .log()
            .unwrap()
            .sum()
            .unwrap();

        let r = vector(client.evaluate(&points, &residual).unwrap());
        let sum_of_squares: f64 = r.iter().map(|v| v * v).sum();
        let log_det = scalar(client.evaluate(&points, &log_det).unwrap());
        let density = scalar(client.log_density(&points).unwrap());

        approx::assert_abs_diff_eq!(
            density,
            distr.constant() - 0.5 * sum_of_squares + log_det,
            epsilon = 1e-9
        );
    }

    #[test]
    fn a_real_density_is_its_squared_residual() {
        let client = ModelClient::<B>::default();
        let mean = client.add_constant(Real, &-1.75f64);
        let std_dev = constant_covariance_diagonal(&client, &[2.5]);
        let distr = Normal::new(mean, std_dev);
        let x = client.add_quantity(distr.clone()).unwrap();

        let values = points().with_quantity(&x, point(&[3.25])).build();
        assert_density_is_the_sum_of_squares(&client, &distr, &x, values);
    }

    /// A non-diagonal covariance, so the whitening actually mixes components: a `Tril` treated
    /// as diagonal would pass a diagonal fixture.
    #[test]
    fn a_euclidean_density_is_its_squared_residual() {
        let client = ModelClient::<B>::default();
        let mean = client.add_constant(EuclideanD(3), &VectorD::new(&[10.0, -20.0, 30.0]));
        let std_dev = client.add_constant(
            Tril(3),
            // Packed row-major lower triangle, far from diagonal.
            &VectorD::new(&[2.0, -1.5, 3.0, 0.75, 2.5, 1.25]),
        );
        let distr = Normal::new(mean, std_dev);
        let x = client.add_quantity(distr.clone()).unwrap();

        let values = points()
            .with_quantity(&x, point(&[11.5, -18.25, 33.0]))
            .build();
        assert_density_is_the_sum_of_squares(&client, &distr, &x, values.clone());

        // And the residual really is `L⁻¹ (x − μ)`: check `L r = x − μ` by hand, which needs
        // no inverse and no library.
        let residual = distr.whitened_residual(&x).unwrap();
        let r = vector(client.evaluate(&values, &residual).unwrap());
        assert_eq!(r.len(), 3, "the residual is sized by manifold dimension");

        let packed = [2.0, -1.5, 3.0, 0.75, 2.5, 1.25];
        let expected = [11.5 - 10.0, -18.25 + 20.0, 33.0 - 30.0];
        for i in 0..3 {
            let lr: f64 = (0..=i).map(|j| packed[i * (i + 1) / 2 + j] * r[j]).sum();
            approx::assert_abs_diff_eq!(lr, expected[i], epsilon = 1e-9);
        }
    }

    /// `SE3` has dimension 6 and storage 12. The residual must be 6 wide.
    #[test]
    fn an_se3_density_is_its_squared_residual() {
        let client = ModelClient::<B>::default();
        // A rotation about z by 0.4 rad, translated. Far enough from the identity that a
        // missing Jacobian in the inverse retraction would show.
        let (c, s) = (0.4f64.cos(), 0.4f64.sin());
        let mean = client.add_constant(
            SE3,
            &SE3Pose::new(
                SO3Matrix([c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0]),
                [1.5, -2.25, 0.75],
            ),
        );
        let std_dev = constant_covariance_diagonal(&client, &[1.0, 2.0, 0.5, 3.0, 1.5, 0.25]);
        let distr = Normal::new(mean, std_dev);
        let x = client.add_quantity(distr.clone()).unwrap();

        // A different pose: rotation about z by 0.9 rad.
        let (c2, s2) = (0.9f64.cos(), 0.9f64.sin());
        let values = points()
            .with_quantity(
                &x,
                point(&[c2, -s2, 0.0, s2, c2, 0.0, 0.0, 0.0, 1.0, 2.0, -1.0, 1.25]),
            )
            .build();

        let residual = distr.whitened_residual(&x).unwrap();
        assert_eq!(
            vector(client.evaluate(&values, &residual).unwrap()).len(),
            6,
            "an SE3 residual is 6 wide — a dimension, not the 12-wide storage"
        );
        assert_density_is_the_sum_of_squares(&client, &distr, &x, values);
    }
}

/// That the factor binds what it built: the nodes under `RESIDUAL`, `LOCATION`, `WHITENING` and
/// `NORMALIZER_PARENT` are the ones its density was computed from, the density they hang off is the
/// term the model's own log-density sums, and the normalizer depends on the covariance factor and
/// nothing else. That last one is what makes the least-squares precondition checkable without
/// inspecting an operation.
mod bindings {
    use super::*;

    /// The evaluated components of node `id`.
    fn values_of(model: &ModelClient<B>, values: &ValueMap<B>, id: Id) -> Vec<f64> {
        let evaluated = model.evaluate_ids(values, &[id]).unwrap();
        let (value, _) = evaluated.evaluations.get(&id).expect("an evaluated node");
        vector(value.unwrap_float_ref().clone())
    }

    /// `Normal`'s normalizer depends on the covariance factor and on nothing else, and the
    /// factor binds exactly that.
    ///
    /// Minimizing `½Σ‖rᵢ‖²` is minimizing `−log p` exactly when nothing bound as a normalizer
    /// parent is being optimized. Binding the mean there would make every Gaussian model look
    /// ineligible.
    #[test]
    fn only_the_covariance_reaches_the_normalizer() {
        let model = ModelClient::<B>::default();
        let mean = model.add_parameter(EuclideanD(2));
        let std_dev = constant_covariance_diagonal(&model, &[1.0, 2.0]);
        let x = model
            .add_quantity(Normal::new(mean.clone(), std_dev.clone()))
            .unwrap();
        let density = model.log_density_of(&x).unwrap();

        assert_eq!(
            model.bound(density, NORMALIZER_PARENT),
            vec![std_dev.id()],
            "the mean must not be bound: it does not appear in the normalizer"
        );
        // The refinement is the other half: `μ` and `L` reachable apart, which is what a filter
        // needs since `H = ∂μ/∂x` and `R = L Lᵀ` are different objects.
        assert_eq!(model.bound(density, LOCATION), vec![mean.id()]);
        assert_eq!(model.bound(density, WHITENING), vec![std_dev.id()]);
        assert_eq!(
            model.bound(density, ARGUMENT),
            vec![x.id(), mean.id(), std_dev.id()]
        );
    }

    /// Every Gaussian added binds the nodes it built, and they are the ones the graph holds.
    #[test]
    fn the_gaussian_builder_binds_what_it_built() {
        let model = ModelClient::<B>::default();
        let mean_a = model.add_constant(EuclideanD(2), &VectorD::new(&[0.0, 0.0]));
        let cov_a = constant_covariance_diagonal(&model, &[1.0, 2.0]);
        let distr_a = Normal::new(mean_a.clone(), cov_a.clone());
        let a = model.add_quantity(distr_a.clone()).unwrap();

        let mean_b = model.add_constant(Real, &1.0f64);
        let cov_b = constant_covariance_diagonal(&model, &[0.5]);
        let distr_b = Normal::new(mean_b.clone(), cov_b.clone());
        let b = model.add_quantity(distr_b.clone()).unwrap();

        let blocks = GaussianBlock::all(&model);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].mean, mean_a.id());
        assert_eq!(blocks[0].whitening, cov_a.id());
        assert_eq!(blocks[1].mean, mean_b.id());
        assert_eq!(blocks[1].whitening, cov_b.id());

        // The terms agree with the blocks, factor for factor and in order.
        let factors = model.factors();
        assert_eq!(factors.len(), 2);
        assert_eq!(factors[0].density, model.log_density_of(&a).unwrap());
        assert_eq!(factors[1].density, model.log_density_of(&b).unwrap());
        for (factor, block) in factors.iter().zip(&blocks) {
            assert_eq!(factor.density, block.density);
            assert_eq!(model.bound(factor.density, RESIDUAL), vec![block.residual]);
            assert_eq!(
                model.bound(factor.density, NORMALIZER_PARENT),
                vec![block.whitening]
            );
        }

        let values = points()
            .with_quantity(&a, point(&[0.5, -1.25]))
            .with_quantity(&b, point(&[2.25]))
            .build();

        // The bound residual is the `r` the distribution describes: it evaluates to what a
        // freshly built `whitened_residual` does.
        let fresh_a = distr_a.whitened_residual(&a).unwrap();
        let fresh_b = distr_b.whitened_residual(&b).unwrap();
        for (block, fresh) in [
            (
                blocks[0],
                vector(model.evaluate(&values, &fresh_a).unwrap()),
            ),
            (
                blocks[1],
                vector(model.evaluate(&values, &fresh_b).unwrap()),
            ),
        ] {
            let bound = values_of(&model, &values, block.residual);
            assert_eq!(bound.len(), fresh.len());
            for (got, want) in bound.iter().zip(&fresh) {
                approx::assert_abs_diff_eq!(got, want, epsilon = 1e-12);
            }
        }

        // Each bound density really names a log-density node: evaluating the two by id and
        // summing must reproduce the model's total log-density. An id that named something else
        // — the outermost `+` of the sugar, say — would not add up.
        let summed: f64 = factors
            .iter()
            .map(|f| values_of(&model, &values, f.density)[0])
            .sum();
        approx::assert_abs_diff_eq!(
            summed,
            scalar(model.log_density(&values).unwrap()),
            epsilon = 1e-9
        );

        // The row layout of `J` is the residuals, sized by dimension: 2 then 1.
        let layout = residual_layout(&model).unwrap();
        assert_eq!(layout.total(), 3);
        assert_eq!(
            layout
                .blocks()
                .iter()
                .map(|b| b.dimension)
                .collect::<Vec<_>>(),
            vec![2, 1]
        );
    }

    /// The bound density really is the term the objective sums.
    #[test]
    fn the_bound_density_is_the_models_own() {
        let model = ModelClient::<B>::default();
        let mean = model.add_constant(EuclideanD(2), &VectorD::new(&[3.0, -1.0]));
        let cov = constant_covariance_diagonal(&model, &[1.0, 2.0]);
        let x = model.add_quantity(Normal::new(mean, cov.clone())).unwrap();

        let factors = model.factors();
        assert_eq!(factors.len(), 1);
        assert_eq!(
            model.bound(factors[0].density, NORMALIZER_PARENT),
            vec![cov.id()]
        );

        // The bound density is the model's whole log-density, this being its only factor.
        let values = points().with_quantity(&x, point(&[3.5, -0.25])).build();
        approx::assert_abs_diff_eq!(
            values_of(&model, &values, factors[0].density)[0],
            scalar(model.log_density(&values).unwrap()),
            epsilon = 1e-9
        );

        assert_eq!(residual_layout(&model).unwrap().total(), 2);
    }

    /// A model nobody told anything has no factors and no rows.
    #[test]
    fn an_untouched_model_binds_nothing() {
        let model = ModelClient::<B>::default();
        assert!(model.factors().is_empty());
        assert!(GaussianBlock::all(&model).is_empty());
        assert_eq!(residual_layout(&model).unwrap().total(), 0);
    }
}

/// The bound nodes survive garbage collection -- a residual is a childless `Mapped` node whose
/// handle nobody keeps, so without the binding the collector would take it. Retiring the middle
/// factor of three keeps the outer two *in order*, since `residual_layout` is the row order of `J`
/// and the block list is read positionally beside it.
mod retirement {
    use super::*;

    /// The bound residuals survive garbage collection, and so does the subgraph beneath them.
    #[test]
    fn cleanup_does_not_collect_the_bound_residuals() {
        let model = ModelClient::<B>::default();
        let mean_a = model.add_constant(EuclideanD(2), &VectorD::new(&[0.0, 0.0]));
        let cov_a = constant_covariance_diagonal(&model, &[1.0, 2.0]);
        let a = model.add_quantity(Normal::new(mean_a, cov_a)).unwrap();

        let mean_b = model.add_constant(Real, &1.0f64);
        let cov_b = constant_covariance_diagonal(&model, &[0.5]);
        let b = model.add_quantity(Normal::new(mean_b, cov_b)).unwrap();

        let before = residual_layout(&model).unwrap();
        model.cleanup();
        let after = residual_layout(&model).unwrap();
        assert_eq!(before, after, "cleanup must not collect a bound residual");

        let values = points()
            .with_quantity(&a, point(&[0.5, -1.25]))
            .with_quantity(&b, point(&[2.25]))
            .build();
        let ids: Vec<Id> = after.ids().collect();
        let evaluated = model.evaluate_ids(&values, &ids).unwrap();
        for id in ids {
            assert!(evaluated.evaluations.get(&id).is_some());
        }
    }

    /// Retiring the middle factor of three leaves the outer two, in their original order.
    ///
    /// Order is load-bearing twice over: `residual_layout` is *the* row order of `J`, and the
    /// `GaussianBlock` list is read positionally beside it. A retirement that reordered the
    /// survivors would hand an optimizer one factor's rows labelled with another's, which no
    /// shape check downstream would reject.
    #[test]
    fn retiring_a_factor_drops_its_term_and_keeps_the_order() {
        let model = ModelClient::<B>::default();

        let add = |dims: &[f64]| {
            let mean = model.add_constant(
                EuclideanD(dims.len()),
                &VectorD::new(&vec![0.0; dims.len()]),
            );
            let cov = constant_covariance_diagonal(&model, dims);
            model.add_quantity(Normal::new(mean, cov)).unwrap()
        };
        let a = add(&[1.0, 2.0]);
        let b = add(&[1.0, 2.0, 3.0]);
        let c = add(&[0.5]);

        let before = model.factors();
        assert_eq!(before.len(), 3);
        assert_eq!(residual_layout(&model).unwrap().total(), 6);
        let residual_of = |density: Id| model.bound(density, RESIDUAL)[0];
        let survivors = [
            residual_of(before[0].density),
            residual_of(before[2].density),
        ];

        model.remove_closure(&[b.id()]).unwrap();
        model.cleanup();

        let after = model.factors();
        assert_eq!(
            after,
            [before[0].clone(), before[2].clone()],
            "the survivors keep their identity and their order"
        );
        assert_eq!(
            GaussianBlock::all(&model)
                .iter()
                .map(|block| block.residual)
                .collect::<Vec<_>>(),
            survivors,
            "the block list follows the terms by derivation"
        );

        // The rows of `J` are now `a` then `c`: 2 then 1, and never 2 then 3.
        let layout = residual_layout(&model).unwrap();
        assert_eq!(layout.ids().collect::<Vec<_>>(), survivors);
        assert_eq!(
            layout
                .blocks()
                .iter()
                .map(|block| block.dimension)
                .collect::<Vec<_>>(),
            vec![2, 1]
        );

        // And the model still evaluates over what is left.
        let values = points()
            .with_quantity(&a, point(&[0.5, -1.25]))
            .with_quantity(&c, point(&[2.25]))
            .build();
        let evaluated = model.evaluate_ids(&values, &survivors).unwrap();
        for id in survivors {
            assert!(evaluated.evaluations.get(&id).is_some());
        }
    }
}
