//! Tests for the two-track (continuous/discrete) operation architecture:
//! discrete `Q` ops, bridges, and a mixed-domain `DiffOperation`.

use skoro_core::error::SkoroResult;
use skoro_model::ModelBuilderOps;
use std::collections::HashMap;

use skoro_backend::{DiscretePoint, NdArrayBackend, Point, TensorData};
use skoro_core::{
    discrete::{Boolean, Categorical, Integer},
    id::Id,
};
use skoro_graph::point_map::PointMap;
use skoro_model::{
    discrete::{
        DiscreteCompare,
        bridge::{DiscreteEmbed, Discretize},
    },
    map::register_map2,
    model::ModelClient,
    q::Q,
};
use skoro_operation::{
    Input,
    diff::{DiffModeState, DiffOperation, DiffOperationForward},
};

type B = NdArrayBackend<f32, i32>;

fn int_input(v: i32) -> Input<B> {
    Input::int(DiscretePoint::<B>::from_data(
        TensorData::new(vec![v], [1, 1]),
        &Default::default(),
    ))
}

fn discretes_for(id: Id, v: i32) -> HashMap<Id, Input<B>> {
    let mut m = HashMap::new();
    m.insert(id, int_input(v));
    m
}

fn int_value(out: &PointMap<B>, id: &Id) -> i32 {
    let (input, _) = out.get(id).expect("discrete value should be present");
    input
        .clone()
        .unwrap_int()
        .into_data()
        .to_vec::<i32>()
        .unwrap()[0]
}

/// The arithmetic, comparison and logic operators over discrete `Q`s. They build ordinary graph
/// nodes, so what is asserted is the integer that comes back out of an evaluation rather than the
/// expression that went in.
mod operators {
    use super::*;

    #[test]
    fn discrete_arithmetic_operators() {
        let model = ModelClient::<B>::default();
        let a = model.add_constant(Integer, &3i32);
        let b = model.add_constant(Integer, &4i32);

        let sum = (a.clone() + b.clone()).unwrap();
        let prod = (a.clone() * b.clone()).unwrap();

        let out = model
            .evaluate_ids(&HashMap::new(), &[sum.id(), prod.id()])
            .unwrap();
        assert_eq!(int_value(&out.evaluations, &sum.id()), 7);
        assert_eq!(int_value(&out.evaluations, &prod.id()), 12);
    }

    #[test]
    fn discrete_comparison_and_logic() {
        let model = ModelClient::<B>::default();
        let a = model.add_constant(Integer, &3i32);
        let b = model.add_constant(Integer, &5i32);

        let lt = a.lt(&b).unwrap(); // 3 < 5 -> true (1)
        let ge = a.ge(&b).unwrap(); // 3 >= 5 -> false (0)
        let both = (lt.clone() & ge.clone()).unwrap(); // true AND false -> false
        let either = (lt.clone() | ge.clone()).unwrap(); // true OR false -> true
        let neither = (!either.clone()).unwrap(); // NOT true -> false

        let out = model
            .evaluate_ids(
                &HashMap::new(),
                &[lt.id(), ge.id(), both.id(), either.id(), neither.id()],
            )
            .unwrap();
        let g = |q: &Q<B, Boolean>| int_value(&out.evaluations, &q.id());
        assert_eq!(g(&lt), 1);
        assert_eq!(g(&ge), 0);
        assert_eq!(g(&both), 0);
        assert_eq!(g(&either), 1);
        assert_eq!(g(&neither), 0);
    }
}

/// The two crossings between the tracks: an embedding from discrete to continuous, and a
/// discretization back. These are the only places a value changes kind, which is why they are
/// tested apart from the operators that stay on one side.
mod bridges {
    use super::*;

    #[test]
    fn embedding_bridge_disc_to_cont() {
        let model = ModelClient::<B>::default();
        let n = model.add_constant(Integer, &6i32);
        let x: Q<B, skoro_riemann::real::Real> = n.to_real().unwrap();

        let out = model.evaluate(&HashMap::new(), &x).unwrap();
        assert_eq!(out.into_data().to_vec::<f32>().unwrap(), vec![6.0]);
    }

    #[test]
    fn discretization_bridge_cont_to_disc() {
        let model = ModelClient::<B>::default();
        let x = model.add_constant(skoro_riemann::real::Real, &2.5f64);
        let flag = x.is_non_negative().unwrap(); // 2.5 >= 0 -> true

        let out = model.evaluate_ids(&HashMap::new(), &[flag.id()]).unwrap();
        assert_eq!(int_value(&out.evaluations, &flag.id()), 1);
    }
}

/// A `DiffOperation` with a continuous output reading one continuous and one discrete input. The
/// gradient must flow to the continuous input *only* -- a pass that handed the discrete slot a
/// cotangent would be writing into a tangent space that does not exist.
mod mixed_kernels {
    use super::*;

    /// `Scale(x_cont, n_disc) = x * n` — continuous output, one continuous and one
    /// discrete input. The gradient flows to the continuous input only.
    #[derive(Debug, Clone, Copy)]
    struct Scale;

    impl DiffOperation<B> for Scale {
        type VjpState = Point<B>; // store the discrete factor (as float) for the pullback
        type HvpState = ();

        fn forward(
            &self,
            inputs: Vec<Input<B>>,
            diff_state: DiffModeState<B>,
        ) -> DiffOperationForward<B, Self::VjpState, Self::HvpState> {
            let [x, n] = inputs.try_into().unwrap();
            let x: Point<B> = x.unwrap_float();
            let factor: Point<B> = n.unwrap_int().float();
            let out = x.clone() * factor.clone();
            match diff_state {
                DiffModeState::ForwardOnly => DiffOperationForward::forward_only(out),
                DiffModeState::Jvp(tvecs) => {
                    let [dx, _dn] = tvecs.try_into().unwrap();
                    let tangent = dx.map(|dx| dx * factor.clone().unsqueeze_dim(2));
                    DiffOperationForward::jvp(out, tangent)
                }
                DiffModeState::Vjp(_) => DiffOperationForward::vjp(out, factor),
                DiffModeState::Hvp { tangents, .. } => {
                    let [dx, _dn] = tangents.try_into().unwrap();
                    let tangent = dx.map(|dx| dx * factor.clone());
                    DiffOperationForward::hvp(out, tangent, factor, ())
                }
            }
        }

        fn vjp(
            &self,
            factor: &Self::VjpState,
            omega: skoro_backend::CotangentFrame<B>,
        ) -> Vec<Option<skoro_backend::CotangentFrame<B>>> {
            // d/dx (x*n) = n  ->  cotangent for x; discrete input gets None.
            vec![Some(omega * factor.clone().unsqueeze_dim(2)), None]
        }

        fn hvp(
            &self,
            _: &Self::VjpState,
            _: &Self::HvpState,
            _: skoro_backend::CotangentFrame<B>,
        ) -> Vec<Option<skoro_backend::CotangentFrame<B>>> {
            vec![None, None]
        }
    }

    #[test]
    fn mixed_kernel_gradient_flows_to_continuous_input_only() {
        use skoro_riemann::real::Real;

        let model = ModelClient::<B>::default();
        let x: Q<B, Real> = model.add_parameter(Real);
        let n = model.add_parameter(Integer);

        // y = x * n  (continuous output, mixed inputs)
        let y = register_map2(Real, Scale, (&x, &n)).unwrap();

        let mut points = HashMap::new();
        points.insert(
            x.id(),
            Input::float(Point::<B>::from_data(
                TensorData::new(vec![4.0f32], [1, 1]),
                &Default::default(),
            )),
        );
        points.extend(discretes_for(n.id(), 3));

        // forward: 4 * 3 = 12
        let out = model.evaluate_ids(&points, &[y.id()]).unwrap();
        let (yv, _) = out.evaluations.get(&y.id()).unwrap();
        let yv = yv
            .clone()
            .unwrap_float()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(yv, vec![12.0]);

        // gradient flows to the continuous input only: dy/dx = n = 3
        let grad = model.evaluate_ord1(&points, &y).unwrap();
        let gx = grad.cotangents[&x.id()]
            .clone()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(gx, vec![3.0]);
        // the discrete input receives no cotangent
        assert!(!grad.cotangents.contains_key(&n.id()));
    }
}

/// Discrete quantities as model citizens: a categorical constant surfaces its value, and a discrete
/// distribution samples, observes and scores through the same `Distribution` interface a continuous
/// one uses.
mod quantities {
    use super::*;

    #[test]
    fn categorical_constant_value() {
        let model = ModelClient::<B>::default();
        let z = model.add_constant(Categorical(4), &2usize);
        let out = model.evaluate_ids(&HashMap::new(), &[z.id()]).unwrap();
        assert_eq!(int_value(&out.constants, &z.id()), 2);
    }

    use skoro_core::kind::Discrete;
    use skoro_core::sampler::Sampler;
    use skoro_model::distribution::{Distribution, Factor};

    #[derive(Debug, Clone)]
    struct PointMass;

    impl Sampler<B, Discrete> for PointMass {
        fn sample(&self, batch_size: usize, _: Vec<Input<B>>) -> DiscretePoint<B> {
            DiscretePoint::<B>::from_data(TensorData::new(vec![1i32], [1, 1]), &Default::default())
                .repeat_dim(0, batch_size)
        }
    }

    impl Distribution<B, Discrete> for PointMass {
        type Qty = Boolean;
        type Sampler = Self;

        fn quantity_type(&self) -> Boolean {
            Boolean
        }
        fn sampler(&self) -> Self::Sampler {
            self.clone()
        }
        fn parent_ids(&self) -> &[Id] {
            &[]
        }
        fn factor(&self, x: &Q<B, Boolean>) -> SkoroResult<Factor<B>> {
            Ok(Factor::new((x.to_real()? * 0.0f32)?))
        }
    }

    #[test]
    fn discrete_distribution_samples_observes_and_scores() {
        let model = ModelClient::<B>::default();

        // The same `add_quantity` call serves continuous and discrete distributions.
        let z = model.add_quantity(PointMass).unwrap();

        // Sampling flows through the kinded sampler into the integer arm.
        let out = model.generate(2).unwrap();
        let (v, _) = out.variables.get(&z.id()).unwrap();
        let data = v.clone().unwrap_int().into_data().to_vec::<i32>().unwrap();
        assert_eq!(data, vec![1, 1]);

        // Observation works through the unified `observe` (packed into int storage).
        z.observe(&false).unwrap();

        // The log-pmf evaluates through the ordinary log-density path.
        let ld = model.log_density(&HashMap::new()).unwrap();
        assert_eq!(ld.into_data().to_vec::<f32>().unwrap(), vec![0.0]);
    }
}
