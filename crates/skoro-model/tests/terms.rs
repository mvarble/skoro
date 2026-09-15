//! The `log p = log q + log w` split. A term is a summand of the proposal `q` when it scores a
//! quantity still being generated, and a summand of the log-density `p` unless it was introduced
//! by `add_proposal`. The weight is the difference, so `q + w == p` by construction.

use skoro_backend::{NdArrayBackend, Point, Tensor, TensorData};
use skoro_core::{error::SkoroResult, id::Id, kind::Continuous};
use skoro_model::{
    ModelBuilderOps, ModelClientOps,
    distribution::{Distribution, Factor, Sampler},
    hash_map_builders::points,
    map::register_map1,
    model::ModelClient,
    q::Q,
};
use skoro_operation::{Input, ops::LiteralScalarMultiplication};
use skoro_riemann::real::Real;

type B = NdArrayBackend<f64, i32>;

fn row(value: f64) -> Tensor<B, 2> {
    Tensor::from_data(TensorData::new(vec![value], [1, 1]), &Default::default())
}

fn scalar(t: Tensor<B, 2>) -> f64 {
    t.into_data().to_vec::<f64>().unwrap()[0]
}

#[derive(Debug, Clone)]
struct NeverSampler;

impl Sampler<B, Continuous> for NeverSampler {
    fn sample(&self, _: usize, _: Vec<Input<B>>) -> Point<B> {
        unimplemented!("these tests never generate")
    }
}

/// `log p(x) = 2x`, over no parents.
struct Linear;

impl Distribution<B, Continuous> for Linear {
    type Qty = Real;
    type Sampler = NeverSampler;

    fn quantity_type(&self) -> Real {
        Real
    }
    fn sampler(&self) -> NeverSampler {
        NeverSampler
    }
    fn parent_ids(&self) -> &[Id] {
        &[]
    }
    fn factor(&self, x: &Q<B, Real>) -> SkoroResult<Factor<B>> {
        Ok(Factor::new(register_map1(
            Real,
            LiteralScalarMultiplication::new(2.0),
            x,
        )?))
    }
}

#[test]
fn proposal_plus_weight_is_the_log_density() {
    let model = ModelClient::<B>::default();
    let x = model.add_quantity(Linear).unwrap();
    let y = model.add_quantity(Linear).unwrap();
    y.observe(&1.5f64).unwrap();
    let a = model.add_parameter(Real);
    let term = register_map1(Real, LiteralScalarMultiplication::new(-3.0), &a).unwrap();
    model.add_factor(Factor::new(term), &[a.id()]).unwrap();

    let values = points()
        .with_quantity(&x, row(0.25))
        .with_quantity(&a, row(2.0))
        .build();
    let whole = scalar(model.log_density(&values).unwrap());
    let proposal = scalar(model.log_proposal(&values).unwrap());
    let weight = scalar(model.log_weight(&values).unwrap());

    // x's term is the only proposal (2 · 0.25); y's (2 · 1.5) and the factor's (−3 · 2) weigh.
    assert!((proposal - 0.5).abs() < 1e-12);
    assert!((weight - (3.0 - 6.0)).abs() < 1e-12);
    assert!((proposal + weight - whole).abs() < 1e-12);
}

#[test]
fn observing_moves_a_term_out_of_the_proposal() {
    let model = ModelClient::<B>::default();
    let x = model.add_quantity(Linear).unwrap();
    let density = model.log_density_of(&x).unwrap();
    let flags_of = |model: &ModelClient<B>| {
        let t = model
            .terms()
            .into_iter()
            .find(|t| t.density == density)
            .unwrap();
        (t.in_density, t.in_proposal)
    };

    assert_eq!(flags_of(&model), (true, true));
    x.observe(&0.5f64).unwrap();
    assert_eq!(flags_of(&model), (true, false));
    x.unobserve().unwrap();
    assert_eq!(flags_of(&model), (true, true));
}
