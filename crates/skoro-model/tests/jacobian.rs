//! What the two Jacobian paths must agree on.
//!
//! [`jacobian_jvp`] concatenates output blocks and is done; [`jacobian_vjp`] concatenates
//! *input* blocks and must then transpose. That asymmetry is invisible on a square
//! Jacobian, so the model here is deliberately non-square — `D_in = 8`, `D_out = 4` — and
//! deliberately not one-dimensional per block: `Euclidean<2>` blocks sit in the middle of
//! both layouts, so `total()` differs from the block *count* and a slot index differs from
//! a block index.

use skoro_backend::{NdArrayBackend, Tensor};
use skoro_core::{error::SkoroError, id::Id};
use skoro_model::{
    TangentLayout,
    hash_map_builders::points,
    jacobian::{jacobian_jvp, jacobian_vjp},
    model::ModelClient,
    q::Q,
};
use skoro_operation::ValueMap;
use skoro_riemann::{euclidean::Euclidean, real::Real};

type B = NdArrayBackend<f64, i32>;

const BATCH: usize = 2;
const D_IN: usize = 8;
const D_OUT: usize = 4;

fn tensor<const N: usize>(values: [[f64; N]; BATCH]) -> Tensor<B, 2> {
    let flat: Vec<f64> = values.into_iter().flatten().collect();
    Tensor::<B, 1>::from_floats(flat.as_slice(), &Default::default()).reshape([BATCH, N])
}

fn layout(model: &ModelClient<B>, ids: &[Id]) -> TangentLayout {
    model
        .server_call(|server| TangentLayout::from_ids(server.graph(), ids))
        .expect("every id is a continuous quantity of this model")
}

/// Scalars `x, y, z, w` and plane vectors `p, q`, mapped to `u = x·y`, `v = y·z`,
/// `m = p − x·q`.
///
/// `w` is free but reaches no row. Column order is `[x, p, y, q, z, w]` and row order is
/// `[u, m, v]`, so in both layouts a two-dimensional block sits in the middle and the
/// blocks after it start at an offset no block index would give. The subtraction is not
/// symmetric in its operands, so a swapped column pair shows up as a sign.
struct Model {
    client: ModelClient<B>,
    x: Q<B, Real>,
    y: Q<B, Real>,
    z: Q<B, Real>,
    w: Q<B, Real>,
    p: Q<B, Euclidean<2>>,
    q: Q<B, Euclidean<2>>,
    u: Q<B, Real>,
    v: Q<B, Real>,
    m: Q<B, Euclidean<2>>,
}

impl Model {
    fn new() -> Self {
        let client = ModelClient::<B>::default();
        let x = client.add_parameter(Real);
        let y = client.add_parameter(Real);
        let z = client.add_parameter(Real);
        let w = client.add_parameter(Real);
        let p = client.add_parameter(Euclidean::<2>);
        let q = client.add_parameter(Euclidean::<2>);
        let u = (x.clone() * y.clone()).unwrap();
        let v = (y.clone() * z.clone()).unwrap();
        let m = (p.clone() - (x.clone() * q.clone()).unwrap()).unwrap();
        Self {
            client,
            x,
            y,
            z,
            w,
            p,
            q,
            u,
            v,
            m,
        }
    }

    /// `x = (2, 7)`, `y = (3, 11)`, `z = (5, 13)`. Every value is exactly representable,
    /// so the Jacobian may be compared exactly.
    fn points(&self) -> ValueMap<B> {
        points()
            .with_quantity(&self.x, tensor([[2.0], [7.0]]))
            .with_quantity(&self.y, tensor([[3.0], [11.0]]))
            .with_quantity(&self.z, tensor([[5.0], [13.0]]))
            .with_quantity(&self.w, tensor([[1.0], [1.0]]))
            .with_quantity(&self.p, tensor([[1.5, -0.5], [0.25, 4.0]]))
            .with_quantity(&self.q, tensor([[0.5, 2.0], [-3.0, 0.75]]))
            .build()
    }

    fn inputs(&self) -> TangentLayout {
        layout(
            &self.client,
            &[
                self.x.id(),
                self.p.id(),
                self.y.id(),
                self.q.id(),
                self.z.id(),
                self.w.id(),
            ],
        )
    }

    fn outputs(&self) -> TangentLayout {
        layout(&self.client, &[self.u.id(), self.m.id(), self.v.id()])
    }
}

/// `J` at the points above, batch-major.
///
/// Columns are `[x, p₀, p₁, y, q₀, q₁, z, w]` and rows are `[u, m₀, m₁, v]`, so
/// `∂u/∂x = y`, `∂u/∂y = x`, `∂mᵢ/∂x = -qᵢ`, `∂mᵢ/∂pᵢ = 1`, `∂mᵢ/∂qᵢ = −x`, `∂v/∂y = z`, `∂v/∂z = y`.
#[rustfmt::skip]
const EXPECTED: [f64; BATCH * D_OUT * D_IN] = [
    // batch 0: x = 2, y = 3, z = 5
      3.0, 0.0, 0.0, 2.0,  0.0,  0.0, 0.0, 0.0,
     -0.5, 1.0, 0.0, 0.0, -2.0,  0.0, 0.0, 0.0,
     -2.0, 0.0, 1.0, 0.0,  0.0, -2.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 5.0,  0.0,  0.0, 3.0, 0.0,
    // batch 1: x = 7, y = 11, z = 13
     11.0, 0.0, 0.0, 7.0,  0.0,  0.0,  0.0, 0.0,
      3.0, 1.0, 0.0, 0.0, -7.0,  0.0,  0.0, 0.0,
    -0.75, 0.0, 1.0, 0.0,  0.0, -7.0,  0.0, 0.0,
      0.0, 0.0, 0.0, 13.0, 0.0,  0.0, 11.0, 0.0,
];

fn into_vec(t: Tensor<B, 3>) -> Vec<f64> {
    t.into_data().to_vec().unwrap()
}

/// Both paths must produce the same `[batch, D_out, D_in]` tensor, and it must be the
/// analytic one. A missing `swap_dims` in the VJP path fails the shape assertion here; a
/// frame count taken from a block count rather than `total()` fails the pass itself.
#[test]
fn jvp_and_vjp_agree_with_the_closed_form() {
    let model = Model::new();
    let points = model.points();
    let (inputs, outputs) = (model.inputs(), model.outputs());

    assert_eq!((inputs.total(), outputs.total()), (D_IN, D_OUT));
    assert_eq!(
        (inputs.blocks().len(), outputs.blocks().len()),
        (6, 3),
        "the layouts must be wider than they are long, or `total()` proves nothing"
    );

    let jvp = jacobian_jvp(&model.client, &points, &inputs, &outputs).unwrap();
    let vjp = jacobian_vjp(&model.client, &points, &inputs, &outputs).unwrap();

    assert_eq!(
        jvp.dims(),
        [BATCH, D_OUT, D_IN],
        "rows are `Out`, columns are `In`"
    );
    assert_eq!(
        vjp.dims(),
        [BATCH, D_OUT, D_IN],
        "the VJP path must transpose back"
    );
    assert_eq!(into_vec(jvp), EXPECTED.to_vec());
    assert_eq!(into_vec(vjp), EXPECTED.to_vec());
}

/// A multi-dimensional block lands at its layout *offset*, not at its block index.
///
/// `p` is the second block of `inputs` but occupies columns 1 and 2, and `m` is the second
/// block of `outputs` but occupies rows 1 and 2. An assembly that indexed by block would
/// put `∂m₁/∂q₁ = −x` at `J[0][1][3]` instead of `J[0][2][5]`.
#[test]
fn multi_dimensional_blocks_land_at_their_offsets() {
    let model = Model::new();
    let j = jacobian_vjp(
        &model.client,
        &model.points(),
        &model.inputs(),
        &model.outputs(),
    )
    .unwrap();
    let j = into_vec(j);
    let at = |row: usize, col: usize| j[row * D_IN + col];

    assert_eq!(at(0, 3), 2.0, "J[0][0][3] = ∂u/∂y = x");
    assert_eq!(at(1, 1), 1.0, "J[0][1][1] = ∂m₀/∂p₀ = 1");
    assert_eq!(at(2, 2), 1.0, "J[0][2][2] = ∂m₁/∂p₁ = 1");
    assert_eq!(at(1, 2), 0.0, "J[0][1][2] = ∂m₀/∂p₁ = 0");
    assert_eq!(at(2, 5), -2.0, "J[0][2][5] = ∂m₁/∂q₁ = −x");
    assert_eq!(at(3, 6), 3.0, "J[0][3][6] = ∂v/∂z = y");
}

/// A free leaf that reaches no row is a zero column, not a panic and not an error.
///
/// `w` is a `Parameter` of the model that no row depends on, so it is not even in the
/// compiled subgraph. This is the regression for the `vjp_forward` zero-defaulting: it
/// used to panic on a continuous leaf that received no tape write.
#[test]
fn a_column_that_reaches_no_row_is_zero() {
    let model = Model::new();
    let points = model.points();
    let (inputs, outputs) = (model.inputs(), model.outputs());

    let w = inputs.block(model.w.id()).unwrap();
    for j in [
        jacobian_jvp(&model.client, &points, &inputs, &outputs).unwrap(),
        jacobian_vjp(&model.client, &points, &inputs, &outputs).unwrap(),
    ] {
        let j = into_vec(j);
        let column: Vec<f64> = j.iter().skip(w.offset).step_by(D_IN).copied().collect();
        assert_eq!(column, vec![0.0; BATCH * D_OUT]);
    }
}

/// A row that is a leaf rather than a mapped node is rejected, in both paths.
///
/// The truth would be `∂x/∂x = 1`, but neither pass seeds a leaf, so both would return a
/// zero row — agreeing with each other on the wrong answer.
#[test]
fn a_leaf_row_is_rejected() {
    let model = Model::new();
    let points = model.points();
    let inputs = model.inputs();
    let outputs = layout(&model.client, &[model.u.id(), model.x.id()]);

    let expected = Err(SkoroError::NotMapped { id: model.x.id() });
    assert_eq!(
        jacobian_jvp(&model.client, &points, &inputs, &outputs).map(|j| j.dims()),
        expected
    );
    assert_eq!(
        jacobian_vjp(&model.client, &points, &inputs, &outputs).map(|j| j.dims()),
        expected
    );
}
