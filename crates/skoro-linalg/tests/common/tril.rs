use approx::assert_abs_diff_eq;
use skoro_backend::{Backend, ElementConversion, TensorData};
use skoro_model::{
    hash_map_builders::{points, tangent_frames},
    model::ModelClient,
    q::Q,
    test_utils::{
        into_vec, tensor, test_hvp_numerically, test_jvp_numerically, test_vjp_numerically,
    },
};
use skoro_operation::Input;
use skoro_riemann::euclidean::Euclidean;

use skoro_linalg::tril::{Tril, TrilOps};
use skoro_pointwise::PointwiseOps;

/// Regression: the packed diagonal must survive a following elementwise `abs`.
///
/// `Tril::diag` gathers with `select(1, ..)`, which `burn-ndarray` returns in column-major
/// layout, and `burn-ndarray`'s SIMD `abs` scrambles such an input once it has 32 or more
/// elements. `Normal::factor` computes its log-determinant as `diag().abs().log().sum()`, so this
/// is the composition that corrupted every `Normal` at a large enough batch. Skoro moved to
/// burn-flex, whose kernels are stride-aware; this pins the symptom end to end. The batch is sized
/// past the threshold (`BATCH * N == 48`); anything under 32 elements passes even against the
/// defect.
pub fn diag_survives_abs_impl<B: Backend>()
where
    Q<B, Tril>: TrilOps<B>,
{
    const N: usize = 6;
    const BATCH: usize = 8;
    const K: usize = N * (N + 1) / 2;

    let model = ModelClient::<B>::default();
    let l = model.add_parameter(Tril(N));
    let d = l.diag().unwrap().abs().unwrap();

    // Every entry distinct, with alternating signs so that `abs` is genuinely exercised
    // and any reordering of the result is detectable.
    let packed: Vec<f64> = (0..BATCH * K)
        .map(|i| {
            let v = (i + 1) as f64;
            if i % 2 == 0 { -v } else { v }
        })
        .collect();

    // The i-th diagonal entry of a packed lower triangle sits at index i * (i + 3) / 2.
    let expected: Vec<f64> = (0..BATCH)
        .flat_map(|b| (0..N).map(move |i| ((b * K + i * (i + 3) / 2 + 1) as f64).elem()))
        .collect();

    let points = points().with_tensor(
        l.id(),
        Input::<B>::Float(tensor::<B, 2>(
            TensorData::new(packed, [BATCH, K]),
            &Default::default(),
        )),
    );

    let eval = d.evaluate(&points).unwrap();
    assert_abs_diff_eq!(
        into_vec(eval).as_slice(),
        expected.as_slice(),
        epsilon = f64::from_elem(1e-6),
    );
}

pub fn vector_product_impl<B: Backend>()
where
    Q<B, Tril>: TrilOps<B>,
{
    let model = ModelClient::<B>::default();

    let l = model.add_parameter(Tril(2));
    let v = model.add_parameter(Euclidean::<2>);
    let lv = l.vector_product(&v).unwrap();

    // ForwardOnly
    let points = points()
        .with_tensor(
            l.id(),
            Input::<B>::Float(tensor([[1.0, 2.0, 3.0]], &Default::default())),
        )
        .with_tensor(
            v.id(),
            Input::<B>::Float(tensor([[4.0, 5.0]], &Default::default())),
        );

    let eval = lv.evaluate(&points).unwrap();
    assert_abs_diff_eq!(
        into_vec(eval).as_slice(),
        into_vec(tensor::<B, 1>([4.0, 23.0], &Default::default())).as_slice(),
        epsilon = f64::from_elem(1e-6),
    );

    // JVP
    let tangents = tangent_frames()
        .with_tensor(
            l.id(),
            tensor::<B, 1>([0.1, 0.2, 0.3], &Default::default()).reshape([1, 3, 1]),
        )
        .with_tensor(
            v.id(),
            tensor::<B, 1>([0.4, 0.5], &Default::default()).reshape([1, 2, 1]),
        );

    let jvp = lv.evaluate_jvp(&points, &tangents).unwrap();
    assert_abs_diff_eq!(
        into_vec(jvp.value).as_slice(),
        into_vec(tensor::<B, 1>([4.0, 23.0], &Default::default())).as_slice(),
        epsilon = f64::from_elem(1e-6),
    );
    assert_abs_diff_eq!(
        into_vec(jvp.tangents.get(&lv.id()).unwrap().clone()).as_slice(),
        into_vec(tensor::<B, 1>([0.8, 4.6], &Default::default())).as_slice(),
        epsilon = f64::from_elem(1e-6),
    );

    // VJP
    let vjp = lv.evaluate_ord1(&points).unwrap();
    assert_abs_diff_eq!(
        into_vec(vjp.value).as_slice(),
        into_vec(tensor::<B, 1>([4.0, 23.0], &Default::default())).as_slice(),
        epsilon = f64::from_elem(1e-6),
    );
    assert_abs_diff_eq!(
        into_vec(vjp.cotangents.get(&l.id()).unwrap().clone()).as_slice(),
        into_vec(tensor::<B, 1>([4.0, 4.0, 5.0], &Default::default())).as_slice(),
        epsilon = f64::from_elem(1e-6),
    );
    assert_abs_diff_eq!(
        into_vec(vjp.cotangents.get(&v.id()).unwrap().clone()).as_slice(),
        into_vec(tensor::<B, 1>([3.0, 3.0], &Default::default())).as_slice(),
        epsilon = f64::from_elem(1e-6),
    );

    // HVP
    let tangents = tangents
        .build()
        .into_iter()
        .map(|(id, frame)| (id, frame.squeeze_dim(2)))
        .collect();
    let hvp = lv.evaluate_ord2(&points, &tangents).unwrap();
    assert_abs_diff_eq!(
        into_vec(hvp.value).as_slice(),
        into_vec(tensor::<B, 1>([4.0, 23.0], &Default::default())).as_slice(),
        epsilon = f64::from_elem(1e-6),
    );
    assert_abs_diff_eq!(
        into_vec(hvp.tangents.get(&lv.id()).unwrap().clone()).as_slice(),
        into_vec(tensor::<B, 1>([0.8, 4.6], &Default::default())).as_slice(),
        epsilon = f64::from_elem(1e-6),
    );
    assert_abs_diff_eq!(
        into_vec(hvp.ord1_cotangents.get(&l.id()).unwrap().clone()).as_slice(),
        into_vec(tensor::<B, 1>([4.0, 4.0, 5.0], &Default::default())).as_slice(),
        epsilon = f64::from_elem(1e-6),
    );
    assert_abs_diff_eq!(
        into_vec(hvp.ord1_cotangents.get(&v.id()).unwrap().clone()).as_slice(),
        into_vec(tensor::<B, 1>([3.0, 3.0], &Default::default())).as_slice(),
        epsilon = f64::from_elem(1e-6),
    );

    // hd_L = pack(omega * dv^T) = pack([[1.0, 1.0]]^T * [[0.4, 0.5]]) = pack([[0.4, 0.5], [0.4, 0.5]])
    // L11=0.4, L21=0.4, L22=0.5 => [0.4, 0.4, 0.5]
    assert_abs_diff_eq!(
        into_vec(hvp.ord2_cotangents.get(&l.id()).unwrap().clone()).as_slice(),
        into_vec(tensor::<B, 1>([0.4, 0.4, 0.5], &Default::default())).as_slice(),
        epsilon = f64::from_elem(1e-6),
    );
    // hd_v = dL^T * omega = [[0.1, 0.2], [0.0, 0.3]] * [1.0, 1.0]^T = [0.3, 0.3]
    assert_abs_diff_eq!(
        into_vec(hvp.ord2_cotangents.get(&v.id()).unwrap().clone()).as_slice(),
        into_vec(tensor::<B, 1>([0.3, 0.3], &Default::default())).as_slice(),
        epsilon = f64::from_elem(1e-6),
    );
}

pub fn invert_impl<B: Backend>()
where
    Q<B, Tril>: TrilOps<B>,
{
    let model = ModelClient::<B>::default();

    let l = model.add_parameter(Tril(2));
    let linv = l.invert().unwrap();

    // ForwardOnly
    let points = points().with_tensor(
        l.id(),
        Input::<B>::Float(tensor([[1.0, 2.0, 3.0]], &Default::default())),
    );
    let eval = linv.evaluate(&points).unwrap();
    assert_abs_diff_eq!(
        into_vec(eval).as_slice(),
        into_vec(tensor::<B, 1>(
            [1.0, -2.0 / 3.0, 1.0 / 3.0],
            &Default::default()
        ))
        .as_slice(),
        epsilon = f64::from_elem(1e-6),
    );

    // JVP
    let tangents = tangent_frames().with_tensor(
        l.id(),
        tensor::<B, 1>([0.2, 0.1, 0.4], &Default::default()).reshape([1, 3, 1]),
    );
    let jvp = linv.evaluate_jvp(&points, &tangents).unwrap();
    assert_abs_diff_eq!(
        into_vec(jvp.value).as_slice(),
        into_vec(tensor::<B, 1>(
            [1.0, -2.0 / 3.0, 1.0 / 3.0],
            &Default::default()
        ))
        .as_slice(),
        epsilon = f64::from_elem(1e-6),
    );
    assert_abs_diff_eq!(
        into_vec(jvp.tangents.get(&linv.id()).unwrap().clone()).as_slice(),
        into_vec(tensor::<B, 1>(
            [-0.2, 17.0 / 90.0, -2.0 / 45.0],
            &Default::default()
        ))
        .as_slice(),
        epsilon = f64::from_elem(1e-6),
    );

    // VJP
    let vjp = linv.evaluate_ord1(&points).unwrap();
    assert_abs_diff_eq!(
        into_vec(vjp.value).as_slice(),
        into_vec(tensor::<B, 1>(
            [1.0, -2.0 / 3.0, 1.0 / 3.0],
            &Default::default()
        ))
        .as_slice(),
        epsilon = f64::from_elem(1e-6),
    );
    assert_abs_diff_eq!(
        into_vec(vjp.cotangents.get(&l.id()).unwrap().clone()).as_slice(),
        into_vec(tensor::<B, 1>(
            // ∂(Y₀₀ + Y₁₀ + Y₁₁)/∂L for Y = L⁻¹: the pullback −Yᵀ Ω Yᵀ.
            [-1.0 / 3.0, -1.0 / 3.0, 1.0 / 9.0],
            &Default::default()
        ))
        .as_slice(),
        epsilon = f64::from_elem(1e-6),
    );

    // HVP
    let tangents = tangents
        .build()
        .into_iter()
        .map(|(id, frame)| (id, frame.squeeze_dim(2)))
        .collect();
    let hvp = linv.evaluate_ord2(&points, &tangents).unwrap();
    assert_abs_diff_eq!(
        into_vec(hvp.value).as_slice(),
        into_vec(tensor::<B, 1>(
            [1.0, -2.0 / 3.0, 1.0 / 3.0],
            &Default::default()
        ))
        .as_slice(),
        epsilon = f64::from_elem(1e-6),
    );
    assert_abs_diff_eq!(
        into_vec(hvp.tangents.get(&linv.id()).unwrap().clone()).as_slice(),
        into_vec(tensor::<B, 1>(
            [-0.2, 17.0 / 90.0, -2.0 / 45.0],
            &Default::default()
        ))
        .as_slice(),
        epsilon = f64::from_elem(1e-6),
    );
    assert_abs_diff_eq!(
        into_vec(hvp.ord1_cotangents.get(&l.id()).unwrap().clone()).as_slice(),
        into_vec(tensor::<B, 1>(
            // ∂(Y₀₀ + Y₁₀ + Y₁₁)/∂L for Y = L⁻¹: the pullback −Yᵀ Ω Yᵀ.
            [-1.0 / 3.0, -1.0 / 3.0, 1.0 / 9.0],
            &Default::default()
        ))
        .as_slice(),
        epsilon = f64::from_elem(1e-6),
    );
    assert_abs_diff_eq!(
        into_vec(hvp.ord2_cotangents.get(&l.id()).unwrap().clone()).as_slice(),
        into_vec(tensor::<B, 1>(
            // The Hessian of Y₀₀ + Y₁₀ + Y₁₁ at L = [1, 2, 3] along dL = [0.2, 0.1, 0.4].
            [7.0 / 90.0, 1.0 / 9.0, -17.0 / 270.0],
            &Default::default()
        ))
        .as_slice(),
        epsilon = f64::from_elem(1e-6),
    );
}

pub fn diag_impl<B: Backend>()
where
    Q<B, Tril>: TrilOps<B>,
{
    let model = ModelClient::<B>::default();

    let l = model.add_parameter(Tril(2));
    let d = l.diag().unwrap();

    // ForwardOnly
    let points = points().with_tensor(
        l.id(),
        Input::<B>::Float(tensor([[1.0, 2.0, 3.0]], &Default::default())),
    );
    let eval = d.evaluate(&points).unwrap();
    assert_abs_diff_eq!(
        into_vec(eval).as_slice(),
        into_vec(tensor::<B, 1>([1.0, 3.0], &Default::default())).as_slice(),
        epsilon = f64::from_elem(1e-6),
    );

    // Jvp
    let tangents = tangent_frames().with_tensor(
        l.id(),
        tensor::<B, 1>([0.2, 0.1, 0.4], &Default::default()).reshape([1, 3, 1]),
    );
    let jvp = d.evaluate_jvp(&points, &tangents).unwrap();
    assert_abs_diff_eq!(
        into_vec(jvp.value).as_slice(),
        into_vec(tensor::<B, 1>([1.0, 3.0], &Default::default())).as_slice(),
        epsilon = f64::from_elem(1e-6),
    );
    assert_abs_diff_eq!(
        into_vec(jvp.tangents.get(&d.id()).unwrap().clone()).as_slice(),
        into_vec(tensor::<B, 1>([0.2, 0.4], &Default::default())).as_slice(),
        epsilon = f64::from_elem(1e-6),
    );

    // Vjp
    let vjp = d.evaluate_ord1(&points).unwrap();
    assert_abs_diff_eq!(
        into_vec(vjp.value).as_slice(),
        into_vec(tensor::<B, 1>([1.0, 3.0], &Default::default())).as_slice(),
        epsilon = f64::from_elem(1e-6),
    );
    assert_abs_diff_eq!(
        into_vec(vjp.cotangents.get(&l.id()).unwrap().clone()).as_slice(),
        into_vec(tensor::<B, 1>([1.0, 0.0, 1.0], &Default::default())).as_slice(),
        epsilon = f64::from_elem(1e-6),
    );

    // Hvp
    let tangents = tangents
        .build()
        .into_iter()
        .map(|(id, frame)| (id, frame.squeeze_dim(2)))
        .collect();
    let hvp = d.evaluate_ord2(&points, &tangents).unwrap();
    assert_abs_diff_eq!(
        into_vec(hvp.value).as_slice(),
        into_vec(tensor::<B, 1>([1.0, 3.0], &Default::default())).as_slice(),
        epsilon = f64::from_elem(1e-6),
    );
    assert_abs_diff_eq!(
        into_vec(hvp.tangents.get(&d.id()).unwrap().clone()).as_slice(),
        into_vec(tensor::<B, 1>([0.2, 0.4], &Default::default())).as_slice(),
        epsilon = f64::from_elem(1e-6),
    );
    assert_abs_diff_eq!(
        into_vec(hvp.ord1_cotangents.get(&l.id()).unwrap().clone()).as_slice(),
        into_vec(tensor::<B, 1>([1.0, 0.0, 1.0], &Default::default())).as_slice(),
        epsilon = f64::from_elem(1e-6),
    );
    assert_abs_diff_eq!(
        into_vec(hvp.ord2_cotangents.get(&l.id()).unwrap().clone()).as_slice(),
        into_vec(tensor::<B, 1>([0.0, 0.0, 0.0], &Default::default())).as_slice(),
        epsilon = f64::from_elem(1e-6),
    );
}

/// `TrilInvert`'s three derivative modes against finite differences.
///
/// The triangle is `3 × 3` and far from both the identity and the diagonal: a pullback that
/// forgot its transposes — `−Y Ω Y` for `−Yᵀ Ω Yᵀ` — agrees with the true one on a diagonal
/// fixture and nowhere else.
pub fn invert_derivatives_impl<B: Backend>(delta: f64, epsilon: f64)
where
    Q<B, Tril>: TrilOps<B>,
{
    let model = ModelClient::<B>::default();
    let l = model.add_parameter(Tril(3));
    let linv = l.invert().unwrap();
    let p = points()
        .with_tensor(
            l.id(),
            Input::<B>::Float(tensor(
                [[1.5, -0.7, 0.9, 0.4, 1.1, 1.3]],
                &Default::default(),
            )),
        )
        .build();

    test_jvp_numerically(&l, &linv, p.clone(), delta, epsilon);
    test_vjp_numerically(&l, &linv, p.clone(), delta, epsilon);
    test_hvp_numerically(&l, &linv, p, delta, epsilon);
}

/// `TrilInvert` and `skoro-dense` must agree on the same triangle.
///
/// The two implementations are deliberately kept apart — `TrilInvert` is a differentiable
/// graph operation on every backend and stays there (`docs/skoro.md`
/// §8), while `skoro-dense` is host `f64` for the estimators. This is the check that the
/// fence did not let them drift: the packed forward substitution and the host triangular
/// solve compute the same inverse.
///
/// The triangle is `4 × 4` and far from the identity, where a transposed index or a
/// mis-derived off-diagonal is visible.
pub fn invert_agrees_with_dense_impl<B: Backend>()
where
    Q<B, Tril>: TrilOps<B>,
{
    // Row-major packed lower triangle: element (i, j≤i) is at `i(i+1)/2 + j`.
    const N: usize = 4;
    #[rustfmt::skip]
    const PACKED: [f64; 10] = [
        2.0,
       -1.5,  3.0,
        0.75, 2.5,  1.25,
       -2.0,  0.5, -1.75, 4.0,
    ];

    let model = ModelClient::<B>::default();
    let l = model.add_parameter(Tril(N));
    let linv = l.invert().unwrap();
    let points = points().with_tensor(
        l.id(),
        Input::<B>::Float(tensor([PACKED], &Default::default())),
    );
    let packed_inverse = into_vec(linv.evaluate(&points).unwrap());

    // The same triangle, unpacked, inverted on the host by a triangular solve against `I`.
    let mut dense_l = vec![0.0; N * N];
    for i in 0..N {
        for j in 0..=i {
            dense_l[i * N + j] = PACKED[i * (i + 1) / 2 + j];
        }
    }
    let dense_inverse = skoro_dense::factor::solve_lower(
        &skoro_dense::dense::Dense::from_row_major(N, N, dense_l).unwrap(),
        &skoro_dense::dense::Dense::identity(N),
    )
    .unwrap();

    for i in 0..N {
        for j in 0..=i {
            assert_abs_diff_eq!(
                packed_inverse[i * (i + 1) / 2 + j],
                f64::from_elem(dense_inverse.get(i, j)),
                epsilon = f64::from_elem(1e-5),
            );
        }
    }
}

/// `retract(p, inverse_retract(p, q)) == q` — the law every manifold owes.
///
/// `Tril` is a vector space, so the retraction is `Addition` and the inverse-retraction has to be
/// `q − p`. The trap is argument order: `DynManifold::inverse_retract(point, other)` forwards
/// `vec![point, other]`, and the shared `Subtraction` operation destructures `[x, y]` and returns
/// `x − y` — the opposite. Using it here gives `2p − q`, which is a well-shaped tangent of the
/// right dimension that no shape check rejects and no other test in the workspace exercises.
///
/// Asserted through the *dyn* surface, because that is the one a joint residual over a
/// heterogeneous blanket reaches: it looks a quantity's manifold up at run time and has no typed
/// path to fall back on.
pub fn inverse_retraction_round_trip_impl<B: Backend>()
where
    Tril: skoro_riemann::Manifold<B>,
{
    use skoro_riemann::DynManifold;

    let manifold = Tril(3);
    // Two distinct points of the 6-component lower triangle, neither near the other.
    let p = tensor::<B, 2>([[1.0, -2.0, 0.5, 3.0, -1.5, 4.0]], &Default::default());
    let q = tensor::<B, 2>([[-0.25, 5.0, 2.5, -3.0, 0.75, 1.0]], &Default::default());

    let delta = DynManifold::inverse_retract(&manifold, p.clone(), q.clone());
    // `q ⊖ p`, not `p ⊖ q`: the sign is half of what this test is for.
    assert_abs_diff_eq!(
        into_vec(delta.clone()).as_slice(),
        into_vec(q.clone() - p.clone()).as_slice()
    );

    let round_tripped = DynManifold::retract(&manifold, p, delta);
    assert_abs_diff_eq!(into_vec(round_tripped).as_slice(), into_vec(q).as_slice());
}
