pub mod group;

use skoro_backend::Backend;
use skoro_backend::Tensor;
use skoro_graph::quantity::ContinuousQuantity;
use skoro_model::{
    hash_map_builders::points,
    model::ModelClient,
    ops::AssociatedTangentVector,
    test_utils::{
        tensor, test_hvp_cross_numerically, test_hvp_numerically, test_jvp_numerically,
        test_vjp_numerically,
    },
};
use skoro_operation::Input;
use skoro_riemann::euclidean::Euclidean;
use skoro_riemann::real::Real;

use skoro_se3::quantity::SE3;

const RETRACT_POSES: [[f64; 12]; 5] = [
    // Exp((0.3, -0.2, 0.5)), |w| = 0.62 — generic.
    [
        0.8595338985586632,
        -0.497991537002922,
        -0.11491695393636675,
        0.43986763295823095,
        0.8353156052067086,
        -0.3297943376922551,
        0.26022671404809444,
        0.23292116428443663,
        0.937032437284918,
        5.0,
        6.0,
        7.0,
    ],
    // Exact identity attitude at the origin.
    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    // Exp((1.2, 0.8, -1.5)), |w| = 2.08 — large rotation, large translation.
    [
        0.006696081733020831,
        0.9590527535966131,
        -0.28314833269538964,
        -0.29914149993827394,
        -0.26826694062462053,
        -0.9157222349504167,
        -0.9541852679139963,
        0.09083316787749285,
        0.28509614187013255,
        -3.0,
        0.5,
        12.0,
    ],
    // Exp((0, 0, 1e-4)) — |w| deep inside the Taylor branch of `exp_jacobian_coeffs`.
    [
        0.999999995,
        -9.999999983333334e-05,
        0.0,
        9.999999983333334e-05,
        0.999999995,
        0.0,
        0.0,
        0.0,
        1.0,
        0.1,
        -0.2,
        0.3,
    ],
    // Exp((-0.7, 1.1, 0.25)), |w| = 1.33 — generic.
    [
        0.45187853044804804,
        -0.5144418593332605,
        0.7288040663208806,
        -0.14890357336615478,
        0.7620140574243981,
        0.6302081419074148,
        -0.8795643919343841,
        -0.39329905880048077,
        0.2677355613058403,
        2.5,
        -4.0,
        0.75,
    ],
];

const RETRACT_TANGENTS: [[f64; 6]; 5] = [
    // |w| = 0.78, large lever arm |v| = 37.
    [0.4, -0.3, 0.6, 10.0, 20.0, 30.0],
    // |w| = 2.3e-5 — Taylor branch of `exp`.
    [1e-05, -2e-05, 5e-06, 1.0, -2.0, 0.5],
    // |w| = 1.79 — large rotation and translation.
    [1.5, -0.9, 0.4, -5.0, 3.0, -8.0],
    // Exactly zero rotation; translation only.
    [0.0, 0.0, 0.0, 2.0, -1.0, 4.0],
    // |w| = 1.35, small translation.
    [-0.25, 0.75, -1.1, 0.5, 0.25, -0.75],
];

const INVERSE_POSES: [[f64; 12]; 5] = [
    // Relative attitude 1.09 rad — generic.
    [
        0.8595338985586632,
        -0.497991537002922,
        -0.11491695393636675,
        0.43986763295823095,
        0.8353156052067086,
        -0.3297943376922551,
        0.26022671404809444,
        0.23292116428443663,
        0.937032437284918,
        0.1,
        0.5,
        0.8,
    ],
    // Both near identity; relative attitude 1e-05 rad — Taylor branch of `log`.
    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    // Relative attitude 2.56 rad — near the far end of `log`'s range.
    [
        0.848155401633665,
        -0.4606492927730993,
        -0.26160016006738207,
        0.014047532872113883,
        0.5132040817079259,
        -0.8581516400605033,
        0.5295612160079733,
        0.7241711120902076,
        0.44174780012376824,
        -2.0,
        3.0,
        1.0,
    ],
    // Identical poses: Delta-r is exactly I and q is exactly 0.
    [
        0.9937772959432721,
        0.0595199734937639,
        0.0941491307606165,
        -0.03960732051223486,
        0.9788428062071254,
        -0.20074366963468865,
        -0.10410545725138103,
        0.1957655063893064,
        0.9751091837730888,
        1.0,
        1.0,
        1.0,
    ],
    // Relative attitude 0.94 rad — generic.
    [
        0.6161625646378023,
        -0.7833584726361584,
        -0.08181196298898546,
        0.6749808438280085,
        0.471659059560269,
        0.5673963270941681,
        -0.40588736664768926,
        -0.4048298838819432,
        0.8193706186530835,
        0.5,
        -0.5,
        0.25,
    ],
];

const INVERSE_OTHER_POSES: [[f64; 12]; 5] = [
    // Relative attitude 1.09 rad — generic.
    [
        0.8706442127616613,
        0.20524274417783794,
        0.4470504118631115,
        -0.2711409754124633,
        0.9585085210744951,
        0.08799992318156921,
        -0.4104402833994308,
        -0.1978303085726115,
        0.8901696146089577,
        0.1,
        0.3,
        -0.4,
    ],
    // Both near identity; relative attitude 1e-05 rad — Taylor branch of `log`.
    [
        0.99999999995,
        -9.999999999833334e-06,
        0.0,
        9.999999999833334e-06,
        0.99999999995,
        0.0,
        0.0,
        0.0,
        1.0,
        0.001,
        0.0,
        0.0,
    ],
    // Relative attitude 2.56 rad — near the far end of `log`'s range.
    [
        0.3386182054122746,
        -0.13030189996224895,
        0.931857889288704,
        -0.6633562535430216,
        0.6693091027061373,
        0.33463981520445507,
        -0.6673051714536139,
        -0.7314688919570903,
        0.140203667035957,
        4.0,
        -1.0,
        2.0,
    ],
    // Identical poses: Delta-r is exactly I and q is exactly 0.
    [
        0.9937772959432721,
        0.0595199734937639,
        0.0941491307606165,
        -0.03960732051223486,
        0.9788428062071254,
        -0.20074366963468865,
        -0.10410545725138103,
        0.1957655063893064,
        0.9751091837730888,
        1.0,
        1.0,
        1.0,
    ],
    // Relative attitude 0.94 rad — generic.
    [
        0.13931030721498838,
        -0.9790773684696483,
        -0.14832445804456665,
        0.9538864506320381,
        0.17289819766513514,
        -0.2453708469740452,
        0.26588207462008046,
        -0.10730200275249617,
        0.9580151369373165,
        -0.75,
        2.0,
        0.1,
    ],
];

#[allow(unused)]
pub fn retract_impl<B: Backend>(delta: f64, epsilon: f64)
where
    SE3: ContinuousQuantity<B> + AssociatedTangentVector<Euclidean<6>, B>,
{
    let model = ModelClient::<B>::default();
    let pose = model.add_parameter(SE3);
    let tangent = model.add_parameter(Euclidean::<6>);
    let out = pose.retract(&tangent).unwrap();
    let point = points()
        .with_quantity(&pose, tensor::<B, 2>(RETRACT_POSES, &Default::default()))
        .with_tensor(
            tangent.id(),
            tensor::<B, 2>(RETRACT_TANGENTS, &Default::default()),
        )
        .build();
    test_jvp_numerically(&pose, &out, point.clone(), delta, epsilon);
    test_vjp_numerically(&pose, &out, point.clone(), delta, epsilon);
    test_hvp_numerically(&pose, &out, point.clone(), delta, epsilon);
    test_jvp_numerically(&tangent, &out, point.clone(), delta, epsilon);
    test_vjp_numerically(&tangent, &out, point.clone(), delta, epsilon);
    test_hvp_numerically(&tangent, &out, point.clone(), delta, epsilon);
    // The VJP of the retraction depends only on `(ω, v)`, never on the pose, so both diagonal HVP
    // blocks above are identically zero. The pose's second-order cotangent only comes alive when
    // the tangent argument is the one being perturbed, so the mixed blocks are what actually pin
    // down the HVP.
    test_hvp_cross_numerically(&tangent, &pose, &out, point.clone(), delta, epsilon);
    test_hvp_cross_numerically(&pose, &tangent, &out, point.clone(), delta, epsilon);
}

#[allow(unused)]
pub fn inverse_retract_impl<B: Backend>(delta: f64, epsilon: f64)
where
    SE3: ContinuousQuantity<B> + AssociatedTangentVector<Euclidean<6>, B>,
{
    let model = ModelClient::<B>::default();
    let pose = model.add_parameter(SE3);
    let other_pose = model.add_parameter(SE3);
    let out = pose.inverse_retract::<Euclidean<6>>(&other_pose).unwrap();
    let point = points()
        .with_quantity(&pose, tensor::<B, 2>(INVERSE_POSES, &Default::default()))
        .with_tensor(
            other_pose.id(),
            tensor::<B, 2>(INVERSE_OTHER_POSES, &Default::default()),
        )
        .build();
    test_jvp_numerically(&pose, &out, point.clone(), delta, epsilon);
    test_vjp_numerically(&pose, &out, point.clone(), delta, epsilon);
    test_hvp_numerically(&pose, &out, point.clone(), delta, epsilon);
    test_jvp_numerically(&other_pose, &out, point.clone(), delta, epsilon);
    test_vjp_numerically(&other_pose, &out, point.clone(), delta, epsilon);
    test_hvp_numerically(&other_pose, &out, point.clone(), delta, epsilon);
    test_hvp_cross_numerically(&pose, &other_pose, &out, point.clone(), delta, epsilon);
    test_hvp_cross_numerically(&other_pose, &pose, &out, point.clone(), delta, epsilon);
}

/// A layout is sized by **manifold dimension**, never by storage size.
///
/// This is the invariant every downstream shape depends on, and `SE3`/`SO3` are the only
/// place in the workspace where the two diverge: `SE3` stores 12 components for 6 degrees
/// of freedom, `SO3` stores 9 for 3. A layout that used `num_components` would produce a
/// 21-wide vector here instead of 10, and every Jacobian built on it would be wrong in a
/// way no dimension check would catch.
pub fn layout_is_sized_by_dimension_impl<B: Backend>()
where
    SE3: ContinuousQuantity<B>,
    skoro_so3::quantity::SO3: ContinuousQuantity<B>,
{
    use skoro_graph::tangent_layout::TangentLayout;
    use skoro_so3::quantity::SO3;

    let model = ModelClient::<B>::default();
    let pose = model.add_parameter(SE3);
    let rot = model.add_parameter(SO3);
    let scalar = model.add_parameter(Real);

    let layout = model.server_call(|server| {
        TangentLayout::from_ids(server.graph(), &[pose.id(), rot.id(), scalar.id()]).unwrap()
    });

    // 6 + 3 + 1, not 12 + 9 + 1.
    assert_eq!(layout.total(), 10);
    let blocks = layout.blocks();
    assert_eq!((blocks[0].offset, blocks[0].dimension), (0, 6));
    assert_eq!((blocks[1].offset, blocks[1].dimension), (6, 3));
    assert_eq!((blocks[2].offset, blocks[2].dimension), (9, 1));

    // The identity seeding agrees: every frame is `[batch, d_i, 10]`.
    let frames = layout.identity_frames::<B>(1, &Default::default());
    assert_eq!(frames[&pose.id()].dims(), [1, 6, 10]);
    assert_eq!(frames[&rot.id()].dims(), [1, 3, 10]);
}

/// `retract_flat` steps only the layout's quantities, through their own retractions.
///
/// The contrast with `retract_mut` is the point: that one demands a tangent for every
/// point in the map, which an optimizer cannot supply because its `PointMap` also holds
/// constants and quantities it is not stepping.
pub fn retract_flat_is_partial_impl<B: Backend>(delta: f64, _epsilon: f64)
where
    SE3: ContinuousQuantity<B>,
{
    use skoro_core::kind::Continuous;
    use skoro_graph::{point_map::PointMap, quantity::QuantityType, tangent_layout::TangentLayout};
    use skoro_riemann::real::Real;

    let model = ModelClient::<B>::default();
    let pose = model.add_parameter(SE3);
    let untouched = model.add_parameter(Real);

    let layout =
        model.server_call(|server| TangentLayout::from_ids(server.graph(), &[pose.id()]).unwrap());

    let mut points = PointMap::<B>::default();
    points.insert(
        pose.id(),
        Input::Float(tensor::<B, 2>([RETRACT_POSES[0]], &Default::default())),
        <SE3 as QuantityType<B, Continuous>>::into_dyn(SE3),
    );
    points.insert(
        untouched.id(),
        Input::Float(tensor::<B, 2>([[1.0]], &Default::default())),
        <Real as QuantityType<B, Continuous>>::into_dyn(Real),
    );

    let step = Tensor::<B, 2>::ones([1, 6], &Default::default()).mul_scalar(delta);
    let stepped = points.retract_flat(&layout, step).unwrap();

    let before = points.get(&pose.id()).unwrap().0.clone().unwrap_float();
    let after = stepped.get(&pose.id()).unwrap().0.clone().unwrap_float();
    assert!(
        before
            .into_data()
            .convert::<f64>()
            .to_vec::<f64>()
            .unwrap()
            .iter()
            .zip(after.into_data().convert::<f64>().to_vec::<f64>().unwrap())
            .any(|(a, b)| *a != b),
        "the pose named by the layout should have moved"
    );

    let quiet_before = points
        .get(&untouched.id())
        .unwrap()
        .0
        .clone()
        .unwrap_float();
    let quiet_after = stepped
        .get(&untouched.id())
        .unwrap()
        .0
        .clone()
        .unwrap_float();
    assert_eq!(
        quiet_before
            .into_data()
            .convert::<f64>()
            .to_vec::<f64>()
            .unwrap(),
        quiet_after
            .into_data()
            .convert::<f64>()
            .to_vec::<f64>()
            .unwrap(),
        "a quantity outside the layout must not move"
    );
}

/// `unflatten(flatten(map)) == map` over **every** manifold shape at once.
///
/// The mix is the point. `layout_is_sized_by_dimension` checks the offsets; this checks that
/// the data actually survives a round trip through them, and it is the only place `Tril`,
/// `SO3` and `SE3` appear in one layout together — `skoro-graph`'s own suite cannot reach
/// the group manifolds, and `skoro-so3`'s cannot reach `SE3`.
///
/// A block sized by storage rather than by dimension fails on the width; a block written at
/// its index rather than its offset fails on the values.
pub fn mixed_layout_round_trip_impl<B: Backend>()
where
    SE3: ContinuousQuantity<B>,
    skoro_so3::quantity::SO3: ContinuousQuantity<B>,
{
    use skoro_graph::tangent_layout::TangentLayout;
    use skoro_linalg::tril::Tril;
    use skoro_riemann::euclidean::EuclideanD;
    use skoro_so3::quantity::SO3;
    use std::collections::HashMap;

    let model = ModelClient::<B>::default();
    let real = model.add_parameter(Real);
    let euclidean = model.add_parameter(EuclideanD(2));
    let tril = model.add_parameter(Tril(3));
    let rotation = model.add_parameter(SO3);
    let pose = model.add_parameter(SE3);

    let ids = [
        real.id(),
        euclidean.id(),
        tril.id(),
        rotation.id(),
        pose.id(),
    ];
    let layout = model
        .server_call(|server| TangentLayout::from_ids(server.graph(), &ids))
        .unwrap();

    // 1 + 2 + 6 + 3 + 6. `SO3` is 3 and `SE3` is 6, never 9 and 12.
    assert_eq!(layout.total(), 18);

    // Distinct, non-repeating values, so a block landing at the wrong offset is visible.
    let mut original = HashMap::new();
    let mut next = 0.5f64;
    for block in layout.blocks() {
        let values: Vec<f64> = (0..block.dimension)
            .map(|_| {
                next += 1.25;
                next
            })
            .collect();
        original.insert(
            block.id,
            Tensor::<B, 1>::from_floats(values.as_slice(), &Default::default())
                .reshape([1, block.dimension]),
        );
    }

    let flat = layout
        .flatten_vectors(&original, 1, &Default::default())
        .unwrap();
    assert_eq!(flat.dims(), [1, 18]);
    let recovered = layout.unflatten_vector(flat).unwrap();

    assert_eq!(recovered.len(), original.len());
    for (id, expected) in &original {
        let actual = recovered.get(id).expect("every block round-trips");
        assert_eq!(actual.dims(), expected.dims());
        let to_f64 = |t: Tensor<B, 2>| t.into_data().convert::<f64>().to_vec::<f64>().unwrap();
        assert_eq!(to_f64(actual.clone()), to_f64(expected.clone()));
    }
}

#[macro_export]
macro_rules! tests {
    ($B:ty, $delta:literal, $epsilon:literal) => {
        /// The pose operations, differentiated in the body-frame chart.
        mod group {
            #[test]
            fn compose() {
                super::common::group::compose_impl::<$B>($delta, $epsilon);
            }

            #[test]
            fn invert() {
                super::common::group::invert_impl::<$B>($delta, $epsilon);
            }
        }

        /// `retract` on `SE3`: the rotation through `SO3`'s own retraction and the translation added,
        /// composed as one rigid-body step.
        mod retraction {
            #[test]
            fn retract() {
                super::common::retract_impl::<$B>($delta, $epsilon);
            }
        }

        /// `b ⊖ a` on `SE3`, a genuine `Log(a⁻¹b)`. This is the residual a Gaussian factor over poses
        /// whitens, and it is the only place in the workspace where the manifold dimension and the storage
        /// size disagree by a factor a shape check would not catch.
        mod inverse_retraction {
            #[test]
            fn inverse_retract() {
                super::common::inverse_retract_impl::<$B>($delta, $epsilon);
            }
        }

        /// Layout mechanics where dimension and storage genuinely diverge: `SE3` stores 12 components for
        /// 6 degrees of freedom and `SO3` 9 for 3. A layout sized by `num_components` would build a 21-wide
        /// vector where a 10-wide one belongs, and every Jacobian on it would be wrong in a way no
        /// dimension check would catch. `retract_flat` steps only the quantities the layout names, and the
        /// mixed round trip is the one place `Tril`, `SO3` and `SE3` appear in a single layout.
        mod tangent_layout {
            #[test]
            fn layout_is_sized_by_dimension() {
                super::common::layout_is_sized_by_dimension_impl::<$B>();
            }

            #[test]
            fn mixed_layout_round_trip() {
                super::common::mixed_layout_round_trip_impl::<$B>();
            }

            #[test]
            fn retract_flat_is_partial() {
                super::common::retract_flat_is_partial_impl::<$B>($delta, $epsilon);
            }
        }
    };
}
