use skoro_backend::{Backend, Tensor};
use skoro_graph::quantity::ContinuousQuantity;
use skoro_model::{
    hash_map_builders::points,
    model::ModelClient,
    q::Q,
    test_utils::{
        into_ndarray, tensor, test_hvp_numerically, test_jvp_numerically, test_vjp_numerically,
    },
};
use skoro_riemann::euclidean::Euclidean;
use skoro_so3::{ops::exp::SO3ExpOps, quantity::SO3};

pub const OMEGAS: [[f32; 3]; 9] = [
    [1.0, 2.0, 3.0],
    [0.0, -0.1, 0.3],
    [0.5, 0.0, 0.0],
    [0.0, 0.5, 0.0],
    [0.0, 0.0, 0.5],
    [1.0e-8, 2.0e-8, 3.0e-8],
    [0.8, 0.3, -0.5],
    [0.0, 1.0, 1e-9],
    [-1.0, -0.1, 0.3],
];

pub const VS: [[[f32; 3]; 2]; 9] = [
    [
        [0.31724943, 1.0110292, 1.1407043],
        [-0.3113397, -0.37689686, -0.66332422],
    ],
    [
        [1.6895131, -1.2428055, -0.6015633],
        [1.3512494, -0.94185084, -1.0952817],
    ],
    [
        [-1.4361159, 1.343009, -0.72159388],
        [1.0759468, -0.21284614, 0.64197176],
    ],
    [
        [-0.75944451, 0.1727813, -0.47317824],
        [-0.12902685, -0.1248749, 1.0429081],
    ],
    [
        [-0.25905617, 0.48243641, 0.30107433],
        [-0.4217284, -0.18306992, 0.6355776],
    ],
    [
        [-1.450236, -0.20751249, -0.34841124],
        [-1.251475, 1.4974955, -1.3524465],
    ],
    [
        [-0.44421393, -0.13065249, -0.84451336],
        [-1.4309004, -1.2018168, 1.05967],
    ],
    [
        [1.3107641, 0.9023516, -0.9672363],
        [-0.40462123, 0.3009396, -1.114864],
    ],
    [
        [1.1668849, -3.006619, -0.3854574],
        [-0.4908385, -0.53564923, 0.0449614],
    ],
];

pub const EXP_OMEGAS: [[[f32; 3]; 3]; 9] = [
    [
        [-0.694921, 0.713521, 0.0892929],
        [-0.192007, -0.303785, 0.933192],
        [0.692978, 0.63135, 0.348107],
    ],
    [
        [0.950415, -0.295025, -0.0983416],
        [0.295025, 0.955374, -0.0148754],
        [0.0983416, -0.0148754, 0.995042],
    ],
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.877583, -0.479426],
        [0.0, 0.479426, 0.877583],
    ],
    [
        [0.877583, 0.0, 0.479426],
        [0.0, 1.0, 0.0],
        [-0.479426, 0.0, 0.877583],
    ],
    [
        [0.877583, -0.479426, 0.0],
        [0.479426, 0.877583, 0.0],
        [0.0, 0.0, 1.0],
    ],
    [
        [1.0, -3e-08, 2e-08],
        [3e-08, 1.0, -1e-08],
        [-2e-08, 1e-08, 1.0],
    ],
    [
        [0.843438, 0.532757, 0.0691547],
        [-0.311728, 0.590175, -0.74466],
        [-0.437537, 0.606517, 0.663851],
    ],
    [
        [0.540302, -8.41471e-10, 0.841471],
        [8.41471e-10, 1.0, 4.59698e-10],
        [-0.841471, 4.59698e-10, 0.540302],
    ],
    [
        [0.954419, -0.202366, -0.219393],
        [0.293528, 0.503162, 0.812815],
        [-0.0540954, -0.840164, 0.539627],
    ],
];

#[rustfmt::skip]
pub const DEXP_OMEGA_VS: [[[f32; 3]; 2]; 9] = [
    [
        [0.523762, 0.819327, 1.19967],
        [-0.17862, -0.410138, -0.685404],
    ],
    [
        [1.44679, -1.47259, -0.678157],
        [1.13443, -1.12335, -1.15578],
    ],
    [
        [-1.43612, 1.11107, -1.02072],
        [1.07595, -0.0469107, 0.667667],
    ],
    [
        [-0.612344, 0.172781, -0.639646],
        [-0.379058, -0.124875, 0.968403],
    ],
    [
        [-0.130279, 0.526011, 0.301074],
        [-0.449197, -0.072283, 0.635578],
    ],
    [
        [-1.45024, -0.207512, -0.348411],
        [-1.25147, 1.4975, -1.35245],
    ],
    [
        [-0.224872, -0.522395, -0.728612],
        [-1.33641, -1.05089, 1.30141],
    ],
    [
        [1.54761, 0.902352, -0.211346],
        [0.172023, 0.30094, -1.12413],
    ],
    [
        [0.690586, -2.45332, -1.78869],
        [-0.564871, -0.404883, -0.158224],
    ],
];

#[rustfmt::skip]
pub const DEXPT_OMEGA_VS: [[[f32; 3]; 2]; 9] = [
    [
        [0.327756, 0.770056, 1.29785],
        [-0.229717, -0.480723, -0.621314],
    ],
    [
        [1.8762, -0.969941, -0.510609],
        [1.52325, -0.72134, -1.02178],
    ],
    [
        [-1.43612, 1.46442, -0.363086],
        [1.07595, -0.361265, 0.563443],
    ],
    [
        [-0.844045, 0.172781, -0.267769],
        [0.131623, -0.124875, 1.03158],
    ],
    [
        [-0.366514, 0.399159, 0.301074],
        [-0.359553, -0.278791, 0.635578],
    ],
    [
        [-1.45024, -0.207513, -0.348411],
        [-1.25148, 1.4975, -1.35245],
    ],
    [
        [-0.518362, 0.304363, -0.702141],
        [-1.59705, -1.17272, 0.811291],
    ],
    [
        [0.658334, 0.902352, -1.41646],
        [-0.852977, 0.30094, -0.752122],
    ],
    [
        [1.548, -2.48558, 1.05861],
        [-0.422475, -0.498134, 0.285344],
    ],
];

pub const EXP_OMEGA_GRADS: [[f32; 3]; 9] = [
    [0.211952, 1.09634, 1.19846],
    [0.785078, 1.12885, 1.04295],
    [1.0, 0.714016, 1.20369],
    [1.20369, 1.0, 0.714016],
    [0.714016, 1.20369, 1.0],
    [1.0, 1.0, 1.0],
    [1.28904, 0.274429, 1.02712],
    [1.30117, 1.0, 0.381773],
    [0.770353, 1.43167, 0.3784],
];

pub const ALPHAS: [[[f32; 3]; 2]; 9] = [
    [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    [[0.0, 0.0, -1.0], [0.0, -1.0, 0.0]],
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
    [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
    [[0.0, -0.8, 0.0], [0.0, 0.0, 1.0]],
    [[0.1, -0.8, 0.5], [0.0, -1.2, 0.0]],
    [[1.0, 0.0, 0.0], [0.0, 0.1, 0.0]],
    [[0.0, -1.0, 0.0], [0.8, -0.1, 0.0]],
    [[0.0, 0.0, -1.0], [-0.6, 0.0, -0.7]],
];

#[inline(always)]
#[allow(unused)]
pub fn exp_forward<B: Backend>(exp_forward: impl Fn(Tensor<B, 2>) -> Tensor<B, 2>, epsilon: f64) {
    let omega = tensor::<B, 2>(OMEGAS, &Default::default());
    let expomega = exp_forward(omega);
    let expomega = into_ndarray(expomega);
    let expomega_true = tensor::<B, 3>(EXP_OMEGAS, &Default::default()).reshape([9, 9]);
    let expomega_true = into_ndarray(expomega_true);
    approx::assert_abs_diff_eq!(expomega, expomega_true, epsilon = epsilon);
}

#[inline(always)]
#[allow(unused)]
pub fn exp_jvp<B: Backend>(
    exp_jvp: impl Fn(Tensor<B, 2>, Tensor<B, 3>) -> Tensor<B, 3>,
    epsilon: f64,
) {
    let omega = tensor::<B, 2>(OMEGAS, &Default::default());
    let v = tensor::<B, 3>(VS, &Default::default()).transpose();
    let dexpomegav = exp_jvp(omega, v);
    let dexpomegav = into_ndarray(dexpomegav);
    let dexpomegav_true = tensor::<B, 3>(DEXP_OMEGA_VS, &Default::default()).transpose();
    let dexpomegav_true = into_ndarray(dexpomegav_true);
    approx::assert_abs_diff_eq!(dexpomegav, dexpomegav_true, epsilon = epsilon);
}

#[inline(always)]
#[allow(unused)]
pub fn exp_vjp<B: Backend>(
    exp_vjp: impl Fn(Tensor<B, 2>, Tensor<B, 3>) -> Tensor<B, 3>,
    epsilon: f64,
) {
    let omega = tensor::<B, 2>(OMEGAS, &Default::default());
    let v = tensor::<B, 3>(VS, &Default::default()).transpose();
    let dexptomegav = exp_vjp(omega, v);
    let dexptomegav = into_ndarray(dexptomegav);
    let dexptomegav_true = tensor::<B, 3>(DEXPT_OMEGA_VS, &Default::default()).transpose();
    let dexptomegav_true = into_ndarray(dexptomegav_true);
    approx::assert_abs_diff_eq!(dexptomegav, dexptomegav_true, epsilon = epsilon);
}

#[inline(always)]
#[allow(unused)]
pub fn exp<B: Backend>(delta: f64, epsilon: f64)
where
    SO3: ContinuousQuantity<B>,
    Q<B, Euclidean<3>>: SO3ExpOps<B>,
{
    let model = ModelClient::<B>::default();
    let omega = model.add_parameter(Euclidean::<3>);
    let expomega = omega.exp().unwrap();
    let point = points()
        .with_quantity(&omega, tensor::<B, 2>(OMEGAS, &Default::default()))
        .build();

    test_vjp_numerically(&omega, &expomega, point.clone(), delta, epsilon);
    test_jvp_numerically(&omega, &expomega, point.clone(), delta, epsilon);
    test_hvp_numerically(&omega, &expomega, point.clone(), delta, epsilon);

    let expomega = expomega.evaluate(&point).unwrap();
    let expomega = into_ndarray(expomega);
    let expomega_true = tensor::<B, 3>(EXP_OMEGAS, &Default::default()).reshape([9, 9]);
    let expomega_true = into_ndarray(expomega_true);
    approx::assert_abs_diff_eq!(expomega, expomega_true, epsilon = epsilon);
}

#[macro_export]
macro_rules! test_exp {
    (($exp_forward:path, $eps_forward:literal), ($exp_jvp:path, $eps_jvp:literal), ($exp_vjp:path, $eps_vjp:literal), ($del:literal, $eps:literal)$(,)?) => {
        use super::B;
        use super::common::exp;

        #[test]
        pub fn exp_forward() {
            exp::exp_forward::<B>($exp_forward, $eps_forward);
        }

        #[test]
        pub fn exp_jvp() {
            exp::exp_jvp::<B>($exp_jvp, $eps_jvp);
        }

        #[test]
        pub fn exp_vjp() {
            exp::exp_vjp::<B>($exp_vjp, $eps_vjp);
        }

        #[test]
        pub fn exp() {
            exp::exp::<B>($del, $eps);
        }
    };
}
