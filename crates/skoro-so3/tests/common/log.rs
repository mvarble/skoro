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
use skoro_so3::{ops::log::SO3LogOps, quantity::SO3};

use super::exp::{EXP_OMEGAS, OMEGAS, VS};

const LOG_EXP_OMEGAS: [[f64; 3]; 9] = [
    [-6.79251767e-01, -1.35850419e+00, -2.03775583e+00],
    [1.25242327e-16, -9.99999410e-02, 3.00000061e-01],
    [5.00000350e-01, 0.00000000e+00, 0.00000000e+00],
    [0.00000000e+00, 5.00000350e-01, 0.00000000e+00],
    [0.00000000e+00, 0.00000000e+00, 5.00000350e-01],
    [1.00000000e-08, 2.00000000e-08, 3.00000000e-08],
    [8.00000177e-01, 3.00000079e-01, -4.99999657e-01],
    [0.00000000e+00, 1.00000018e+00, 1.00000018e-09],
    [-9.99999887e-01, -9.99996097e-02, 2.99999923e-01],
];

#[rustfmt::skip]
const DLOG_R_VS: [[[f64; 3]; 2]; 9] = [
    [
        [0.09025834601386568, 0.6197890360675368, 1.4771947695423946],
        [-0.2623059353362468, -0.606255802590437, -0.5267628642351531],
    ],
    [
        [1.89190928276563, -0.9785354730298905, -0.5134732744057406],
        [1.5360119498722196, -0.7293449524048075, -1.0244464257071275],
    ],
    [
        [-1.4361158609390259, 1.4953108262564065, -0.3707453891491633],
        [1.0759468078613281, -0.3688862051856654, 0.5753297583699485],
    ],
    [
        [-0.8618510141788724, 0.172781303524971, -0.2734179035017294],
        [0.13439949962765843, -0.12487489730119705, 1.0533464597618922],
    ],
    [
        [-0.3742456593721333, 0.40757948563102425, 0.3010743260383606],
        [-0.3671380877027031, -0.2846720719469237, 0.6355776190757751],
    ],
    [
        [-1.4502359632391617, -0.20751250301223786, -0.34841121876825043],
        [-1.2514750125265093, 1.497495520025936, -1.3524465360890814],
    ],
    [
        [-0.5647930224435664, 0.32975856178228624, -0.7611932721895254],
        [-1.5915353289024063, -1.2198982578498434, 0.7918052918391472],
    ],
    [
        [0.7160506313855463, 0.9023516183865131, -1.5406391047575971],
        [-0.927759105083368, 0.30093958944204363, -0.8180618023633499],
    ],
    [
        [1.611534504205952, -2.7351831082389704, 1.1871865153562267],
        [-0.4142648291437742, -0.5415053329262828, 0.29825492171070617],
    ],
];

#[rustfmt::skip]
const DLOGT_R_VS: [[[f64; 3]; 2]; 9] = [
    [
        [0.841937528066478, 0.80874505833011, 1.1006643606831417],
        [-0.0663480362426189, -0.3355609209128095, -0.7725454183847807],
    ],
    [
        [1.4589112952806875, -1.4853894193522492, -0.6824245856505757],
        [1.143928512682628, -1.1347198054346481, -1.15957137336121],
    ],
    [
        [-1.4361158609390259, 1.1345138978506448, -1.0422498866772394],
        [1.0759468078613281, -0.047900321711315405, 0.6817528308868667],
    ],
    [
        [-0.6252618953405622, 0.172781303524971, -0.6531401708910269],
        [-0.38705453660815087, -0.12487489730119705, 0.9888330372131212],
    ],
    [
        [-0.13302745010547068, 0.5371075759888047, 0.3010743260383606],
        [-0.45867304499239886, -0.07380787075883621, 0.6355776190757751],
    ],
    [
        [-1.4502359624963115, -0.2075124629892728, -0.34841124569784465],
        [-1.251474940552714, 1.4974955440457187, -1.3524465760935358],
    ],
    [
        [-0.24611276173149882, -0.5679590970296806, -0.7899354641427248],
        [-1.3085279343462413, -1.087612495305331, 1.3239885905570792],
    ],
    [
        [1.6832869127291819, 0.9023516170757492, -0.22987503043203567],
        [0.18710488695493718, 0.30093958984666486, -1.2226830458402176],
    ],
    [
        [0.6710030351612528, -2.6997911953838805, -1.9361209528895253],
        [-0.5704634674342876, -0.43921517804196236, -0.1883104677871126],
    ],
];

#[allow(unused)]
pub fn log_forward<B: Backend>(
    log_forward: impl Fn(Tensor<B, 2>) -> Tensor<B, 2>,
    epsilon: B::FloatElem,
) where
    B::FloatElem: approx::AbsDiffEq<B::FloatElem, Epsilon = B::FloatElem> + Copy,
{
    let r = tensor::<B, 3>(EXP_OMEGAS, &Default::default()).reshape([9, 9]);
    let omega = log_forward(r);
    let omega = into_ndarray(omega);
    let omega_true = tensor::<B, 2>(LOG_EXP_OMEGAS, &Default::default());
    let omega_true = into_ndarray(omega_true);
    approx::assert_abs_diff_eq!(omega, omega_true, epsilon = epsilon);
}

#[inline(always)]
#[allow(unused)]
pub fn log_jvp<B: Backend>(
    log_jvp: impl Fn(Tensor<B, 2>, Tensor<B, 3>) -> Tensor<B, 3>,
    epsilon: B::FloatElem,
) where
    B::FloatElem: approx::AbsDiffEq<B::FloatElem, Epsilon = B::FloatElem> + Copy,
{
    let omega = tensor::<B, 2>(OMEGAS, &Default::default());
    let v = tensor::<B, 3>(VS, &Default::default()).transpose();
    let dlogomegav = log_jvp(omega, v);
    let dlogomegav = into_ndarray(dlogomegav);
    let dlogomegav_true = tensor::<B, 3>(DLOG_R_VS, &Default::default()).transpose();
    let dlogomegav_true = into_ndarray(dlogomegav_true);
    approx::assert_abs_diff_eq!(dlogomegav, dlogomegav_true, epsilon = epsilon);
}

#[inline(always)]
#[allow(unused)]
pub fn log_vjp<B: Backend>(
    log_vjp: impl Fn(Tensor<B, 2>, Tensor<B, 3>) -> Tensor<B, 3>,
    epsilon: B::FloatElem,
) where
    B::FloatElem: approx::AbsDiffEq<B::FloatElem, Epsilon = B::FloatElem> + Copy,
{
    let omega = tensor::<B, 2>(OMEGAS, &Default::default());
    let v = tensor::<B, 3>(VS, &Default::default()).transpose();
    let dlogtomegav = log_vjp(omega, v);
    let dlogtomegav = into_ndarray(dlogtomegav);
    let dlogtomegav_true = tensor::<B, 3>(DLOGT_R_VS, &Default::default()).transpose();
    let dlogtomegav_true = into_ndarray(dlogtomegav_true);
    approx::assert_abs_diff_eq!(dlogtomegav, dlogtomegav_true, epsilon = epsilon);
}

#[inline(always)]
#[allow(unused)]
pub fn log<B: Backend>(delta: B::FloatElem, epsilon: B::FloatElem)
where
    SO3: ContinuousQuantity<B>,
    Q<B, SO3>: SO3LogOps<B, Euclidean<3>>,
    B::FloatElem: approx::AbsDiffEq<B::FloatElem, Epsilon = B::FloatElem> + Copy,
{
    let model = ModelClient::<B>::default();
    let r = model.add_parameter(SO3);
    let omega = r.log().unwrap();
    let point = points()
        .with_tensor(
            r.id(),
            tensor::<B, 3>(EXP_OMEGAS, &Default::default()).reshape([9, 9]),
        )
        .build();

    test_vjp_numerically(&r, &omega, point.clone(), delta, epsilon);
    test_jvp_numerically(&r, &omega, point.clone(), delta, epsilon);
    test_hvp_numerically(&r, &omega, point.clone(), delta, epsilon);

    let omega = omega.evaluate(&point).unwrap();
    let omega = into_ndarray(omega);
    let omega_true = tensor::<B, 2>(LOG_EXP_OMEGAS, &Default::default());
    let omega_true = into_ndarray(omega_true);
    approx::assert_abs_diff_eq!(omega, omega_true, epsilon = epsilon);
}

#[macro_export]
macro_rules! test_log {
    (($log_forward:path, $eps_forward:literal), ($log_jvp:path, $eps_jvp:literal), ($log_vjp:path, $eps_vjp:literal), ($del:literal, $eps:literal)$(,)?) => {
        use super::B;
        use super::common::log;

        #[test]
        pub fn log_forward() {
            log::log_forward::<B>($log_forward, $eps_forward);
        }

        #[test]
        pub fn log_jvp() {
            log::log_jvp::<B>($log_jvp, $eps_jvp);
        }

        #[test]
        pub fn log_vjp() {
            log::log_vjp::<B>($log_vjp, $eps_vjp);
        }

        #[test]
        pub fn log() {
            log::log::<B>($del, $eps);
        }
    };
}
