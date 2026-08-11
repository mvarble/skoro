use skoro_model::ModelBuilderOps;
use std::collections::HashMap;

use skoro_backend::{NdArrayBackend, Point, TangentVector, Tensor, TensorData};
use skoro_linalg::tril::Tril;
use skoro_model::model::ModelClient;
use skoro_operation::Input;
use skoro_riemann::euclidean::{Euclidean, EuclideanD, VectorD};

use skoro_normal::distribution::{Normal, NormalSampler};

pub type B = NdArrayBackend<f32, i32>;

#[test]
fn sample_statistics() {
    let sample_count = 1000;
    let mean = Tensor::<B, 1>::from_floats([1.0, 2.0, 3.0], &Default::default());
    let mean_repeat = mean
        .clone()
        .unsqueeze_dim::<2>(0)
        .repeat_dim(0, sample_count);
    let std_dev = Tensor::<B, 1>::from_floats([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &Default::default());
    let std_dev_repeat = std_dev
        .clone()
        .unsqueeze_dim::<2>(0)
        .repeat_dim(0, sample_count);
    let samples = NormalSampler::new(Euclidean::<3>).sample(mean_repeat.clone(), std_dev_repeat);

    let sample_mean = samples.clone().mean_dim(0).squeeze_dim::<1>(0);
    let [mx, my, mz] = sample_mean
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .try_into()
        .unwrap();

    let zscore = 5.0;
    let sc = sample_count as f32;

    approx::assert_abs_diff_eq!(mx, 1.0, epsilon = zscore * (1.0 / sc).sqrt());
    approx::assert_abs_diff_eq!(my, 2.0, epsilon = zscore * (13.0 / sc).sqrt());
    approx::assert_abs_diff_eq!(mz, 3.0, epsilon = zscore * (77.0 / sc).sqrt());

    let sample_diffs = (samples.clone() - mean_repeat).unsqueeze_dim::<3>(2);
    let sample_cov = sample_diffs
        .clone()
        .matmul(sample_diffs.transpose())
        .mean_dim(0)
        .squeeze_dim::<2>(0);

    let [cxx, cxy, cxz, _, cyy, cyz, _, _, czz] = sample_cov
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .try_into()
        .unwrap();

    approx::assert_abs_diff_eq!(
        cxx,
        1.0,
        epsilon = zscore * ((1.0 * 1.0 + 1.0f32.powi(2)) / sc).sqrt(),
    );
    approx::assert_abs_diff_eq!(
        cxy,
        2.0,
        epsilon = zscore * ((1.0 * 13.0 + 2.0f32.powi(2)) / sc).sqrt(),
    );
    approx::assert_abs_diff_eq!(
        cxz,
        4.0,
        epsilon = zscore * ((1.0 * 77.0 + 4.0f32.powi(2)) / sc).sqrt(),
    );
    approx::assert_abs_diff_eq!(
        cyy,
        13.0,
        epsilon = zscore * ((13.0 * 13.0 + 13.0f32.powi(2)) / sc).sqrt(),
    );
    approx::assert_abs_diff_eq!(
        cyz,
        23.0,
        epsilon = zscore * ((13.0 * 77.0 + 23.0f32.powi(2)) / sc).sqrt(),
    );
    approx::assert_abs_diff_eq!(
        czz,
        77.0,
        epsilon = zscore * ((77.0 * 77.0 + 77.0f32.powi(2)) / sc).sqrt(),
    );
}

#[test]
fn log_density() {
    let model = ModelClient::<B>::new(Default::default());
    let mean = model.add_constant(EuclideanD(3), &VectorD::new(&[10.0, 20.0, 30.0]));
    let std_dev = model.add_constant(Tril(3), &VectorD::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    let dist = Normal::new(mean, std_dev);
    let x = model.add_quantity(dist).unwrap();
    let points = {
        let mut out = HashMap::new();
        out.insert(
            x.id(),
            Input::float(Point::from_data(
                TensorData::new(vec![11.0, 19.0, 30.5], [1, 3]),
                &Default::default(),
            )),
        );
        out
    };
    let tangents = {
        let mut out = HashMap::new();
        out.insert(
            x.id(),
            TangentVector::from_data(
                TensorData::new(vec![162.0, 0.0, 0.0], [1, 3]),
                &Default::default(),
            ),
        );
        out
    };
    let eval = model.log_density_ord2(&points, &tangents).unwrap();
    let [log_density] = eval
        .value
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .try_into()
        .unwrap();
    let [dldx, dldy, dldz] = eval.ord1_cotangents[&x.id()]
        .to_data()
        .to_vec::<f32>()
        .unwrap()
        .try_into()
        .unwrap();
    let [hlxx, hlxy, hlxz] = eval.ord2_cotangents[&x.id()]
        .to_data()
        .to_vec::<f32>()
        .unwrap()
        .try_into()
        .unwrap();

    assert_eq!(
        log_density,
        -1.5 * std::f32::consts::TAU.ln() - 18.0f32.ln() - 33.0 / 32.0
    );
    approx::assert_abs_diff_eq!(dldx, -59.0 / 36.0, epsilon = 1e-6);
    approx::assert_abs_diff_eq!(dldy, 29.0 / 72.0, epsilon = 1e-6);
    approx::assert_abs_diff_eq!(dldz, -1.0 / 24.0, epsilon = 1e-6);

    approx::assert_abs_diff_eq!(hlxx, -236.0, epsilon = 1e-6);
    approx::assert_abs_diff_eq!(hlxy, 31.0, epsilon = 1e-6);
    approx::assert_abs_diff_eq!(hlxz, 3.0, epsilon = 1e-6);
}
