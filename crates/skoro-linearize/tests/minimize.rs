#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! The Levenberg–Marquardt method, apart from the examples, on the crate's own models.

mod common;

use common::{close, host, rotations};
use skoro_core::NodeId;
use skoro_linearize::{Assembly, Linearizer};
use skoro_tensor::Target;

#[test]
fn the_objective_is_the_negative_log_density() {
    let target = Target::host();
    let (writer, query, point) = rotations(&target);
    let snapshot = writer.snapshot();
    let linearizer = Linearizer::new(&snapshot, &query, &target, Assembly::Residual).unwrap();
    let objective = linearizer.objective(&point).unwrap();
    assert_eq!(objective.len(), 2);
    let program = snapshot.compile(&query, &target).unwrap();
    let density: Vec<f64> = host(&program.forward(&point).unwrap(), NodeId::DENSITY)
        .into_iter()
        .map(|l| -l)
        .collect();
    close(&objective, &density, 1e-14, "the query's own program");
    let zero = vec![0.0; linearizer.layout().dimension()];
    let at_zero: Vec<f64> = linearizer
        .at(&point)
        .unwrap()
        .problems()
        .iter()
        .map(|p| p.value(&zero))
        .collect();
    close(&objective, &at_zero, 1e-12, "the linearization at zero");
}
