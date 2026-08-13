//! Recording a factor requires a distribution that has a residual. A bare type does not,
//! and no amount of having the right ids to hand substitutes for it.

use skoro_backend::NdArrayBackend;
use skoro_model::{ResidualCertificate, model::ModelClient};
use skoro_riemann::euclidean::EuclideanD;

/// Implements nothing at all — in particular not `WhitenedResidual`.
struct NotAWhitenedResidual;

fn main() {
    let client = ModelClient::<NdArrayBackend<f64, i32>>::default();
    // A perfectly real residual-shaped quantity, obtained the ordinary way.
    let residual = client.add_parameter(EuclideanD(2));

    let mut certificate = ResidualCertificate::new();
    certificate
        .record(&NotAWhitenedResidual, residual.id(), &residual)
        .unwrap();
}
