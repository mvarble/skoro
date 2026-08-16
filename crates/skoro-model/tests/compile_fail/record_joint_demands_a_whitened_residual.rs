//! The joint sibling is bounded too. `record_joint` widens the *shape* of factor a certificate
//! admits — a joint over several existing quantities, which no `Distribution` can describe — and
//! not the evidence it demands. A bare type still cannot get an id into a certificate.

use skoro_backend::NdArrayBackend;
use skoro_model::{ResidualCertificate, model::ModelClient};
use skoro_riemann::euclidean::EuclideanD;

/// Implements nothing at all — in particular not `JointWhitenedResidual`.
struct NotAJointWhitenedResidual;

fn main() {
    let client = ModelClient::<NdArrayBackend<f64, i32>>::default();
    // A perfectly real residual-shaped quantity, obtained the ordinary way.
    let residual = client.add_parameter(EuclideanD(2));

    let mut certificate = ResidualCertificate::new();
    certificate
        .record_joint(&NotAJointWhitenedResidual, residual.id(), &residual)
        .unwrap();
}
