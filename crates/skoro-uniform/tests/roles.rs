#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! The roles a uniform's term binds: a residual of no components, and nothing else.

use skoro_core::scope::quantity;
use skoro_core::{AtMostOne, Boolean, Error, Role, Space, Writer};
use skoro_normal::{Form, roles};
use skoro_spatial::Rotation;
use skoro_uniform::Uniform;

#[test]
fn binds_an_empty_residual_and_nothing_else() {
    let mut writer = Writer::new();
    let terms = writer
        .build(|| {
            let heading = quantity("heading", Rotation);
            let outlier = quantity("outlier", Boolean);
            [
                Uniform::on(Rotation).score(heading).id(),
                Uniform::on(Boolean).score(outlier).id(),
            ]
        })
        .unwrap();
    let snapshot = writer.snapshot();
    for term in terms {
        let residual = snapshot.bound(term, &roles::RESIDUAL).unwrap();
        assert_eq!(snapshot.space(residual), Some(&Space::Vector(0)));
        // A constant: reached from no quantity, and so of the term.
        assert!(snapshot.node(residual).unwrap().is_fixed());
        assert_eq!(
            snapshot.bound(term, &roles::MEAN).err(),
            Some(Error::Binding {
                role: "skoro.normal.mean",
                count: 0
            })
        );
        let unbound = |role: &Role<AtMostOne>| snapshot.bound(term, role).unwrap().is_none();
        for form in [
            Form::Isotropic,
            Form::Diagonal,
            Form::CovarianceFactor,
            Form::Whitening,
        ] {
            assert!(unbound(roles::scale(form)), "{form:?}");
        }
        assert!(unbound(&roles::TANGENT));
        assert!(unbound(&roles::LOG_VOLUME));
        assert!(unbound(&roles::INFORMATION_ROOT));
        // The sampler is exact, and reads nothing.
        let sampler = snapshot.term(term).unwrap().sampler.clone().unwrap();
        assert_eq!(sampler.sample.name(), "skoro.uniform");
        assert!(sampler.inputs.is_empty() && sampler.log_density.is_none());
    }
}
