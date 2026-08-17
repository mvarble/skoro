//! The certificate is only as good as the fact that nothing can populate it dishonestly.
//!
//! `NlsStructure` is deliberately open — it must be, since the Gaussian builder that
//! implements it lives in a different crate and a sealed trait cannot be implemented
//! outside its own. So the guarantee is not on the trait. It is on
//! [`ResidualCertificate::record`], which demands a `WhitenedResidual` distribution and has
//! no unbounded sibling.
//!
//! Two halves, and both are needed:
//!
//! - a certificate anyone can construct is **empty** — checked here at runtime;
//! - `record` **cannot be reached** without the bound — checked by `trybuild`, because a
//!   test that compiles cannot assert that something else does not.
//!
//! A third case lives in the same directory and guards the same evidence from the other side:
//! `remove_closure` must stay off `ModelClientOps`, or an estimator could delete the nodes a
//! certificate names instead of adding to it.
//!
//! **This is the gate on the whole design.** Do *not* replace the compile-fail cases with a
//! prose comment: if `record` ever grows an unbounded sibling, this is the only thing that
//! will notice.

use skoro_model::ResidualCertificate;

/// A hand-built certificate holds no factors, so a hand-rolled `NlsStructure` implementor
/// can only claim an empty model — never a fabricated one.
#[test]
fn a_certificate_built_by_hand_is_empty() {
    let certificate = ResidualCertificate::new();

    assert!(certificate.factors().is_empty());
    assert_eq!(certificate.residual_ids().count(), 0);
    assert_eq!(certificate, ResidualCertificate::default());
}

/// Every negative claim in the design, as a case that must fail to compile.
///
/// Two families live here. The `record*` cases pin that a certificate cannot be populated
/// without the bound. The `remove_closure` case pins the other side of the same guarantee —
/// that an `NlsStructure` consumer cannot *un*-populate one by deleting the nodes it names.
///
/// The `.stderr` fixtures are regenerated with `TRYBUILD=overwrite cargo test -p skoro-model
/// --test certificate`. They are expected to need updating when the compiler's wording
/// changes; what must not change is that the cases fail to compile at all.
#[test]
fn record_cannot_be_reached_without_a_whitened_residual() {
    trybuild::TestCases::new().compile_fail("tests/compile_fail/*.rs");
}
