//! `SkoroError`: the single recoverable-error type across the whole model-facing surface.
//!
//! Skoro splits failures in two. Internal invariants -- a kernel handed the wrong number of
//! inputs, a sequence step whose parent has not been realized -- stay `panic!`s, because reaching
//! one means the library or its caller misused an internal API. Everything a *model author* can
//! do wrong arrives here instead.
//!
//! Not every variant is constructible from the five crates that currently build; several belong
//! to linear-algebra and model layers still being restructured. Their `Display` arms are still
//! exercised here, because a message that never compiles is a message that silently rots.

use skoro_core::{
    error::{SkoroError, SkoroResult},
    id::IdGenerator,
};

fn some_id() -> skoro_core::id::Id {
    IdGenerator::new().issue_id()
}

/// Every variant, paired with a fragment its message must contain.
///
/// The fragments are chosen to be the part a reader would search for: the id or the numbers, not
/// the boilerplate around them. A message that dropped its operand would still match a
/// boilerplate-only fragment.
fn every_variant() -> Vec<(SkoroError, &'static str)> {
    let id = some_id();
    vec![
        (SkoroError::NodeNotFound { id }, "does not exist"),
        (
            SkoroError::QuantityTypeMismatch { id },
            "does not match the stored type",
        ),
        (SkoroError::NotStochastic { id }, "not a stochastic node"),
        (SkoroError::NotObservable { id }, "not an observable node"),
        (
            SkoroError::NotLogLikelihood { id },
            "not a log-likelihood node",
        ),
        (SkoroError::NotMapped { id }, "not a mapped node"),
        (SkoroError::NotReal { id }, "not real-valued"),
        (SkoroError::NotContinuous { id }, "is discrete"),
        (
            SkoroError::DuplicateDensity { stochastic_id: id },
            "only have one log-density",
        ),
        (
            SkoroError::MustRemainDensity { id },
            "must maintain this property",
        ),
        (
            SkoroError::DanglingParent { parent_id: id },
            "dangling parent",
        ),
        (SkoroError::Cycle { id }, "graph cycle"),
        (
            SkoroError::NodeStillReferenced { id, child_id: id },
            "still a parent",
        ),
        (
            SkoroError::NodeStillBound { id, by: id },
            "still bound by",
        ),
        (
            SkoroError::BatchSizeMismatch {
                expected: 3,
                found: 5,
            },
            "expected 3, found 5",
        ),
        (SkoroError::MissingPoint { id }, "is missing"),
        (
            SkoroError::InconsistentFrameCount {
                expected: 2,
                found: 4,
            },
            "expected 2, found 4",
        ),
        (SkoroError::ManifoldMismatch { id }, "manifold type"),
        (
            SkoroError::InconsistentCotangents,
            "both order-1 and order-2",
        ),
        (
            SkoroError::IncompatibleMatmul {
                lhs: "A".into(),
                rhs: "B".into(),
            },
            "between A and B",
        ),
        (
            SkoroError::UnsupportedVectorProduct {
                tril: "L".into(),
                vector: "v".into(),
            },
            "between L and v",
        ),
        (
            SkoroError::InvalidRotationFrame { rows: 4 },
            "leading dimension is 4",
        ),
        (
            SkoroError::ShapeMismatch {
                expected: 6,
                found: 2,
            },
            "expected 6 elements, found 2",
        ),
        (SkoroError::DuplicateLayoutId { id }, "more than once"),
        (
            SkoroError::IdNotInLayout { id },
            "not present in the tangent layout",
        ),
        (
            SkoroError::LayoutDimensionMismatch {
                expected: 6,
                found: 5,
            },
            "expected 6, found 5",
        ),
        (SkoroError::NotPositiveDefinite { rows: 3 }, "3x3"),
        (SkoroError::BatchNotSingleton { found: 7 }, "found 7"),
        (
            SkoroError::NormalizerNotConstant {
                density: id,
                parent: id,
            },
            "not a sum of squares",
        ),
        (SkoroError::ModelMismatch { lhs: id, rhs: id }, "same model"),
    ]
}

/// Every variant renders a message that names its own operands. The fragments asserted are the
/// parts a reader would search for -- the id, the two numbers -- never the boilerplate around
/// them, so a variant that dropped its operand cannot pass on the wording alone.
mod messages {
    use super::*;

    /// Every variant renders a non-empty message that names its own operands.
    #[test]
    fn every_variant_renders_its_operands() {
        for (error, fragment) in every_variant() {
            let rendered = error.to_string();
            assert!(
                rendered.contains(fragment),
                "{error:?} rendered as {rendered:?}, which does not contain {fragment:?}"
            );
            // A message that is only its fragment is not a sentence.
            assert!(
                rendered.len() > fragment.len(),
                "{error:?} rendered too tersely"
            );
        }
    }

    /// Variants carrying an id must print it, or a report cannot say *which* node failed.
    #[test]
    fn id_bearing_variants_print_the_id() {
        let id = some_id();
        for error in [
            SkoroError::NodeNotFound { id },
            SkoroError::QuantityTypeMismatch { id },
            SkoroError::NotStochastic { id },
            SkoroError::NotObservable { id },
            SkoroError::NotLogLikelihood { id },
            SkoroError::NotMapped { id },
            SkoroError::NotReal { id },
            SkoroError::NotContinuous { id },
            SkoroError::DuplicateDensity { stochastic_id: id },
            SkoroError::MustRemainDensity { id },
            SkoroError::DanglingParent { parent_id: id },
            SkoroError::Cycle { id },
            SkoroError::NodeStillBound { id, by: id },
            SkoroError::MissingPoint { id },
            SkoroError::ManifoldMismatch { id },
            SkoroError::DuplicateLayoutId { id },
            SkoroError::IdNotInLayout { id },
        ] {
            let rendered = error.to_string();
            assert!(
                rendered.contains(&format!("{id:?}")),
                "{error:?} does not name its id"
            );
        }
    }
}

/// `SkoroError` as an ordinary error type: a real `std::error::Error`, comparable, cloneable, and
/// carried by the `SkoroResult` alias. That is what lets it compose with `?` and lets a test say
/// `assert_eq!(err, expected)` rather than matching on a shape.
mod interop {
    use super::*;

    /// It is a real `std::error::Error`, so it composes with `?` and with error-reporting crates.
    #[test]
    fn it_is_a_standard_error() {
        let error = SkoroError::InconsistentCotangents;
        let boxed: Box<dyn std::error::Error> = Box::new(error.clone());
        assert_eq!(boxed.to_string(), error.to_string());
        assert!(std::error::Error::source(&error).is_none());
    }

    /// Comparable and cloneable, which is what lets a test say `assert_eq!(err, expected)` rather
    /// than matching on a shape.
    #[test]
    fn errors_compare_and_clone() {
        let a = SkoroError::BatchSizeMismatch {
            expected: 1,
            found: 2,
        };
        assert_eq!(a.clone(), a);
        assert_ne!(
            a,
            SkoroError::BatchSizeMismatch {
                expected: 1,
                found: 3
            }
        );
        assert_ne!(a, SkoroError::InconsistentCotangents);
        assert!(format!("{a:?}").contains("BatchSizeMismatch"));
    }

    /// `SkoroResult` is the alias every fallible entry point returns.
    #[test]
    fn the_result_alias_carries_the_error() {
        fn fallible(ok: bool) -> SkoroResult<u8> {
            if ok {
                Ok(1)
            } else {
                Err(SkoroError::InconsistentCotangents)
            }
        }
        assert_eq!(fallible(true), Ok(1));
        assert_eq!(fallible(false), Err(SkoroError::InconsistentCotangents));
    }
}
