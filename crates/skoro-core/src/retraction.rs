//! The retractions of the core's spaces, their inverses, and their charts' log-volumes and
//! domains, as composites cached per space.
//!
//! | space | `x ⊕ t` | `y ⊖ x` |
//! | --- | --- | --- |
//! | `Real`, `Vector`, `Matrix` | `x + reshape(t)` | `reshape(y - x)` |
//! | `Positive` | `positive.mul(x, positive.exp(t))` | `positive.log(y) - positive.log(x)` |
//! | `Product` | each factor's, on the slices of `x` and `t` by `blocks()`, concatenated | the same |
//!
//! | space | `λ_x(t)` | `t ∈ U_x` |
//! | --- | --- | --- |
//! | `Real`, `Vector`, `Matrix` | `0` | global |
//! | `Positive` | `sum(positive.log(x) + t)` | global |
//! | `Product` | the factors', on the slices of `x` and `t`, summed | global if every factor's is; otherwise `is_nonnegative(Σ to_real(Ωᵢ) - k)` over the `k` factors that are not |
//! | `Embedded`, with a volume | `Λ(include(x), t)` | the declaration's, `Ω(include(x), t)` |
//!
//! The log-volumes are with respect to each space's reference measure: Lebesgue measure on the
//! Euclidean and positive spaces, the measure an embedded type's retraction normalizes, and the
//! product of the factors' on a product. An embedded type that declares no volume has none, and
//! neither has a product with it among its factors (`NoVolume`).
//!
//! The tangent of every space is a `Vector` of the space's dimension.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use crate::primitive::{
    Concatenate, Include, IsNonnegative, PositiveExp, PositiveLog, PositiveMul, Retract, Slice,
    ToReal,
};
use crate::scope::{apply, constant};
use crate::{Composite, Domain, Error, Param, QDyn, Real, Space};

/// Which of a space's maps.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Map {
    Retraction,
    Inverse,
    LogVolume,
    Domain,
}

type Cache = Mutex<HashMap<(Space, Map), Arc<Composite>>>;

static CACHE: OnceLock<Cache> = OnceLock::new();

impl Space {
    /// The composite of `point ⊕ tangent`, of inputs this space and `Vector(dimension)`; an
    /// error for a discrete space (`NotContinuous`).
    pub fn retraction(&self) -> Result<Arc<Composite>, Error> {
        cached(self, Map::Retraction)
    }

    /// The composite of `to ⊖ from`, of inputs this space twice, to `Vector(dimension)`; an
    /// error for a discrete space (`NotContinuous`).
    pub fn inverse_retraction(&self) -> Result<Arc<Composite>, Error> {
        cached(self, Map::Inverse)
    }

    /// The composite of `λ_x(t)`, the log-volume of the chart at `x`, of inputs this space and
    /// `Vector(dimension)`, to `Real` (`NotContinuous` for a discrete space, `NoVolume` for one
    /// without a measure).
    pub fn log_volume(&self) -> Result<Arc<Composite>, Error> {
        self.measured()?;
        cached(self, Map::LogVolume)
    }

    /// Whether every chart of this space is global, and otherwise the composite of `t ∈ U_x`, of
    /// inputs this space and `Vector(dimension)`, to `Boolean` (errors as `log_volume`'s).
    pub fn domain(&self) -> Result<Domain, Error> {
        self.measured()?;
        if self.is_global() {
            Ok(Domain::Global)
        } else {
            cached(self, Map::Domain).map(Domain::Within)
        }
    }

    /// Whether `λ_x(t)` depends on the point `x`: false on the Euclidean spaces, whose log-volume
    /// is zero, true on `Positive`, the factors' on a product (true if any factor's is), and on
    /// an embedded type whether its declared `Λ` reads its first input (`NotContinuous` for a
    /// discrete space, `NoVolume` for one without a measure).
    pub fn volume_reads_point(&self) -> Result<bool, Error> {
        self.measured()?;
        Ok(match self {
            Space::Real | Space::Vector(_) | Space::Matrix(..) => false,
            Space::Positive(_) => true,
            Space::Product(factors) => {
                let mut reads = false;
                for factor in factors.iter() {
                    reads |= factor.volume_reads_point()?;
                }
                reads
            }
            Space::Embedded(declaration) => match declaration.volume() {
                Some(volume) => volume.log_volume().reads(0),
                None => unreachable!("measured() refuses a type without a volume"),
            },
            Space::Categorical(_) | Space::Integer => {
                unreachable!("measured() refuses a discrete space")
            }
        })
    }

    /// Whether the space has a reference measure: false only for an embedded type that declares
    /// no volume, and for a product with such a factor. A discrete space has counting measure.
    pub fn has_measure(&self) -> bool {
        match self {
            Space::Embedded(declaration) => declaration.volume().is_some(),
            Space::Product(factors) => factors.iter().all(Space::has_measure),
            _ => true,
        }
    }

    /// Refuses a discrete space (`NotContinuous`) and one without a measure (`NoVolume`).
    fn measured(&self) -> Result<(), Error> {
        if !self.is_continuous() {
            return Err(Error::NotContinuous {
                space: self.clone(),
            });
        }
        if !self.has_measure() {
            return Err(Error::NoVolume {
                space: self.clone(),
                quantity: None,
            });
        }
        Ok(())
    }

    /// Whether every chart is global, for a continuous space with a measure.
    fn is_global(&self) -> bool {
        match self {
            Space::Embedded(declaration) => {
                matches!(
                    declaration.volume().map(|v| v.domain()),
                    Some(Domain::Global)
                )
            }
            Space::Product(factors) => factors.iter().all(Space::is_global),
            _ => true,
        }
    }
}

fn cached(space: &Space, map: Map) -> Result<Arc<Composite>, Error> {
    let cache = CACHE.get_or_init(Cache::default);
    let key = (space.clone(), map);
    if let Some(composite) = cache
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get(&key)
    {
        return Ok(composite.clone());
    }
    // Defined with the lock released: a product's retraction asks for its factors' as it is
    // defined. Two threads may define one at once, and define the same composite.
    let composite = define(space, map)?;
    cache
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .insert(key, composite.clone());
    Ok(composite)
}

fn define(space: &Space, map: Map) -> Result<Arc<Composite>, Error> {
    let Some(dimension) = space.dimension() else {
        return Err(Error::NotContinuous {
            space: space.clone(),
        });
    };
    let tangent = Space::Vector(dimension);
    let params = vec![Param::Space(space.clone())];
    let defined = match map {
        Map::Retraction => {
            Composite::define("skoro.retraction", params, &[space.clone(), tangent], |q| {
                retract(space, q[0], q[1])
            })
        }
        Map::Inverse => Composite::define(
            "skoro.inverse_retraction",
            params,
            &[space.clone(), space.clone()],
            |q| local(space, q[0], q[1]),
        ),
        Map::LogVolume => {
            Composite::define("skoro.log_volume", params, &[space.clone(), tangent], |q| {
                log_volume(space, q[0], q[1])
            })
        }
        Map::Domain => Composite::define("skoro.domain", params, &[space.clone(), tangent], |q| {
            domain(space, q[0], q[1])
        }),
    };
    defined.map_err(|failure| failure.error)
}

/// `t`, a `Vector` of the same storage, as a value of `space`, or as it is if it is one.
fn reshape(t: QDyn, from: &Space, to: &Space) -> QDyn {
    if from == to {
        t
    } else {
        apply(Arc::new(Concatenate::new(to.clone())), &[t])
    }
}

/// `x ⊕ t` on a space, built in the open definition.
fn retract(space: &Space, x: QDyn, t: QDyn) -> QDyn {
    let tangent = Space::Vector(space.dimension().unwrap_or(0));
    match space {
        Space::Real | Space::Vector(_) | Space::Matrix(_, _) => x + reshape(t, &tangent, space),
        Space::Positive(_) => {
            let e = apply(Arc::new(PositiveExp), &[t]);
            apply(Arc::new(PositiveMul), &[x, e])
        }
        Space::Product(_) => {
            let factors: Vec<QDyn> = space
                .blocks()
                .into_iter()
                .map(|block| {
                    let d = block.tangent.len();
                    let xi = Slice::new(block.space.clone(), block.storage, block.tangent.clone());
                    let ti = Slice::new(Space::Vector(d), block.tangent.clone(), block.tangent);
                    let (xi, ti) = (apply(Arc::new(xi), &[x]), apply(Arc::new(ti), &[t]));
                    match block.space.retraction() {
                        Ok(retraction) => retraction.call(&[xi, ti]),
                        Err(_) => unreachable!("a product's factors are continuous"),
                    }
                })
                .collect();
            apply(Arc::new(Concatenate::new(space.clone())), &factors)
        }
        // One node: the boundary operation, whose kernel runs the declaration's R.
        Space::Embedded(declaration) => apply(Arc::new(Retract::new(declaration.clone())), &[x, t]),
        Space::Categorical(_) | Space::Integer => {
            unreachable!("a discrete space has no retraction, refused before it is defined")
        }
    }
}

/// `y ⊖ x` on a space, built in the open definition.
fn local(space: &Space, y: QDyn, x: QDyn) -> QDyn {
    let tangent = Space::Vector(space.dimension().unwrap_or(0));
    match space {
        Space::Real | Space::Vector(_) | Space::Matrix(_, _) => reshape(y - x, space, &tangent),
        Space::Positive(_) => {
            let log = |q| apply(Arc::new(PositiveLog), &[q]);
            log(y) - log(x)
        }
        Space::Product(_) => {
            let factors: Vec<QDyn> = space
                .blocks()
                .into_iter()
                .map(|block| {
                    let slice = Slice::new(block.space.clone(), block.storage, block.tangent);
                    let (yi, xi) = (
                        apply(Arc::new(slice.clone()), &[y]),
                        apply(Arc::new(slice), &[x]),
                    );
                    match block.space.inverse_retraction() {
                        Ok(inverse) => inverse.call(&[yi, xi]),
                        Err(_) => unreachable!("a product's factors are continuous"),
                    }
                })
                .collect();
            apply(Arc::new(Concatenate::new(tangent)), &factors)
        }
        // Q(ι(x), ι(y)).
        Space::Embedded(declaration) => {
            let include = |q| apply(Arc::new(Include::new(declaration.clone())), &[q]);
            declaration.inverse().call(&[include(x), include(y)])
        }
        Space::Categorical(_) | Space::Integer => {
            unreachable!("a discrete space has no retraction, refused before it is defined")
        }
    }
}

/// The slices of `x` and `t` that belong to `block`.
fn block_slices(block: &crate::Block, x: QDyn, t: QDyn) -> (QDyn, QDyn) {
    let d = block.tangent.len();
    let xi = Slice::new(
        block.space.clone(),
        block.storage.clone(),
        block.tangent.clone(),
    );
    let ti = Slice::new(
        Space::Vector(d),
        block.tangent.clone(),
        block.tangent.clone(),
    );
    (apply(Arc::new(xi), &[x]), apply(Arc::new(ti), &[t]))
}

/// `λ_x(t)` on a space with a measure, built in the open definition.
fn log_volume(space: &Space, x: QDyn, t: QDyn) -> QDyn {
    match space {
        Space::Real | Space::Vector(_) | Space::Matrix(_, _) => constant(Real, 0.0).untyped(),
        Space::Positive(_) => (apply(Arc::new(PositiveLog), &[x]) + t).sum(),
        Space::Product(_) => {
            let terms: Vec<QDyn> = space
                .blocks()
                .iter()
                .map(|block| {
                    let (xi, ti) = block_slices(block, x, t);
                    match block.space.log_volume() {
                        Ok(log_volume) => log_volume.call(&[xi, ti]),
                        Err(_) => unreachable!("a product with a measure has factors with one"),
                    }
                })
                .collect();
            terms
                .into_iter()
                .reduce(|a, b| a + b)
                .unwrap_or_else(|| constant(Real, 0.0).untyped())
        }
        Space::Embedded(declaration) => {
            let Some(volume) = declaration.volume() else {
                unreachable!("a space without a measure is refused before it is defined")
            };
            let include = apply(Arc::new(Include::new(declaration.clone())), &[x]);
            volume.log_volume().call(&[include, t])
        }
        Space::Categorical(_) | Space::Integer => {
            unreachable!("a discrete space has no chart, refused before it is defined")
        }
    }
}

/// `t ∈ U_x` on a space whose charts are not all global, built in the open definition.
fn domain(space: &Space, x: QDyn, t: QDyn) -> QDyn {
    match space {
        Space::Product(_) => {
            let within: Vec<QDyn> = space
                .blocks()
                .iter()
                .filter_map(|block| match block.space.domain() {
                    Ok(Domain::Global) => None,
                    Ok(Domain::Within(domain)) => {
                        let (xi, ti) = block_slices(block, x, t);
                        Some(apply(Arc::new(ToReal), &[domain.call(&[xi, ti])]))
                    }
                    Err(_) => unreachable!("a product with a measure has factors with one"),
                })
                .collect();
            let k = within.len() as f64;
            match within.into_iter().reduce(|a, b| a + b) {
                // The sum of k booleans is k exactly when every one is true.
                Some(count) => apply(Arc::new(IsNonnegative), &[count - k]),
                None => unreachable!("a product with a domain has a factor with one"),
            }
        }
        Space::Embedded(declaration) => match declaration.volume().map(|v| v.domain()) {
            Some(Domain::Within(domain)) => {
                let include = apply(Arc::new(Include::new(declaration.clone())), &[x]);
                domain.call(&[include, t])
            }
            _ => unreachable!("a global chart has no domain to define"),
        },
        _ => unreachable!("the core's own spaces have global charts"),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::panic)]

    use skoro_tensor::Target;

    use super::*;
    use crate::testing::{check_agreement, check_operation, random};

    fn inputs(space: &Space, rows: usize, seed: u64) -> Vec<(Space, skoro_tensor::HostValue)> {
        let d = space.dimension().unwrap();
        vec![
            (space.clone(), random(space, rows, seed, 0.3, 2.0)),
            (
                Space::Vector(d),
                random(&Space::Vector(d), rows, seed + 1, -1.5, 1.5),
            ),
        ]
    }

    /// Zero on every row, read from both inputs, so that a program of it has both roots.
    fn reads_both(q: &[QDyn]) -> QDyn {
        (q[0] * 0.0).sum() + (q[1] * 0.0).sum()
    }

    #[test]
    fn euclidean_log_volume_is_zero() {
        let target = Target::host();
        for space in [Space::Real, Space::Vector(3), Space::Matrix(2, 3)] {
            check_agreement(
                |q| q[0].log_volume(q[1]) + reads_both(q),
                reads_both,
                &inputs(&space, 4, 10),
                &target,
            )
            .unwrap_or_else(|e| panic!("{space:?}: {e}"));
            assert!(matches!(space.domain(), Ok(Domain::Global)));
            assert!(space.has_measure());
        }
    }

    #[test]
    fn positive_log_volume_is_the_log_of_the_moved_point() {
        let target = Target::host();
        let space = Space::Positive(3);
        let at = inputs(&space, 5, 20);
        // The change of variables: the log-determinant of diag(x e^t) is Σ log (x ⊕ t)ᵢ.
        check_agreement(
            |q| q[0].log_volume(q[1]),
            |q| q[0].retract(q[1]).log().sum(),
            &at,
            &target,
        )
        .unwrap();
        // Read at the coordinate of a point, it is Σ log yᵢ.
        let points = [
            (space.clone(), random(&space, 5, 30, 0.3, 2.0)),
            (space.clone(), random(&space, 5, 31, 0.3, 2.0)),
        ];
        check_agreement(
            |q| q[0].log_volume(q[1].local(q[0])),
            // Reading the base point too, so that both programs have both roots.
            |q| q[1].log().sum() + (q[0].log() * 0.0).sum(),
            &points,
            &target,
        )
        .unwrap();
        let composite = space.log_volume().unwrap();
        check_operation(&*composite.operation(), &at, &target).unwrap();
        assert!(matches!(space.domain(), Ok(Domain::Global)));
    }

    #[test]
    fn product_log_volume_is_the_sum_of_its_factors() {
        let target = Target::host();
        let inner = Space::product([Space::Real, Space::Positive(2)]).unwrap();
        let space = Space::product([inner.clone(), Space::Vector(2)]).unwrap();
        // By hand: the inner product's real factor adds 0, its positive factor Σ (log x + t),
        // and the vector adds 0.
        let by_hand = |q: &[QDyn]| {
            let (x, t) = (q[0], q[1]);
            let positive = x.factor(0).factor(1);
            let ts = apply(Arc::new(Slice::new(Space::Vector(2), 1..3, 1..3)), &[t]);
            (positive.log() + ts).sum()
        };
        check_agreement(
            |q| q[0].log_volume(q[1]),
            by_hand,
            &inputs(&space, 4, 40),
            &target,
        )
        .unwrap();
        assert!(matches!(space.domain(), Ok(Domain::Global)));
        assert!(space.has_measure());
    }

    #[test]
    fn discrete_space_has_a_measure_and_no_log_volume() {
        for space in [Space::Integer, Space::Categorical(3)] {
            assert!(space.has_measure());
            let not_continuous = Error::NotContinuous {
                space: space.clone(),
            };
            assert_eq!(space.log_volume().err(), Some(not_continuous.clone()));
            assert!(matches!(space.domain(), Err(e) if e == not_continuous));
        }
    }
}
