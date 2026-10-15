//! The retractions of the core's spaces, and their inverses, as composites cached per space.
//!
//! | space | `x ⊕ t` | `y ⊖ x` |
//! | --- | --- | --- |
//! | `Real`, `Vector`, `Matrix` | `x + reshape(t)` | `reshape(y - x)` |
//! | `Positive` | `positive.mul(x, positive.exp(t))` | `positive.log(y) - positive.log(x)` |
//! | `Product` | each factor's, on the slices of `x` and `t` by `blocks()`, concatenated | the same |
//!
//! The tangent of every space is a `Vector` of the space's dimension.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use crate::primitive::{Concatenate, PositiveExp, PositiveLog, PositiveMul, Slice};
use crate::scope::apply;
use crate::{Composite, Error, Param, QDyn, Space};

/// Which of a space's two maps.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Map {
    Retraction,
    Inverse,
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
        Space::Categorical(_) | Space::Integer => {
            unreachable!("a discrete space has no retraction, refused before it is defined")
        }
    }
}
