//! Generation: the free quantities of a snapshot drawn ancestrally, each from the sampler of the
//! term that scores it or from its proposal, and the importance weight of what was drawn.

use std::collections::HashMap;
use std::sync::Arc;

use skoro_tensor::{Points, Target, Value};

use crate::compile::Builder;
use crate::primitive::{Neg, Sub};
use crate::program::Slot;
use crate::{
    Error, Forward, Kernel, Key, Lowering, NodeId, Observations, Sample, Snapshot, Term, Values,
};

/// What `generate` gives: the value of every free quantity, one per row, and each row's
/// log-weight, `[rows, 1]`.
#[derive(Clone, Debug)]
pub struct Generated {
    pub values: Values,
    pub log_weight: Points,
}

/// A draw step's kernel: the sample, and the keys of its quantity, one per row.
struct Draw {
    sample: Arc<dyn Sample>,
    keys: Vec<Key>,
    target: Target,
}

impl Forward for Draw {
    fn forward(&self, inputs: &[Value]) -> Value {
        self.sample.draw(inputs, &self.keys, &self.target)
    }
}

/// How a free quantity is drawn: what draws it, and from which nodes.
struct Source<'a> {
    sample: &'a Arc<dyn Sample>,
    inputs: Vec<NodeId>,
}

impl Snapshot {
    /// Draws every quantity `given` does not observe, in increasing identifier, `rows` times, the
    /// `i`-th row of a quantity from the key `key.split(rows)[i].at(address)`; and weighs each
    /// row by [the importance weight](eq:archive/core/math/weight): the log-density of every term
    /// that scores an observed quantity or nothing, that of a term scoring a free quantity less its
    /// sampler's unless the sampler is exact, and minus each free parameter's proposal. Refuses a
    /// free quantity that is neither scored by a term with a sampler nor a parameter with a
    /// proposal (`CannotGenerate`).
    pub fn generate(
        &self,
        given: &Observations,
        key: &Key,
        rows: usize,
        target: &Target,
    ) -> Result<Generated, Error> {
        for id in given.ids() {
            if self.node(id).is_some_and(|node| !node.is_quantity()) {
                return Err(Error::NotAQuantity { id });
            }
        }
        let free: Vec<NodeId> = self
            .quantities()
            .into_iter()
            .filter(|&q| given.get(q).is_none())
            .collect();
        let sources = free
            .iter()
            .map(|&q| self.source(q))
            .collect::<Result<Vec<Source<'_>>, Error>>()?;

        let mut lowering = Lowering::new(target);
        let mut builder = Builder::new(&mut lowering);
        let mut env: HashMap<NodeId, Slot> = HashMap::new();
        let row_keys = key.split(rows);
        let mut drawn = Vec::with_capacity(free.len());
        for (&q, source) in free.iter().zip(&sources) {
            let inputs = builder.emit(&self.0, &source.inputs, &mut env, None, Some(given))?;
            let Some(address) = self.address(q) else {
                unreachable!("a quantity has an address")
            };
            let draw = Draw {
                sample: source.sample.clone(),
                keys: row_keys.iter().map(|k| k.at(address)).collect(),
                target: target.clone(),
            };
            let Some(space) = self.space(q).cloned() else {
                unreachable!("a quantity of the snapshot has a space")
            };
            let name = source.sample.name();
            let slot = builder.draw(Kernel::Zero(Arc::new(draw)), name, inputs, space, q);
            env.insert(q, slot);
            drawn.push((q, slot));
        }

        // The weight, its terms in the order a density query sums them: those whose every
        // quantity is observed first, then by identifier.
        let observed = |q: &NodeId| given.get(*q).is_some();
        let mut terms: Vec<&Term> = self.terms();
        terms.sort_by_key(|term| (!term.involves.iter().all(observed), term.log_density));
        let mut parts = Vec::new();
        for term in terms {
            let density = |builder: &mut Builder<'_, '_>, env: &mut HashMap<NodeId, Slot>| {
                builder
                    .emit(&self.0, &[term.log_density], env, None, Some(given))
                    .map(|slots| slots[0])
            };
            match term.scores.filter(|s| !observed(s)) {
                None => parts.push(density(&mut builder, &mut env)?),
                Some(_) => {
                    let sampler = term.sampler.as_ref().and_then(|s| s.log_density);
                    if let Some(proposed) = sampler {
                        let l = density(&mut builder, &mut env)?;
                        let q =
                            builder.emit(&self.0, &[proposed.id()], &mut env, None, Some(given))?
                                [0];
                        parts.push(builder.step(&Sub, &[l, q], term.log_density)?);
                    }
                }
            }
        }
        for &q in free.iter().filter(|&&q| self.is_parameter(q)) {
            let Some(proposal) = self.proposal(q) else {
                unreachable!("a free parameter has a proposal, checked above")
            };
            let density = proposal.log_density.id();
            let slot = builder.emit(&self.0, &[density], &mut env, None, Some(given))?[0];
            parts.push(builder.step(&Neg, &[slot], density)?);
        }
        let weight = builder.sum(&parts)?;

        let mut targets = drawn;
        targets.push((NodeId::DENSITY, weight));
        let mut program = builder.finish(targets)?;
        if let Some(free) = program.free.first() {
            unreachable!(
                "every quantity is drawn or observed before it is read: {:?}",
                free.node
            )
        }
        match program.batch {
            Some(pinned) if pinned != rows => {
                return Err(Error::BatchSize {
                    expected: rows,
                    found: pinned,
                });
            }
            _ => program.batch = Some(rows),
        }
        let mut values = program.forward(&Values::new())?;
        let log_weight = match values.remove(NodeId::DENSITY) {
            Some(Value::Continuous(points)) => points,
            _ => unreachable!("the weight is a continuous target"),
        };
        Ok(Generated { values, log_weight })
    }

    /// What draws the free quantity `q`: the sampler of the term that scores it, or its proposal
    /// (`CannotGenerate`).
    fn source(&self, q: NodeId) -> Result<Source<'_>, Error> {
        let Some(address) = self.address(q) else {
            unreachable!("a quantity has an address")
        };
        let cannot = || Error::CannotGenerate {
            quantity: q,
            address: address.clone(),
        };
        let ids = |inputs: &[crate::QDyn]| inputs.iter().map(|i| i.id()).collect();
        match self.scored_by(q) {
            Some(term) => {
                let sampler = term.sampler.as_ref().ok_or_else(cannot)?;
                Ok(Source {
                    sample: &sampler.sample,
                    inputs: ids(&sampler.inputs),
                })
            }
            None => {
                let proposal = self.proposal(q).ok_or_else(cannot)?;
                Ok(Source {
                    sample: &proposal.sample,
                    inputs: ids(&proposal.inputs),
                })
            }
        }
    }
}
