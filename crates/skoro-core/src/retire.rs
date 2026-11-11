//! Retirement: the quantities at a collection of addresses removed, and what they said about the
//! rest kept as a replacement the caller builds, in [the three steps](math.svx#retirement) of the mathematics,
//! with the caller's step between two of the core's.

use std::collections::{BTreeSet, HashSet};
use std::panic::Location;
use std::sync::Arc;

use rpds::HashTrieSetSync;

use crate::model::Model;
use crate::scope::{self, Draft, Frame};
use crate::term::{Proposal, Term, named, named_by, roots};
use crate::{
    Address, Error, Failure, Multiplicity, NodeId, Observations, Q, QDyn, Real, Role, Snapshot,
    Space, TermHandle, Writer,
};

/// The addresses a retirement retires: some named one by one, and every address under some
/// namespaces.
#[derive(Clone, Default, Debug)]
pub struct Addresses {
    exact: BTreeSet<Address>,
    namespaces: BTreeSet<Address>,
}

impl Addresses {
    pub fn new() -> Addresses {
        Addresses::default()
    }

    /// The quantity at `address`, which must exist when the retirement is read.
    pub fn at(mut self, address: impl Into<Address>) -> Addresses {
        self.exact.insert(address.into());
        self
    }

    /// Every quantity whose address has `namespace` as a proper prefix, when the retirement is read.
    pub fn under(mut self, namespace: impl Into<Address>) -> Addresses {
        self.namespaces.insert(namespace.into());
        self
    }

    /// The quantities at these addresses in `snapshot`, in ancestral order; `UnknownAddress` for
    /// an exact address with no quantity.
    fn resolve(&self, snapshot: &Snapshot) -> Result<Vec<NodeId>, Error> {
        let mut quantities: BTreeSet<NodeId> = BTreeSet::new();
        for address in &self.exact {
            let q = snapshot
                .lookup(address)
                .ok_or_else(|| Error::UnknownAddress {
                    address: address.clone(),
                })?;
            quantities.insert(q);
        }
        for namespace in &self.namespaces {
            quantities.extend(
                snapshot
                    .under(namespace)
                    .into_iter()
                    .filter(|&q| snapshot.address(q) != Some(namespace)),
            );
        }
        Ok(quantities.into_iter().collect())
    }
}

/// What a retirement reads before anything changes.
#[derive(Clone, Debug)]
pub struct Plan {
    /// `E`: the quantities at the retirement's addresses, in ancestral order.
    pub retired: Vec<QDyn>,
    /// The values of `E_O`.
    pub observed: Observations,
    /// `E_F`: the quantities of `E` the replacement integrates out.
    pub integrated: Vec<QDyn>,
    /// `𝒯_E`: the terms joined to a quantity of `E`, in the order they were added.
    pub terms: Vec<Arc<Term>>,
    /// `B`: the other quantities joined to those terms, with their spaces and addresses.
    pub blanket: Vec<(QDyn, Space, Address)>,
    /// Parameters outside `E` whose proposal reads a quantity of `E`, and so loses it.
    pub unproposed: Vec<QDyn>,
}

/// What the caller builds: the replacement `ℓ_B`, a likelihood factor over the blanket,
/// proposals for the quantities the retirement leaves to be drawn by one, and the nodes the
/// factor binds to roles, as `TermHandle::bind` binds them on a term.
pub struct Replacement {
    pub log_density: Q<Real>,
    pub proposals: Vec<(QDyn, Proposal)>,
    /// Each a role's name and the node bound to it.
    pub bindings: Vec<(&'static str, QDyn)>,
}

impl Replacement {
    /// Binds `node` to `role` on the factor the retirement adds: a node reached from no quantity
    /// outside the factor's, which `Retirement::replace` checks (`BindingOutsideTerm`).
    pub fn bind<M: Multiplicity>(mut self, role: &Role<M>, node: impl Into<QDyn>) -> Replacement {
        self.bindings.push((role.name(), node.into()));
        self
    }
}

/// A retirement read and not yet applied: the writer, borrowed, the snapshot it read, and the plan.
pub struct Retirement<'w> {
    writer: &'w mut Writer,
    snapshot: Snapshot,
    plan: Plan,
}

impl Writer {
    /// Reads the retirement of the quantities at `addresses`, resolved on the latest snapshot,
    /// with `observed` the values of those it fixes rather than integrates; refuses an exact
    /// address with no quantity (`UnknownAddress`), an observed quantity outside them
    /// (`ObservedNotRetired`), and a quantity to integrate whose space has no reference measure
    /// (`NoVolume`). Changes nothing.
    pub fn retire(
        &mut self,
        addresses: Addresses,
        observed: Observations,
    ) -> Result<Retirement<'_>, Error> {
        let snapshot = self.snapshot();
        let retired = addresses.resolve(&snapshot)?;
        let inside: HashSet<NodeId> = retired.iter().copied().collect();
        for id in observed.ids() {
            if !inside.contains(&id) {
                return Err(Error::ObservedNotRetired { quantity: id });
            }
            if matches!(observed.get(id), Some(None)) {
                let space = snapshot.space(id).cloned().unwrap_or(Space::Real);
                return Err(Error::ValueShape { space });
            }
        }
        let model = &snapshot.0;
        let mut term_ids: Vec<NodeId> = retired
            .iter()
            .filter_map(|q| model.joined.get(q))
            .flat_map(|terms| terms.iter().copied())
            .collect();
        term_ids.sort_unstable();
        term_ids.dedup();
        let terms: Vec<Arc<Term>> = term_ids
            .iter()
            .filter_map(|id| model.terms.get(id).cloned())
            .collect();
        let mut blanket: Vec<NodeId> = terms
            .iter()
            .flat_map(|term| term.involves.iter().copied())
            .filter(|q| !inside.contains(q))
            .collect();
        blanket.sort_unstable();
        blanket.dedup();
        let mut unproposed: Vec<NodeId> = model
            .proposals
            .iter()
            .filter(|(q, proposal)| !inside.contains(q) && reads(model, proposal, &inside))
            .map(|(q, _)| *q)
            .collect();
        unproposed.sort_unstable();
        let handle = |id: &NodeId| QDyn::from_id(*id);
        let plan = Plan {
            integrated: retired
                .iter()
                .filter(|&&q| observed.get(q).is_none())
                .map(handle)
                .collect(),
            retired: retired.iter().map(handle).collect(),
            observed,
            terms,
            blanket: blanket
                .iter()
                .filter_map(|&q| {
                    let node = snapshot.node(q)?;
                    Some((handle(&q), node.space().clone(), node.address()?.clone()))
                })
                .collect(),
            unproposed: unproposed.iter().map(handle).collect(),
        };
        // Integrating a quantity out is an integral against its space's measure.
        for q in &plan.integrated {
            let space = snapshot.space(q.id()).cloned().unwrap_or(Space::Real);
            if !space.has_measure() {
                return Err(Error::NoVolume {
                    space,
                    quantity: Some(q.id()),
                });
            }
        }
        Ok(Retirement {
            writer: self,
            snapshot,
            plan,
        })
    }
}

/// Whether `proposal` reads a quantity of `inside`, through its inputs or its log-density.
fn reads(model: &Model, proposal: &Proposal, inside: &HashSet<NodeId>) -> bool {
    first_read(model, proposal, inside).is_some()
}

fn first_read(model: &Model, proposal: &Proposal, inside: &HashSet<NodeId>) -> Option<NodeId> {
    proposal
        .inputs
        .iter()
        .map(QDyn::id)
        .chain([proposal.log_density.id()])
        .flat_map(|id| roots(model, id))
        .find(|q| inside.contains(q))
}

impl Retirement<'_> {
    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    /// The model as it is, to evaluate the terms of the plan on.
    pub fn snapshot(&self) -> &Snapshot {
        &self.snapshot
    }

    /// Builds the replacement in a scope, checks it, and applies the change: the replacement
    /// must be reached from blanket quantities alone (`ReplacementOutsideBlanket`), and each
    /// proposal be for a blanket quantity the retirement leaves unscored or a parameter of
    /// `unproposed` (`ProposalNotNeeded`), reading no retired quantity (`ProposalReadsRetired`),
    /// and each binding be of a node reached from no quantity outside the replacement's
    /// (`BindingOutsideTerm`). Then `E`, the terms of `𝒯_E`, the proposals of `E` and of
    /// `unproposed`, and every mapped node reached from `E` are removed, the replacement is added
    /// as a likelihood factor with its bindings and the proposals attached, and one snapshot is
    /// published; the factor's handle is given back. A failure at any point leaves the model as
    /// it was. A retirement of nothing changes nothing, does not call `body`, and gives back no
    /// factor.
    #[track_caller]
    pub fn replace(
        self,
        body: impl FnOnce(&Plan) -> Replacement,
    ) -> Result<Option<TermHandle>, Failure> {
        let location = Location::caller();
        let fail = |error: Error| Failure { error, location };
        let Retirement {
            writer,
            snapshot,
            plan,
        } = self;
        if plan.retired.is_empty() {
            return Ok(None);
        }
        let draft = Draft::new(snapshot.0.as_ref().clone());
        let (replacement, frame) = scope::open(Frame::Scope(draft), || body(&plan));
        let Frame::Scope(mut draft) = frame else {
            unreachable!("replace gets back the scope it opened")
        };
        if let Some(failure) = draft.failure {
            return Err(failure);
        }

        // 1. The checks.
        let retired: HashSet<NodeId> = plan.retired.iter().map(QDyn::id).collect();
        let blanket: HashSet<NodeId> = plan.blanket.iter().map(|(q, ..)| q.id()).collect();
        let replaced: HashSet<NodeId> = plan.terms.iter().map(|t| t.log_density).collect();
        let unproposed: HashSet<NodeId> = plan.unproposed.iter().map(QDyn::id).collect();
        let log_density = replacement.log_density.untyped();
        draft.spaces(&[log_density]).map_err(fail)?;
        let involves = roots(&draft.model, log_density.id());
        if let Some(&quantity) = involves.iter().find(|q| !blanket.contains(q)) {
            return Err(fail(Error::ReplacementOutsideBlanket { quantity }));
        }
        for &(role, node) in &replacement.bindings {
            draft.spaces(&[node]).map_err(fail)?;
            if roots(&draft.model, node.id())
                .iter()
                .any(|q| !involves.contains(q))
            {
                return Err(fail(Error::BindingOutsideTerm {
                    role,
                    node: node.id(),
                }));
            }
        }
        for (q, proposal) in &replacement.proposals {
            let id = q.id();
            let unscored = blanket.contains(&id)
                && snapshot
                    .scored_by(id)
                    .is_some_and(|term| replaced.contains(&term.log_density));
            if !unscored && !unproposed.contains(&id) {
                return Err(fail(Error::ProposalNotNeeded { quantity: id }));
            }
            if let Some(read) = first_read(&draft.model, proposal, &retired) {
                return Err(fail(Error::ProposalReadsRetired {
                    quantity: id,
                    retired: read,
                }));
            }
        }

        // 2. The removal.
        remove(&mut draft.model, &plan, &retired, &unproposed);

        // 3. The replacement, its bindings, and its proposals.
        let factor = draft.add_term(None, log_density).map_err(fail)?;
        if let Some(term) = draft.model.terms.get(&factor) {
            let mut term = (**term).clone();
            term.bindings.extend(
                replacement
                    .bindings
                    .iter()
                    .map(|&(role, node)| (role, node.id())),
            );
            for &(_, node) in &replacement.bindings {
                draft.model.hold(node.id());
            }
            draft.model.terms.insert_mut(factor, Arc::new(term));
        }
        for (q, proposal) in replacement.proposals {
            draft.add_proposal(q, proposal).map_err(fail)?;
        }
        writer.publish(draft.model);
        Ok(Some(TermHandle::from_id(factor)))
    }
}

/// Removes `E`, the terms of `𝒯_E`, the proposals of `E` and of `unproposed`, and every node
/// reached from `E` through the consumers, with every edge that names one of them.
fn remove(model: &mut Model, plan: &Plan, retired: &HashSet<NodeId>, unproposed: &HashSet<NodeId>) {
    let mut removed: HashSet<NodeId> = retired.clone();
    let mut stack: Vec<NodeId> = retired.iter().copied().collect();
    while let Some(id) = stack.pop() {
        for consumer in model.consumers.get(&id).into_iter().flatten() {
            if removed.insert(*consumer) {
                stack.push(*consumer);
            }
        }
    }

    for term in &plan.terms {
        for node in named(term) {
            model.release(node);
        }
        model.terms.remove_mut(&term.log_density);
        if let Some(scored) = term.scores {
            model.scoring.remove_mut(&scored);
        }
        for q in term.involves.iter() {
            let Some(joined) = model.joined.get(q) else {
                continue;
            };
            let joined: HashTrieSetSync<NodeId> = joined.remove(&term.log_density);
            if joined.is_empty() || retired.contains(q) {
                model.joined.remove_mut(q);
            } else {
                model.joined.insert_mut(*q, joined);
            }
        }
    }
    for q in retired.iter().chain(unproposed) {
        if let Some(proposal) = model.proposals.get(q).cloned() {
            for node in named_by(&proposal) {
                model.release(node);
            }
        }
        model.proposals.remove_mut(q);
    }

    for id in &removed {
        let Some(node) = model.nodes.get(id).cloned() else {
            continue;
        };
        for input in node.inputs() {
            if removed.contains(input) {
                continue;
            }
            if let Some(consumers) = model.consumers.get(input) {
                let consumers = consumers.remove(id);
                if consumers.is_empty() {
                    model.consumers.remove_mut(input);
                } else {
                    model.consumers.insert_mut(*input, consumers);
                }
            }
        }
        if let Some(address) = node.address() {
            model.addresses.remove_mut(address);
        }
        model.consumers.remove_mut(id);
        model.nodes.remove_mut(id);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use skoro_tensor::{Target, Value};

    use super::*;
    use crate::scope::{constant, factor, propose, quantity, score};
    use crate::term::{Sample, Sampler};
    use crate::{AnyNumber, Key, Param, Real};

    const SCALE: Role<AnyNumber> = Role::new("scale");

    /// A sampler that never draws: the index reads its nodes, and nothing is generated.
    #[derive(Debug)]
    struct Never;

    impl Sample for Never {
        fn name(&self) -> &'static str {
            "test.never"
        }

        fn params(&self) -> Vec<Param> {
            Vec::new()
        }

        fn draw(&self, _: &[Value], _: &[Key], _: &Target) -> Value {
            unreachable!("these tests draw nothing")
        }
    }

    fn sampler(inputs: Vec<QDyn>, log_density: Option<Q<Real>>) -> Sampler {
        Sampler {
            sample: Arc::new(Never),
            inputs,
            log_density,
        }
    }

    fn square(x: Q<Real>, m: Q<Real>) -> Q<Real> {
        (x - m) * (x - m) * -0.5
    }

    /// `held`, recounted from every term's bindings and sampler and every proposal, and every
    /// spared node a node of the model.
    fn assert_index(writer: &Writer) {
        let model = &writer.snapshot().0;
        let mut counts: HashMap<NodeId, u32> = HashMap::new();
        let terms = model.terms.values().flat_map(|term| named(term));
        let proposals = model.proposals.values().flat_map(|p| named_by(p));
        for node in terms.chain(proposals) {
            *counts.entry(node).or_default() += 1;
        }
        let held: HashMap<NodeId, u32> = model.held.iter().map(|(&k, &v)| (k, v)).collect();
        assert_eq!(held, counts);
        for id in model.spared.iter() {
            assert!(model.nodes.contains_key(id), "{id:?} is spared but gone");
        }
    }

    #[test]
    fn held_index_matches_the_terms() {
        let mut writer = Writer::new();
        let (x1, p, x2) = writer
            .build(|| {
                let m = constant(Real, 0.5);
                let s = constant(Real, 2.0);
                let c = constant(Real, 1.0);
                let x0 = quantity("t/0/x", Real);
                score(x0, square(x0, m))
                    .bind(&SCALE, s)
                    .bind(&SCALE, s)
                    .sampler(sampler(vec![m.untyped()], None))
                    .sampler(sampler(vec![c.untyped(), m.untyped()], Some(square(x0, c))));
                let x1 = quantity("t/1/x", Real);
                score(x1, square(x1, x0)).sampler(sampler(vec![x0.untyped()], None));
                let p = quantity("t/1/p", Real);
                let proposal = |mean: Q<Real>| Proposal {
                    sample: Arc::new(Never),
                    inputs: vec![mean.untyped()],
                    log_density: square(p, mean),
                };
                propose(p, proposal(m));
                propose(p, proposal(c));
                let x2 = quantity("t/2/x", Real);
                factor(square(x2, x1) + square(p, x1));
                (x1, p, x2)
            })
            .unwrap();
        assert_index(&writer);

        writer
            .retire(Addresses::new().under("t/0"), Observations::new())
            .unwrap()
            .replace(|_| {
                let k = constant(Real, 3.0);
                Replacement {
                    log_density: square(x1, k),
                    proposals: Vec::new(),
                    bindings: Vec::new(),
                }
                .bind(&SCALE, k)
            })
            .unwrap();
        assert_index(&writer);

        writer
            .retire(Addresses::new().under("t/1"), Observations::new())
            .unwrap()
            .replace(|_| Replacement {
                log_density: square(x2, constant(Real, 0.0)),
                proposals: Vec::new(),
                bindings: Vec::new(),
            })
            .unwrap();
        assert_index(&writer);
        let model = &writer.snapshot().0;
        assert!(model.proposals.get(&p.id()).is_none());
        assert_eq!(
            model.held.size(),
            0,
            "nothing left names a node outside the edges"
        );
    }
}
