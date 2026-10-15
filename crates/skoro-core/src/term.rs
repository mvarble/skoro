//! The probabilistic graph: terms, the samplers and proposals that draw quantities, and the roles
//! by which an algorithm finds a term's parts.

use std::collections::HashSet;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::panic::Location;
use std::sync::Arc;

use skoro_tensor::{Target, Value};

use crate::model::Model;
use crate::scope;
use crate::{Error, Key, NodeId, Param, Q, QDyn, Real, Typed};

/// A term: a log-density, the quantity it scores if any, and the quantities it involves.
#[derive(Clone, Debug)]
pub struct Term {
    /// A mapped node of space `Real`.
    pub log_density: NodeId,
    /// `s_t`: the quantity the term scores, `None` for a likelihood factor.
    pub scores: Option<NodeId>,
    /// `V_t`, sorted: the quantities the log-density is reached from, with the one it scores.
    pub involves: Box<[NodeId]>,
    pub sampler: Option<Sampler>,
    bindings: Vec<(&'static str, NodeId)>,
}

impl Term {
    /// The quantities the term is conditioned on: those it involves, but the one it scores.
    pub fn conditioned_on(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.involves
            .iter()
            .copied()
            .filter(|&q| Some(q) != self.scores)
    }
}

/// What draws values: a named, parameterized map from its inputs' values and keys to one draw per
/// row.
pub trait Sample: Debug + Send + Sync + 'static {
    fn name(&self) -> &'static str;
    fn params(&self) -> Vec<Param>;
    /// One draw per row, from the values of the sampler's inputs and one key per row.
    fn draw(&self, inputs: &[Value], keys: &[Key], target: &Target) -> Value;
}

/// How the quantity a term scores is drawn.
#[derive(Clone, Debug)]
pub struct Sampler {
    pub sample: Arc<dyn Sample>,
    pub inputs: Vec<QDyn>,
    /// `None` for an exact sampler; otherwise `log q_t`, a `Real` node reading `s_t` and `C_t`.
    pub log_density: Option<Q<Real>>,
}

/// How a parameter is drawn when a model is generated.
#[derive(Clone, Debug)]
pub struct Proposal {
    pub sample: Arc<dyn Sample>,
    pub inputs: Vec<QDyn>,
    pub log_density: Q<Real>,
}

/// A handle on a term, which amends it while the scope that added it is open.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TermHandle {
    id: NodeId,
}

impl From<TermHandle> for NodeId {
    fn from(term: TermHandle) -> NodeId {
        term.id
    }
}

/// The handle of the term's log-density.
impl From<TermHandle> for QDyn {
    fn from(term: TermHandle) -> QDyn {
        QDyn::from_id(term.id)
    }
}

impl TermHandle {
    /// The term's log-density.
    pub fn log_density(&self) -> Q<Real> {
        Q::from_id(self.id)
    }

    /// The term's identifier: its log-density node.
    pub fn id(&self) -> NodeId {
        self.id
    }

    pub fn is_failed(&self) -> bool {
        QDyn::from_id(self.id).is_failed()
    }

    /// Draws the quantity the term scores with `sampler`, whose inputs read none but the
    /// quantities the term is conditioned on, and whose log-density none but those the term
    /// involves (`SamplerOutsideTerm`).
    #[track_caller]
    pub fn sampler(self, sampler: Sampler) -> TermHandle {
        let location = Location::caller();
        let id = self.id;
        TermHandle {
            id: scope::build(location, |draft, _| {
                draft.term_handle(id)?;
                let density: Vec<QDyn> = sampler.log_density.iter().map(|q| q.untyped()).collect();
                draft.spaces(&sampler.inputs)?;
                draft.spaces(&density)?;
                let mut term = term_of(&draft.model, id)?;
                let conditioned: HashSet<NodeId> = term.conditioned_on().collect();
                let outside = |nodes: &[QDyn], allowed: &dyn Fn(NodeId) -> bool| {
                    nodes
                        .iter()
                        .flat_map(|node| roots(&draft.model, node.id()))
                        .find(|&q| !allowed(q))
                };
                let reads = outside(&sampler.inputs, &|q| conditioned.contains(&q))
                    .or_else(|| outside(&density, &|q| term.involves.contains(&q)));
                if let Some(quantity) = reads {
                    return Err(Error::SamplerOutsideTerm { term: id, quantity });
                }
                term.sampler = Some(sampler);
                draft.model.terms.insert_mut(id, Arc::new(term));
                Ok(id)
            }),
        }
    }

    /// Binds `node` to `role`: a node reached from no quantity outside the term
    /// (`BindingOutsideTerm`).
    #[track_caller]
    pub fn bind<M: Multiplicity>(self, role: &Role<M>, node: impl Into<QDyn>) -> TermHandle {
        let (location, node, name) = (Location::caller(), node.into(), role.name);
        let id = self.id;
        TermHandle {
            id: scope::build(location, |draft, _| {
                draft.term_handle(id)?;
                draft.spaces(&[node])?;
                let mut term = term_of(&draft.model, id)?;
                let outside = roots(&draft.model, node.id())
                    .into_iter()
                    .any(|q| !term.involves.contains(&q));
                if outside {
                    return Err(Error::BindingOutsideTerm {
                        role: name,
                        node: node.id(),
                    });
                }
                term.bindings.push((name, node.id()));
                draft.model.terms.insert_mut(id, Arc::new(term));
                Ok(id)
            }),
        }
    }
}

fn term_of(model: &Model, id: NodeId) -> Result<Term, Error> {
    model
        .terms
        .get(&id)
        .map(|term| (**term).clone())
        .ok_or(Error::NotATerm { id })
}

/// The quantities among the roots `id` is reached from, walking back through mapped nodes and
/// stopping at nodes fixed in the model.
pub(crate) fn roots(model: &Model, id: NodeId) -> Vec<NodeId> {
    let mut seen = HashSet::new();
    let mut quantities = Vec::new();
    let mut stack = vec![id];
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        let Some(node) = model.nodes.get(&id) else {
            continue;
        };
        if node.is_quantity() {
            quantities.push(id);
        } else if !node.is_fixed() {
            stack.extend(node.inputs().iter().copied());
        }
    }
    quantities.sort_unstable();
    quantities
}

/// A typed key by which an algorithm finds a part of a term: its name and how many nodes it binds.
#[derive(Debug)]
pub struct Role<M: Multiplicity> {
    name: &'static str,
    multiplicity: PhantomData<fn() -> M>,
}

impl<M: Multiplicity> Role<M> {
    pub const fn new(name: &'static str) -> Role<M> {
        Role {
            name,
            multiplicity: PhantomData,
        }
    }

    pub fn name(&self) -> &'static str {
        self.name
    }
}

/// How many nodes a role binds, and what a lookup of it gives.
pub trait Multiplicity: 'static {
    type Handles;
    /// The handles of the nodes bound, `None` when their number is wrong.
    fn handles(nodes: Vec<QDyn>) -> Option<Self::Handles>;
}

/// Exactly one node: a `QDyn`.
#[derive(Debug)]
pub struct ExactlyOne;
/// None or one: an `Option<QDyn>`.
#[derive(Debug)]
pub struct AtMostOne;
/// Any number: a `Vec<QDyn>`.
#[derive(Debug)]
pub struct AnyNumber;

impl Multiplicity for ExactlyOne {
    type Handles = QDyn;
    fn handles(nodes: Vec<QDyn>) -> Option<QDyn> {
        match nodes.as_slice() {
            [one] => Some(*one),
            _ => None,
        }
    }
}

impl Multiplicity for AtMostOne {
    type Handles = Option<QDyn>;
    fn handles(nodes: Vec<QDyn>) -> Option<Option<QDyn>> {
        match nodes.as_slice() {
            [] => Some(None),
            [one] => Some(Some(*one)),
            _ => None,
        }
    }
}

impl Multiplicity for AnyNumber {
    type Handles = Vec<QDyn>;
    fn handles(nodes: Vec<QDyn>) -> Option<Vec<QDyn>> {
        Some(nodes)
    }
}

impl Term {
    /// The nodes bound to `role`, as its multiplicity gives them (`Binding` when their number is
    /// wrong).
    pub fn bound<M: Multiplicity>(&self, role: &Role<M>) -> Result<M::Handles, Error> {
        let nodes: Vec<QDyn> = self
            .bindings
            .iter()
            .filter(|(name, _)| *name == role.name)
            .map(|(_, node)| QDyn::from_id(*node))
            .collect();
        let count = nodes.len();
        M::handles(nodes).ok_or(Error::Binding {
            role: role.name,
            count,
        })
    }
}

/// Adds the term `log_density` that scores `scored`: refusing a log-density that is not a mapped
/// `Real` (`NotALogDensity`), a quantity already scored (`AlreadyScored`), and a term conditioned
/// on a quantity made after the one it scores (`ConditionedOnLater`).
#[track_caller]
pub fn score<T: Typed>(scored: Q<T>, log_density: Q<Real>) -> TermHandle {
    let location = Location::caller();
    TermHandle {
        id: scope::build(location, |draft, _| {
            draft.add_term(Some(scored.untyped()), log_density.untyped())
        }),
    }
}

/// Adds the likelihood factor `log_density`, which scores nothing.
#[track_caller]
pub fn factor(log_density: Q<Real>) -> TermHandle {
    let location = Location::caller();
    TermHandle {
        id: scope::build(location, |draft, _| {
            draft.add_term(None, log_density.untyped())
        }),
    }
}

/// Attaches `proposal` to `parameter`, conditioned on quantities made before it alone
/// (`ConditionedOnLater`).
#[track_caller]
pub fn propose<T: Typed>(parameter: Q<T>, proposal: Proposal) -> Q<T> {
    let location = Location::caller();
    Q::from_id(scope::build(location, |draft, _| {
        draft.add_proposal(parameter.untyped(), proposal)
    }))
}

impl scope::Draft {
    /// Refuses a failed term handle as a failed operand would be.
    fn term_handle(&self, id: NodeId) -> Result<(), Error> {
        self.spaces(&[QDyn::from_id(id)]).map(|_| ())
    }

    pub(crate) fn add_term(
        &mut self,
        scores: Option<QDyn>,
        log_density: QDyn,
    ) -> Result<NodeId, Error> {
        if !self.in_model() {
            return Err(Error::NotInAModel);
        }
        let inputs: Vec<QDyn> = scores.into_iter().chain([log_density]).collect();
        self.spaces(&inputs)?;
        let id = log_density.id();
        let node = self.model.nodes.get(&id).ok_or(Error::UnknownNode { id })?;
        if node.operation().is_none()
            || *node.space() != crate::Space::Real
            || self.model.terms.contains_key(&id)
        {
            return Err(Error::NotALogDensity { id });
        }
        let scores = match scores {
            Some(q) => {
                let quantity = self
                    .model
                    .nodes
                    .get(&q.id())
                    .ok_or(Error::UnknownNode { id: q.id() })?;
                if !quantity.is_quantity() {
                    return Err(Error::NotAQuantity { id: q.id() });
                }
                if self.model.scoring.contains_key(&q.id()) {
                    return Err(Error::AlreadyScored { quantity: q.id() });
                }
                Some(q.id())
            }
            None => None,
        };
        let mut involves = roots(&self.model, id);
        if let Some(scored) = scores {
            if let Some(&later) = involves.iter().find(|&&q| q > scored) {
                return Err(Error::ConditionedOnLater {
                    quantity: scored,
                    later,
                });
            }
            if !involves.contains(&scored) {
                involves.push(scored);
                involves.sort_unstable();
            }
        }
        let term = Term {
            log_density: id,
            scores,
            involves: involves.into(),
            sampler: None,
            bindings: Vec::new(),
        };
        self.model.insert_term(term);
        Ok(id)
    }

    pub(crate) fn add_proposal(
        &mut self,
        parameter: QDyn,
        proposal: Proposal,
    ) -> Result<NodeId, Error> {
        if !self.in_model() {
            return Err(Error::NotInAModel);
        }
        let mut inputs = vec![parameter, proposal.log_density.untyped()];
        inputs.extend(proposal.inputs.iter().copied());
        self.spaces(&inputs)?;
        let id = parameter.id();
        let node = self.model.nodes.get(&id).ok_or(Error::UnknownNode { id })?;
        if !node.is_quantity() {
            return Err(Error::NotAQuantity { id });
        }
        // Its inputs read quantities made before the parameter; its log-density those and the
        // parameter.
        let inputs = proposal
            .inputs
            .iter()
            .flat_map(|q| roots(&self.model, q.id()));
        let density = roots(&self.model, proposal.log_density.id());
        let later = inputs
            .filter(|&q| q >= id)
            .chain(density.into_iter().filter(|&q| q > id))
            .next();
        if let Some(later) = later {
            return Err(Error::ConditionedOnLater {
                quantity: id,
                later,
            });
        }
        self.model.proposals.insert_mut(id, Arc::new(proposal));
        Ok(id)
    }
}
