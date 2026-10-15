use std::sync::{Arc, Mutex};

use rpds::{HashTrieMapSync, HashTrieSetSync, RedBlackTreeMapSync};
use skoro_tensor::HostValue;

use crate::compile::ProgramKey;
use crate::term::{Multiplicity, Proposal, Role, Term};
use crate::{Address, Error, NodeId, Operation, Program, Space};

/// An immutable model, sharing structure with the snapshots before it.
#[derive(Clone)]
pub struct Snapshot(pub(crate) Arc<Model>);

/// The compilation graph and the addresses of its quantities, in persistent maps, so that a clone
/// is a handful of reference counts and a change copies only the path it touches.
pub(crate) struct Model {
    pub(crate) nodes: HashTrieMapSync<NodeId, Arc<Node>>,
    /// The compilation graph's edges read forward: each node's consumers.
    pub(crate) consumers: HashTrieMapSync<NodeId, HashTrieSetSync<NodeId>>,
    pub(crate) addresses: RedBlackTreeMapSync<Address, NodeId>,
    /// The probabilistic graph: terms by their log-density nodes, each quantity's scoring term,
    /// and the terms each quantity is joined to.
    pub(crate) terms: RedBlackTreeMapSync<NodeId, Arc<Term>>,
    pub(crate) scoring: HashTrieMapSync<NodeId, NodeId>,
    pub(crate) joined: HashTrieMapSync<NodeId, HashTrieSetSync<NodeId>>,
    pub(crate) proposals: HashTrieMapSync<NodeId, Arc<Proposal>>,
    /// The programs compiled from this model, most recently used last.
    pub(crate) programs: Mutex<Vec<(ProgramKey, Arc<Program>)>>,
}

/// A clone shares every map with the model it came from and none of its programs: a program
/// compiled for one snapshot is never run against another.
impl Clone for Model {
    fn clone(&self) -> Model {
        Model {
            nodes: self.nodes.clone(),
            consumers: self.consumers.clone(),
            addresses: self.addresses.clone(),
            terms: self.terms.clone(),
            scoring: self.scoring.clone(),
            joined: self.joined.clone(),
            proposals: self.proposals.clone(),
            programs: Mutex::new(Vec::new()),
        }
    }
}

/// A node of the compilation graph.
#[derive(Debug)]
pub struct Node {
    id: NodeId,
    space: Space,
    pub(crate) kind: NodeKind,
}

#[derive(Debug)]
pub(crate) enum NodeKind {
    Quantity {
        address: Address,
    },
    /// An input of a composite's body.
    Input,
    Constant {
        value: HostValue,
    },
    Mapped {
        operation: Arc<dyn Operation>,
        inputs: Box<[NodeId]>,
        folded: Option<HostValue>,
    },
}

impl Node {
    pub(crate) fn new(id: NodeId, space: Space, kind: NodeKind) -> Node {
        Node { id, space, kind }
    }

    pub fn id(&self) -> NodeId {
        self.id
    }

    pub fn space(&self) -> &Space {
        &self.space
    }

    /// A quantity's address; `None` for a constant or a mapped node.
    pub fn address(&self) -> Option<&Address> {
        match &self.kind {
            NodeKind::Quantity { address } => Some(address),
            _ => None,
        }
    }

    /// A mapped node's operation.
    pub fn operation(&self) -> Option<&dyn Operation> {
        match &self.kind {
            NodeKind::Mapped { operation, .. } => Some(operation.as_ref()),
            _ => None,
        }
    }

    /// A mapped node's inputs; empty for any other.
    pub fn inputs(&self) -> &[NodeId] {
        match &self.kind {
            NodeKind::Mapped { inputs, .. } => inputs,
            _ => &[],
        }
    }

    /// The value of a node fixed in the model: a constant's, or a mapped node's folded one.
    pub fn value(&self) -> Option<&HostValue> {
        match &self.kind {
            NodeKind::Constant { value } => Some(value),
            NodeKind::Mapped { folded, .. } => folded.as_ref(),
            NodeKind::Quantity { .. } | NodeKind::Input => None,
        }
    }

    /// Whether the node is fixed in the model: a constant, or a mapped node with a folded value.
    pub fn is_fixed(&self) -> bool {
        self.value().is_some()
    }

    pub fn is_quantity(&self) -> bool {
        matches!(self.kind, NodeKind::Quantity { .. })
    }
}

impl Model {
    pub(crate) fn new() -> Model {
        Model {
            nodes: HashTrieMapSync::new_sync(),
            consumers: HashTrieMapSync::new_sync(),
            addresses: RedBlackTreeMapSync::new_sync(),
            terms: RedBlackTreeMapSync::new_sync(),
            scoring: HashTrieMapSync::new_sync(),
            joined: HashTrieMapSync::new_sync(),
            proposals: HashTrieMapSync::new_sync(),
            programs: Mutex::new(Vec::new()),
        }
    }

    pub(crate) fn insert_term(&mut self, term: Term) {
        let id = term.log_density;
        if let Some(scored) = term.scores {
            self.scoring.insert_mut(scored, id);
        }
        for q in term.involves.iter() {
            let mut terms = self
                .joined
                .get(q)
                .cloned()
                .unwrap_or_else(HashTrieSetSync::new_sync);
            terms.insert_mut(id);
            self.joined.insert_mut(*q, terms);
        }
        self.terms.insert_mut(id, Arc::new(term));
    }

    pub(crate) fn insert(&mut self, node: Node) {
        let id = node.id;
        for input in node.inputs() {
            let mut consumers = self
                .consumers
                .get(input)
                .cloned()
                .unwrap_or_else(HashTrieSetSync::new_sync);
            consumers.insert_mut(id);
            self.consumers.insert_mut(*input, consumers);
        }
        if let Some(address) = node.address() {
            self.addresses.insert_mut(address.clone(), id);
        }
        self.nodes.insert_mut(id, Arc::new(node));
    }
}

impl Snapshot {
    pub(crate) fn new(model: Model) -> Snapshot {
        Snapshot(Arc::new(model))
    }

    /// A writer for a new branch, sharing everything with this snapshot, with a cell of its own.
    pub fn fork(&self) -> crate::Writer {
        crate::Writer::from_snapshot(self.clone())
    }

    pub fn node(&self, id: impl Into<NodeId>) -> Option<&Node> {
        self.0.nodes.get(&id.into()).map(Arc::as_ref)
    }

    pub fn space(&self, id: impl Into<NodeId>) -> Option<&Space> {
        self.node(id).map(Node::space)
    }

    pub fn address(&self, id: impl Into<NodeId>) -> Option<&Address> {
        self.node(id).and_then(Node::address)
    }

    /// The quantity at `address`.
    pub fn lookup(&self, address: &Address) -> Option<NodeId> {
        self.0.addresses.get(address).copied()
    }

    /// The quantities, in ancestral order: the order they were made in.
    pub fn quantities(&self) -> Vec<NodeId> {
        let mut quantities: Vec<NodeId> = self.0.addresses.values().copied().collect();
        quantities.sort_unstable();
        quantities
    }

    /// The quantities whose addresses lie under `namespace`, in ancestral order.
    pub fn under(&self, namespace: &Address) -> Vec<NodeId> {
        let mut quantities: Vec<NodeId> = self
            .0
            .addresses
            .range(namespace.clone()..)
            .take_while(|(address, _)| address.starts_with(namespace))
            .map(|(_, id)| *id)
            .collect();
        quantities.sort_unstable();
        quantities
    }

    /// The terms, in the order they were added.
    pub fn terms(&self) -> Vec<&Term> {
        self.0.terms.values().map(Arc::as_ref).collect()
    }

    /// The term whose log-density is `id`.
    pub fn term(&self, id: impl Into<NodeId>) -> Option<&Term> {
        self.0.terms.get(&id.into()).map(Arc::as_ref)
    }

    /// The term that scores `q`, if one does.
    pub fn scored_by(&self, q: impl Into<NodeId>) -> Option<&Term> {
        let id = self.0.scoring.get(&q.into())?;
        self.term(*id)
    }

    /// The terms `q` is joined to, in the order they were added.
    pub fn terms_of(&self, q: impl Into<NodeId>) -> Vec<&Term> {
        let mut ids: Vec<NodeId> = self
            .0
            .joined
            .get(&q.into())
            .map(|terms| terms.iter().copied().collect())
            .unwrap_or_default();
        ids.sort_unstable();
        ids.into_iter().filter_map(|id| self.term(id)).collect()
    }

    /// Whether `q` is a quantity no term scores.
    pub fn is_parameter(&self, q: impl Into<NodeId>) -> bool {
        let id = q.into();
        self.node(id).is_some_and(Node::is_quantity) && !self.0.scoring.contains_key(&id)
    }

    /// The proposal attached to the parameter `q`.
    pub fn proposal(&self, q: impl Into<NodeId>) -> Option<&Proposal> {
        self.0.proposals.get(&q.into()).map(Arc::as_ref)
    }

    /// The Markov blanket of `quantities`: the other quantities joined to the terms they are
    /// joined to, in ancestral order.
    pub fn blanket<Q: Into<NodeId>>(&self, quantities: impl IntoIterator<Item = Q>) -> Vec<NodeId> {
        let inside: std::collections::HashSet<NodeId> =
            quantities.into_iter().map(Into::into).collect();
        let mut blanket: Vec<NodeId> = inside
            .iter()
            .flat_map(|q| self.terms_of(*q))
            .flat_map(|term| term.involves.iter().copied())
            .filter(|q| !inside.contains(q))
            .collect();
        blanket.sort_unstable();
        blanket.dedup();
        blanket
    }

    /// The nodes the term `term` binds to `role`, by the role's multiplicity (`NotATerm`,
    /// `Binding`).
    pub fn bound<M: Multiplicity>(
        &self,
        term: impl Into<NodeId>,
        role: &Role<M>,
    ) -> Result<M::Handles, Error> {
        let term = term.into();
        self.term(term)
            .ok_or(Error::NotATerm { id: term })?
            .bound(role)
    }

    /// The number of nodes in the compilation graph.
    pub fn len(&self) -> usize {
        self.0.nodes.size()
    }

    pub fn is_empty(&self) -> bool {
        self.0.nodes.is_empty()
    }
}

/// Two snapshots are equal when they hold the same nodes: the same identifiers, each the same
/// allocation. Identifiers are never reused and nodes never change, so this is equality of models.
impl PartialEq for Snapshot {
    fn eq(&self, other: &Snapshot) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
            || (self.0.nodes.size() == other.0.nodes.size()
                && self
                    .0
                    .nodes
                    .iter()
                    .all(|(id, node)| other.0.nodes.get(id).is_some_and(|o| Arc::ptr_eq(node, o))))
    }
}

impl std::fmt::Debug for Snapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Snapshot({} nodes)", self.len())
    }
}
