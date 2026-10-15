use std::sync::Arc;

use rpds::{HashTrieMapSync, HashTrieSetSync, RedBlackTreeMapSync};
use skoro_tensor::HostValue;

use crate::{Address, NodeId, Operation, Space};

/// An immutable model, sharing structure with the snapshots before it.
#[derive(Clone)]
pub struct Snapshot(pub(crate) Arc<Model>);

/// The compilation graph and the addresses of its quantities, in persistent maps, so that a clone
/// is a handful of reference counts and a change copies only the path it touches.
#[derive(Clone)]
pub(crate) struct Model {
    pub(crate) nodes: HashTrieMapSync<NodeId, Arc<Node>>,
    /// The compilation graph's edges read forward: each node's consumers.
    pub(crate) consumers: HashTrieMapSync<NodeId, HashTrieSetSync<NodeId>>,
    pub(crate) addresses: RedBlackTreeMapSync<Address, NodeId>,
}

/// A node of the compilation graph.
#[derive(Debug)]
pub struct Node {
    id: NodeId,
    space: Space,
    kind: NodeKind,
}

#[derive(Debug)]
pub(crate) enum NodeKind {
    Quantity {
        address: Address,
    },
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
            NodeKind::Quantity { .. } => None,
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
        }
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

    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.0.nodes.get(&id).map(Arc::as_ref)
    }

    pub fn space(&self, id: NodeId) -> Option<&Space> {
        self.node(id).map(Node::space)
    }

    pub fn address(&self, id: NodeId) -> Option<&Address> {
        self.node(id).and_then(Node::address)
    }

    /// The quantity at `address`.
    pub fn lookup(&self, address: &Address) -> Option<NodeId> {
        self.0.addresses.get(address).copied()
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
