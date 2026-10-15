use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use rpds::HashTrieMapSync;
use skoro_tensor::HostValue;

use crate::{IntoHostValue, NodeId, Q, QDyn, Typed};

/// Values for some quantities, under an identity of their own: a set that is extended is a new
/// set. Reference-counted, so a caller keeps one and passes it to every `compile`.
#[derive(Clone, Debug)]
pub struct Observations {
    identity: u64,
    /// `None` for a value that could not be read as one of its quantity's marker.
    values: Arc<HashTrieMapSync<NodeId, Option<HostValue>>>,
}

/// Identities of extended sets; the empty set is one set, of identity zero.
static IDENTITIES: AtomicU64 = AtomicU64::new(1);

fn fresh() -> u64 {
    IDENTITIES.fetch_add(1, Ordering::Relaxed)
}

impl Observations {
    /// The empty set: one set, whose identity every empty set shares.
    pub fn new() -> Observations {
        Observations {
            identity: 0,
            values: Arc::new(HashTrieMapSync::new_sync()),
        }
    }

    /// This set with `quantity` observed at `value`: a new set.
    pub fn observe<T: Typed>(self, quantity: Q<T>, value: impl IntoHostValue<T>) -> Observations {
        self.insert(quantity.id(), value.into_host_value())
    }

    /// This set with `quantity` observed at `value`: a new set.
    pub fn observe_dyn(self, quantity: QDyn, value: HostValue) -> Observations {
        self.insert(quantity.id(), Some(value))
    }

    fn insert(self, id: NodeId, value: Option<HostValue>) -> Observations {
        Observations {
            identity: fresh(),
            values: Arc::new(self.values.insert(id, value)),
        }
    }

    /// The set's identity, by which a compiled program is cached.
    pub fn identity(&self) -> u64 {
        self.identity
    }

    /// The quantities observed.
    pub(crate) fn ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.values.keys().copied()
    }

    /// The value observed for `id`: `Some(None)` when one was given that cannot be read.
    pub(crate) fn get(&self, id: NodeId) -> Option<Option<&HostValue>> {
        self.values.get(&id).map(Option::as_ref)
    }
}

impl Default for Observations {
    fn default() -> Observations {
        Observations::new()
    }
}

/// What a program computes: its targets, and the observations it is given.
#[derive(Clone, Debug)]
pub struct Query {
    pub(crate) targets: Targets,
    pub(crate) observations: Observations,
}

/// A query's targets: nodes, or the sum of every term's log-density, or of some terms', kept as a
/// set so that the order they are named in changes neither the program nor its cache key.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Targets {
    Nodes(Vec<NodeId>),
    Density,
    DensityOf(BTreeSet<NodeId>),
}

impl Query {
    /// The values of `targets`, with nothing observed.
    pub fn of(targets: impl IntoIterator<Item = QDyn>) -> Query {
        Query {
            targets: Targets::Nodes(targets.into_iter().map(|q| q.id()).collect()),
            observations: Observations::new(),
        }
    }

    /// The sum of every term's log-density: the program's one target, `NodeId::DENSITY`.
    pub fn log_density() -> Query {
        Query {
            targets: Targets::Density,
            observations: Observations::new(),
        }
    }

    /// The sum of the log-densities of these terms, named by their log-density nodes.
    pub fn log_density_of<T: Into<NodeId>>(terms: impl IntoIterator<Item = T>) -> Query {
        Query {
            targets: Targets::DensityOf(terms.into_iter().map(Into::into).collect()),
            observations: Observations::new(),
        }
    }

    /// The same targets, given `observations`.
    pub fn given(self, observations: Observations) -> Query {
        Query {
            observations,
            ..self
        }
    }
}
