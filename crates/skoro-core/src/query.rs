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

static IDENTITIES: AtomicU64 = AtomicU64::new(0);

fn fresh() -> u64 {
    IDENTITIES.fetch_add(1, Ordering::Relaxed)
}

impl Observations {
    pub fn new() -> Observations {
        Observations {
            identity: fresh(),
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
    pub(crate) targets: Vec<NodeId>,
    pub(crate) observations: Observations,
}

impl Query {
    /// The values of `targets`, with nothing observed.
    pub fn of(targets: impl IntoIterator<Item = QDyn>) -> Query {
        Query {
            targets: targets.into_iter().map(|q| q.id()).collect(),
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
