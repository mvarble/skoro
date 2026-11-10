use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use rpds::HashTrieMapSync;
use skoro_tensor::HostValue;

use crate::{IntoHostValue, NodeId, Q, QDyn, Snapshot, Typed};

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

    /// The observations the query is given.
    pub fn observations(&self) -> &Observations {
        &self.observations
    }

    /// The terms whose log-densities a density query sums, in the order they were added: every
    /// term of `snapshot` for `log_density()`, the named ones for `log_density_of`; `None` for a
    /// query of nodes.
    pub fn terms(&self, snapshot: &Snapshot) -> Option<Vec<NodeId>> {
        match &self.targets {
            Targets::Nodes(_) => None,
            Targets::Density => Some(snapshot.0.terms.keys().copied().collect()),
            Targets::DensityOf(terms) => Some(terms.iter().copied().collect()),
        }
    }
}

impl Observations {
    /// This set with the values of `other` for the quantities this set does not observe: a new
    /// set. Where both observe a quantity, this set's value is kept. When `other` adds nothing,
    /// the set is this one, of the same identity.
    pub fn union(self, other: &Observations) -> Observations {
        let mut values = (*self.values).clone();
        let mut added = false;
        for (id, value) in other.values.iter() {
            if !values.contains_key(id) {
                values.insert_mut(*id, value.clone());
                added = true;
            }
        }
        if !added {
            return self;
        }
        Observations {
            identity: fresh(),
            values: Arc::new(values),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value(x: f64) -> HostValue {
        HostValue::new(None, 1, skoro_tensor::HostData::Continuous(vec![x])).unwrap()
    }

    #[test]
    fn union_keeps_this_sets_values() {
        let (a, b, c) = (
            QDyn::from_id(NodeId::fresh()),
            QDyn::from_id(NodeId::fresh()),
            QDyn::from_id(NodeId::fresh()),
        );
        let this = Observations::new()
            .observe_dyn(a, value(1.0))
            .observe_dyn(b, value(2.0));
        let other = Observations::new()
            .observe_dyn(b, value(20.0))
            .observe_dyn(c, value(30.0));
        let identity = this.identity();
        let joined = this.union(&other);
        assert_ne!(joined.identity(), identity);
        assert_eq!(joined.get(a.id()), Some(Some(&value(1.0))));
        assert_eq!(
            joined.get(b.id()),
            Some(Some(&value(2.0))),
            "this set's value is kept"
        );
        assert_eq!(joined.get(c.id()), Some(Some(&value(30.0))));
        let same = joined
            .clone()
            .union(&Observations::new().observe_dyn(a, value(5.0)));
        assert_eq!(
            same.identity(),
            joined.identity(),
            "nothing added, the same set"
        );
    }
}
