//! Labelled reference edges from a graph node to others, stored beside the arena.
//!
//! A binding `(from, role, to)` means "`to` plays `role` for `from`". Bindings are not data
//! edges — the compiler never reads them — but they *are* references:
//! [`Graph::remove`](crate::graph::Graph::remove) refuses to take a bound target out from under
//! its referrer, and a collector marks a bound target reachable whenever its referrer is.

use std::collections::{HashMap, HashSet};

use skoro_core::id::Id;

/// The label on a binding. Defined in `skoro-core` so that
/// [`SkoroError`](skoro_core::error::SkoroError) can name one, and re-exported here because this
/// is the module whose edges carry it.
pub use skoro_core::role::Role;

/// The bindings of a graph: every `(from, role, to)` triple recorded so far, plus the reverse
/// index [`referrers_of`](Bindings::referrers_of) needs.
///
/// Insertion order is preserved per `from`, and a `(role, to)` pair may repeat — nothing here
/// deduplicates.
#[derive(Default, Debug, Clone)]
pub struct Bindings {
    by_source: HashMap<Id, Vec<(Role, Id)>>,
    referrers: HashMap<Id, HashSet<Id>>,
}

impl Bindings {
    /// Record `(from, role, to)`. Does not check that either id exists; the graph does that
    /// first, since only it has the arena.
    pub fn bind(&mut self, from: Id, role: Role, to: Id) {
        self.by_source.entry(from).or_default().push((role, to));
        self.referrers.entry(to).or_default().insert(from);
    }

    /// Every `(role, to)` pair `from` has bound, in binding order. Empty if it bound nothing.
    pub fn of(&self, from: Id) -> &[(Role, Id)] {
        self.by_source.get(&from).map(Vec::as_slice).unwrap_or(&[])
    }

    /// The `to`s `from` has bound under `role`, in binding order.
    pub fn bound(&self, from: Id, role: Role) -> Vec<Id> {
        self.of(from)
            .iter()
            .filter(|(r, _)| *r == role)
            .map(|(_, to)| *to)
            .collect()
    }

    /// Every `from` that has bound `to` under any role, in no particular order.
    pub fn referrers_of(&self, to: Id) -> impl Iterator<Item = Id> + '_ {
        self.referrers.get(&to).into_iter().flatten().copied()
    }

    /// Every `(from, pairs)` entry, in no particular order.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (Id, &[(Role, Id)])> + '_ {
        self.by_source
            .iter()
            .map(|(from, pairs)| (*from, pairs.as_slice()))
    }

    /// Drop every binding `from` made, fixing up the reverse index for each `to` it named.
    pub fn remove_source(&mut self, from: Id) {
        let Some(pairs) = self.by_source.remove(&from) else {
            return;
        };
        for (_, to) in pairs {
            if let Some(referrers) = self.referrers.get_mut(&to) {
                referrers.remove(&from);
                if referrers.is_empty() {
                    self.referrers.remove(&to);
                }
            }
        }
    }

    /// Drop every binding that names `to` as its target, fixing up each referrer's list.
    pub fn remove_target(&mut self, to: Id) {
        let Some(referrers) = self.referrers.remove(&to) else {
            return;
        };
        for from in referrers {
            if let Some(pairs) = self.by_source.get_mut(&from) {
                pairs.retain(|(_, t)| *t != to);
                if pairs.is_empty() {
                    self.by_source.remove(&from);
                }
            }
        }
    }

    /// Check that `referrers` is the exact reverse of `by_source`, and that neither map holds an
    /// empty entry.
    pub fn validate(&self) -> Result<(), String> {
        let mut derived: HashMap<Id, HashSet<Id>> = HashMap::new();
        for (from, pairs) in &self.by_source {
            if pairs.is_empty() {
                return Err(format!("bindings of {from:?} is empty; should be absent"));
            }
            for (_, to) in pairs {
                derived.entry(*to).or_default().insert(*from);
            }
        }
        for (to, actual) in &self.referrers {
            if actual.is_empty() {
                return Err(format!("referrers of {to:?} is empty; should be absent"));
            }
            match derived.get(to) {
                Some(expected) if expected == actual => {}
                Some(expected) => {
                    return Err(format!(
                        "referrers of {to:?} is {actual:?}, but the bindings give {expected:?}"
                    ));
                }
                None => {
                    return Err(format!(
                        "referrers of {to:?} is {actual:?}, but no binding targets it"
                    ));
                }
            }
        }
        if let Some((to, expected)) = derived
            .iter()
            .find(|(to, _)| !self.referrers.contains_key(to))
        {
            return Err(format!(
                "referrers has no entry for {to:?}, but {expected:?} bind it"
            ));
        }
        Ok(())
    }
}
