//! Labelled reference edges from a graph node to others, stored beside the arena.
//!
//! A binding `(from, role, to)` means "`to` plays `role` for `from`". Bindings are not data
//! edges — the compiler never reads them (R10) — but they *are* references:
//! [`Graph::remove`](crate::graph::Graph::remove) refuses to take a bound target out from under
//! its referrer, and the collector marks a bound target reachable whenever its referrer is.

use std::collections::{HashMap, HashSet};

use skoro_core::id::Id;

/// A labelled edge kind. Two roles compare equal iff their strings are identical.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Role(pub &'static str);

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// The bindings of a graph: every `(from, role, to)` triple recorded so far, plus the reverse
/// index [`referrers_of`](Bindings::referrers_of) needs.
///
/// Insertion order is preserved per `from`, and a `(role, to)` pair may repeat — nothing here
/// deduplicates, because a term legitimately reads the same argument under the same role twice
/// (e.g. a stacked-tangent factor over a repeated quantity).
#[derive(Default, Debug, Clone)]
pub struct Bindings {
    by_source: HashMap<Id, Vec<(Role, Id)>>,
    referrers: HashMap<Id, HashSet<Id>>,
}

impl Bindings {
    /// Record `(from, role, to)`. Does not check that either id exists — callers (`Graph::bind`)
    /// do that first, since only they have the arena.
    pub fn bind(&mut self, from: Id, role: Role, to: Id) {
        self.by_source.entry(from).or_default().push((role, to));
        self.referrers.entry(to).or_default().insert(from);
    }

    /// Every `(role, to)` pair `from` has bound, in binding order. Empty if `from` has bound
    /// nothing.
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
        self.referrers
            .get(&to)
            .into_iter()
            .flat_map(|set| set.iter().copied())
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

    /// Drop every binding that named `to` as its target, fixing up every referrer's forward
    /// list.
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

    /// Every `(from, [(role, to), ..])` pair, for validation only.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (Id, &[(Role, Id)])> + '_ {
        self.by_source
            .iter()
            .map(|(from, pairs)| (*from, pairs.as_slice()))
    }

    /// `referrers` is the exact reverse of `by_source`, and neither map holds an empty value
    /// set/list.
    pub fn validate(&self) -> Result<(), String> {
        let mut derived: HashMap<Id, HashSet<Id>> = HashMap::new();
        for (from, pairs) in &self.by_source {
            if pairs.is_empty() {
                return Err(format!("by_source[{from:?}] is empty; should be absent"));
            }
            for (_, to) in pairs {
                derived.entry(*to).or_default().insert(*from);
            }
        }
        for (to, actual) in &self.referrers {
            if actual.is_empty() {
                return Err(format!("referrers[{to:?}] is empty; should be absent"));
            }
            match derived.get(to) {
                Some(expected) if expected == actual => {}
                Some(expected) => {
                    return Err(format!(
                        "referrers[{to:?}] is {actual:?}, but by_source gives {expected:?}"
                    ));
                }
                None => {
                    return Err(format!(
                        "referrers[{to:?}] is {actual:?}, but no binding targets {to:?}"
                    ));
                }
            }
        }
        for (to, expected) in &derived {
            if !self.referrers.contains_key(to) {
                return Err(format!(
                    "referrers has no entry for {to:?}, but {expected:?} bind it"
                ));
            }
        }
        Ok(())
    }
}
