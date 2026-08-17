//! The garbage collector's bookkeeping: which ids are still named by a live [`Q`], and
//! which dropped handles are waiting to be examined.
//!
//! The collector this feeds is deliberately dumb about the past. A guard registers its id
//! when it is created and reports its drop, and [`ModelServer::cleanup`] re-derives
//! eligibility from the graph's *current* state — kind, children, pins, model ownership —
//! rather than trusting anything that was true when the handle died. That is what makes
//! collecting an already-collected node, or skipping a node that has since gained a child,
//! ordinary rather than an internal error.
//!
//! [`Q`]: crate::q::Q
//! [`ModelServer::cleanup`]: crate::model::ModelServer::cleanup

use crossbeam_queue::SegQueue;
use dashmap::DashSet;
use skoro_core::id::Id;

/// The shared end of the handle registry.
///
/// One per server, cloned into every `NodeDropGuard` so a guard can outlive its server
/// without consequence — deregistering into a dead registry is harmless.
#[derive(Debug, Default)]
pub(crate) struct CleanupRegistry {
    /// Ids whose last handle dropped since the last collection, plus ids a mutation may
    /// have orphaned. Candidates only: each *may* be collectable, decided fresh at
    /// collection time, so a stale or duplicated entry costs one failed check.
    ///
    /// **Popping is destructive, so a "not yet" verdict is final.** A candidate the
    /// collector declines — because it still has a child, or a certificate still names it
    /// — is gone from here, and nothing re-derives it. Every mutation that reduces a
    /// node's reachability must therefore [`offer`](Self::offer) what it orphaned; a
    /// severed edge or a dropped record that stays silent is a permanent leak, not a
    /// delayed collection.
    candidates: SegQueue<Id>,
    /// The ids a live handle names.
    ///
    /// A set rather than a count, because a guard is 1:1 with a node: `NodeDropGuard::new`
    /// runs once per id, beside the node it names, and `Q` clones share the one `Arc`. A
    /// counter here would describe a shape — several guards for one id — that cannot occur.
    live: DashSet<Id>,
}

impl CleanupRegistry {
    /// A new handle names `id`.
    pub(crate) fn track(&self, id: Id) {
        self.live.insert(id);
    }

    /// The guard for `id` is gone; offer the id for collection.
    pub(crate) fn untrack(&self, id: Id) {
        self.live.remove(&id);
        self.candidates.push(id);
    }

    /// Whether a handle still names `id`.
    pub(crate) fn is_tracked(&self, id: Id) -> bool {
        self.live.contains(&id)
    }

    /// The next candidate awaiting examination, if any.
    pub(crate) fn pop(&self) -> Option<Id> {
        self.candidates.pop()
    }

    /// Offer `ids` as candidates — for nodes a mutation may have orphaned, whose handles
    /// are long gone and which would otherwise never be examined again. See the note on
    /// [`candidates`](Self::candidates) for why this is obligatory rather than an
    /// optimization.
    pub(crate) fn offer(&self, ids: impl IntoIterator<Item = Id>) {
        for id in ids {
            self.candidates.push(id);
        }
    }
}
