//! The garbage collector's root set: which ids are still named by a live [`Q`].
//!
//! The collector this feeds is a mark-and-sweep over the graph's *current* shape. It asks
//! one question — what is reachable from a root — and everything else follows, so there is
//! nothing here but the roots and a flag saying whether anything has changed since the last
//! sweep.
//!
//! **This is deliberately less machinery than it replaces.** The previous collector kept a
//! queue of candidate ids and popped them destructively, which made a "not yet" verdict
//! final: every mutation that reduced a node's reachability had to *offer* what it orphaned,
//! or the node was unreachable and uncollected forever. That obligation was documented,
//! spread across five call sites, and enforced by nothing. Reachability is not an event, so
//! nothing has to report it.
//!
//! [`Q`]: crate::q::Q

use std::sync::atomic::{AtomicBool, Ordering};

use dashmap::DashSet;
use skoro_core::id::Id;

/// The shared end of the handle registry.
///
/// One per server, cloned into every `NodeDropGuard` so a guard can outlive its server
/// without consequence — deregistering into a dead registry is harmless.
#[derive(Debug, Default)]
pub(crate) struct HandleRegistry {
    /// The ids a live handle names — the collector's roots, beside the model's own.
    ///
    /// A set rather than a count, because a guard is 1:1 with a node: `NodeDropGuard::new`
    /// runs once per id, beside the node it names, and `Q` clones share the one `Arc`. A
    /// counter here would describe a shape — several guards for one id — that cannot occur.
    live: DashSet<Id>,
    /// Whether anything has happened that could make a node unreachable.
    ///
    /// Purely an optimization, and one that can only ever cause *extra* work: a sweep it
    /// skips is a sweep that would have found nothing. It is set by a dropped handle and by
    /// every mutation that reaches [`ModelServer::cleanup`](crate::model::ModelServer::cleanup),
    /// and cleared by the sweep itself.
    dirty: AtomicBool,
}

impl HandleRegistry {
    /// A new handle names `id`.
    pub(crate) fn track(&self, id: Id) {
        self.live.insert(id);
    }

    /// The guard for `id` is gone. The node may now be unreachable — the next sweep decides,
    /// from the graph rather than from this.
    pub(crate) fn untrack(&self, id: Id) {
        self.live.remove(&id);
        self.mark_dirty();
    }

    /// Note that something may have become unreachable.
    pub(crate) fn mark_dirty(&self) {
        self.dirty.store(true, Ordering::Release);
    }

    /// Whether a sweep is worth running, clearing the flag if so.
    pub(crate) fn take_dirty(&self) -> bool {
        self.dirty.swap(false, Ordering::AcqRel)
    }

    /// The ids a live handle names, as a snapshot.
    pub(crate) fn roots(&self) -> Vec<Id> {
        self.live.iter().map(|id| *id).collect()
    }
}
