#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "identifiers are made by the model, which the building tasks add"
    )
)]

use std::sync::atomic::{AtomicU64, Ordering};

/// The identifier of a node of the compilation graph.
///
/// Identifiers come from one process-wide counter, so they are unique across models and branches,
/// and increase with time: a node that existed before another has the smaller identifier.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct NodeId(u64);

static NEXT: AtomicU64 = AtomicU64::new(0);

impl NodeId {
    /// A new identifier, greater than every one made before it.
    pub(crate) fn fresh() -> NodeId {
        NodeId(NEXT.fetch_add(1, Ordering::Relaxed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifiers_increase_across_threads() {
        let first = NodeId::fresh();
        let others: Vec<NodeId> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..4)
                .map(|_| scope.spawn(|| (0..100).map(|_| NodeId::fresh()).collect::<Vec<_>>()))
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().unwrap())
                .collect()
        });
        let mut sorted = others.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), others.len());
        assert!(others.iter().all(|&id| id > first));
        assert!(NodeId::fresh() > *sorted.last().unwrap());
    }
}
