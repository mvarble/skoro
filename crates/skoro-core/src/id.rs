use std::sync::atomic::{AtomicU64, Ordering};

/// The identifier of a node of the compilation graph.
///
/// Identifiers come from one process-wide counter, so they are unique across models and branches,
/// and increase with time: a node that existed before another has the smaller identifier.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct NodeId(u64);

static NEXT: AtomicU64 = AtomicU64::new(0);

impl NodeId {
    /// The identifier of a handle whose operation failed inside a scope. The counter never reaches
    /// it, nor the next.
    pub(crate) const FAILED: NodeId = NodeId(u64::MAX);
    /// The identifier of a handle whose operation was attempted with no scope open.
    pub(crate) const OUTSIDE_SCOPE: NodeId = NodeId(u64::MAX - 1);

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
