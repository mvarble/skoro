//! `Id` and its generator.
//!
//! An `Id` is deliberately unforgeable: the tuple field is private, so the only way to obtain one
//! is from a generator. That is what lets a graph treat "I hold an `Id`" as evidence that the
//! node once existed.

use std::collections::{HashMap, HashSet};

use skoro_core::id::{Id, IdGenerator};

#[test]
fn ids_are_issued_in_order_from_zero() {
    let generator = IdGenerator::new();
    let ids: Vec<Id> = (0..5).map(|_| generator.issue_id()).collect();

    for (i, id) in ids.iter().enumerate() {
        assert_eq!(**id, i as u64, "ids count up from zero");
    }
    // Strictly increasing, which is what makes a parent always older than its child -- and
    // therefore what makes a cycle unconstructible in the graph.
    assert!(ids.windows(2).all(|pair| pair[0] < pair[1]));
}

#[test]
fn a_default_generator_starts_where_a_new_one_does() {
    let generator = IdGenerator::default();
    assert_eq!(*generator.issue_id(), 0);
}

/// Issuing does not need `&mut`, so a generator can sit behind a shared reference.
#[test]
fn issuing_takes_a_shared_reference() {
    let generator = IdGenerator::new();
    let issue = |g: &IdGenerator| g.issue_id();
    assert_eq!(*issue(&generator), 0);
    assert_eq!(*issue(&generator), 1);
}

/// Cloning snapshots the counter rather than sharing it.
///
/// The two generators then issue the *same* ids independently, which is why a clone belongs to a
/// cloned graph and must never be used alongside the original.
#[test]
fn cloning_snapshots_the_counter() {
    let generator = IdGenerator::new();
    let _ = generator.issue_id();
    let _ = generator.issue_id();

    let copy = generator.clone();
    assert_eq!(*copy.issue_id(), 2, "the clone resumes from the snapshot");
    assert_eq!(*generator.issue_id(), 2, "and the original is unaffected");
}

/// The iterator interface is an endless stream of fresh ids.
#[test]
fn the_iterator_yields_fresh_ids() {
    let mut generator = IdGenerator::new();
    let ids: Vec<Id> = generator.iter().take(4).collect();
    assert_eq!(
        ids.iter().map(|id| **id).collect::<Vec<_>>(),
        vec![0, 1, 2, 3]
    );

    // It picks up where the borrow left off.
    assert_eq!(*generator.issue_id(), 4);
    assert!(format!("{:?}", generator.iter()).contains("IdGeneratorIter"));
}

/// `Id` is a map key and a set member throughout the workspace, so all four derived traits are
/// load-bearing.
#[test]
fn ids_hash_order_and_compare() {
    let generator = IdGenerator::new();
    let a = generator.issue_id();
    let b = generator.issue_id();

    assert_eq!(a, a);
    assert_ne!(a, b);
    assert!(a < b);
    assert_eq!(a.cmp(&b), std::cmp::Ordering::Less);
    assert_eq!(a.partial_cmp(&b), Some(std::cmp::Ordering::Less));

    let mut set = HashSet::new();
    assert!(set.insert(a));
    assert!(!set.insert(a), "an id hashes to itself");
    assert!(set.insert(b));

    let mut map = HashMap::new();
    map.insert(a, "first");
    map.insert(b, "second");
    assert_eq!(map[&a], "first");

    // `Copy`, so passing one around does not move it.
    let copy = a;
    assert_eq!(copy, a);

    assert!(format!("{a:?}").contains("Id"));
}

/// Two generators issue overlapping ids, which is why an `Id` only means anything relative to the
/// graph that issued it.
#[test]
fn separate_generators_collide() {
    assert_eq!(IdGenerator::new().issue_id(), IdGenerator::new().issue_id());
}
