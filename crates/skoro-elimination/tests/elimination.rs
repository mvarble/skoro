#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! Elimination in square-root form, against the examples and statements of the note.

use std::sync::Arc;

use skoro_elimination::{Schedule, Structure};

/// The note's star: a hub, block 0, joined to each of `n` leaves by a factor of its own.
fn star(n: usize) -> Arc<Structure> {
    let factors = (1..=n).map(|leaf| vec![0, leaf]).collect();
    Arc::new(Structure::new(vec![2; n + 1], factors).unwrap())
}

/// The note's map: poses `x1, x2, x3` (blocks 0 to 2) and landmarks `l1, l2` (blocks 3 and 4).
/// A prior on `x1`, odometry between consecutive poses, a loop closure from `x1` to `x3`, `l1`
/// seen from `x1` and `x2`, and `l2` from `x2` and `x3`.
fn map() -> Arc<Structure> {
    let factors = vec![
        vec![0],
        vec![0, 1],
        vec![1, 2],
        vec![0, 2],
        vec![0, 3],
        vec![1, 3],
        vec![1, 4],
        vec![2, 4],
    ];
    Arc::new(Structure::new(vec![3, 3, 3, 2, 2], factors).unwrap())
}

#[test]
fn the_star_eliminates_its_leaves_first() {
    let n = 6;
    let structure = star(n);
    let schedule = Schedule::all(&structure).unwrap();
    assert_eq!(schedule.order().len(), n + 1);
    // The hub comes after the leaves, but for one it may share the last problem with: once the
    // others are gone, the hub and one leaf are a problem of two blocks either way.
    let hub = schedule.order().iter().position(|&k| k == 0).unwrap();
    assert!(
        hub >= n - 1,
        "the hub after the leaves: {:?}",
        schedule.order()
    );
    // No fill: every separator is at most one block.
    for i in 0..schedule.cliques() {
        assert!(
            schedule.separator(i).len() <= 1,
            "{:?}",
            schedule.separator(i)
        );
    }
    // The hub first instead leaves a factor over every leaf, and everything after it is one dense
    // front of every block.
    let hub_first = Schedule::in_order(&structure, (0..=n).collect()).unwrap();
    assert_eq!(hub_first.cliques(), 1);
    assert_eq!(hub_first.frontals(0), (0..=n).collect::<Vec<_>>());
}

#[test]
fn the_map_has_three_cliques() {
    let structure = map();
    let schedule = Schedule::in_order(&structure, vec![3, 4, 0, 1, 2]).unwrap();
    assert_eq!(schedule.cliques(), 3);
    let cliques: Vec<(Vec<usize>, Vec<usize>)> = (0..3)
        .map(|i| {
            (
                schedule.frontals(i).to_vec(),
                schedule.separator(i).to_vec(),
            )
        })
        .collect();
    assert!(cliques.contains(&(vec![3], vec![0, 1])), "{cliques:?}");
    assert!(cliques.contains(&(vec![4], vec![1, 2])), "{cliques:?}");
    assert_eq!(cliques[2], (vec![0, 1, 2], vec![]), "the root, last");
    assert!(schedule.kept().is_empty());
}
