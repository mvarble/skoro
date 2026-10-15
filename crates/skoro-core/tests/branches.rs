#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! Snapshots and branches: what a fork shares, and what it does not.

use std::sync::Arc;

use skoro_core::primitive::Add;
use skoro_core::scope::{apply, quantity};
use skoro_core::{Error, Operation, QDyn, Real, Writer};

#[track_caller]
fn add(a: impl Into<QDyn>, b: impl Into<QDyn>) -> QDyn {
    apply(Arc::new(Add), &[a.into(), b.into()])
}

#[test]
fn handle_is_valid_on_every_branch_that_shares_it() {
    let mut main = Writer::new();
    let x = main.build(|| quantity("x", Real)).unwrap();
    let mut branch = main.snapshot().fork();
    let on_branch = branch.build(|| add(x, x)).unwrap();
    let on_main = main.build(|| add(x, x)).unwrap();
    assert!(!on_branch.is_failed() && !on_main.is_failed());
    assert_ne!(on_branch.id(), on_main.id());
}

#[test]
fn handle_from_another_model_fails() {
    let mut first = Writer::new();
    let x = first.build(|| quantity("x", Real)).unwrap();
    let mut second = Writer::new();
    let failure = second.build(|| add(x, x)).unwrap_err();
    assert_eq!(failure.error, Error::UnknownNode { id: x.id() });

    // A node made on a branch after the fork is unknown to the main line.
    let mut branch = first.snapshot().fork();
    let y = branch.build(|| quantity("y", Real)).unwrap();
    let failure = first.build(|| add(x, y)).unwrap_err();
    assert_eq!(failure.error, Error::UnknownNode { id: y.id() });
}

#[test]
fn branches_do_not_see_each_other() {
    let mut main = Writer::new();
    main.build(|| quantity("x", Real)).unwrap();
    let mut branch = main.snapshot().fork();
    let (main_reader, branch_reader) = (main.reader(), branch.reader());
    let a = main.build(|| quantity("a", Real)).unwrap();
    let b = branch.build(|| quantity("b", Real)).unwrap();
    let (main, branch) = (main_reader.latest(), branch_reader.latest());
    assert!(main.node(a.id()).is_some() && main.node(b.id()).is_none());
    assert!(branch.node(b.id()).is_some() && branch.node(a.id()).is_none());
    // Each branch may use an address the other took.
    let mut other = branch.fork();
    assert!(other.build(|| quantity("a", Real)).is_ok());
}

#[test]
fn fork_copies_nothing_it_does_not_change() {
    let mut main = Writer::new();
    let nodes = main
        .build(|| {
            (0..50usize)
                .map(|i| quantity(i, Real).id())
                .collect::<Vec<_>>()
        })
        .unwrap();
    let mut branch = main.snapshot().fork();
    branch.build(|| quantity("more", Real)).unwrap();
    let (main, branch) = (main.snapshot(), branch.snapshot());
    for id in nodes {
        let (a, b) = (main.node(id).unwrap(), branch.node(id).unwrap());
        assert!(
            std::ptr::eq(a, b),
            "node {id:?} is the same allocation on both branches"
        );
    }
}

#[test]
fn dropping_the_last_snapshot_frees_retired_nodes() {
    // A node lives as long as a snapshot holds it, and no longer: the branch's node holds the
    // only strong reference to its operation once the test lets its own go.
    let mut main = Writer::new();
    let x = main.build(|| quantity("x", Real)).unwrap();
    let operation: Arc<dyn Operation> = Arc::new(Add);
    let weak = Arc::downgrade(&operation);
    let mut branch = main.snapshot().fork();
    branch
        .build(move || apply(operation, &[x.into(), x.into()]))
        .unwrap();
    let held = branch.snapshot();
    drop(branch);
    assert!(weak.upgrade().is_some(), "a snapshot still holds the node");
    drop(held);
    assert!(
        weak.upgrade().is_none(),
        "no snapshot holds the node, and it is freed"
    );
    assert!(main.snapshot().node(x.id()).is_some());
}
