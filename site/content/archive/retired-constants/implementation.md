---
summary: A retirement frees, in the same change, every node it leaves needed by nothing, by reference counting from its candidates over an index of the nodes the model holds outside the compilation graph's edges; the user pins and unpins nodes in a scope to keep them.
depends_on: [archive/retired-constants/math, archive/core/implementation]
---

## Overview

When this is built, `Retirement::replace` does one more thing before it publishes: it frees the
nodes the retirement leaves needed by nothing, the set $F$ of
[](eq:archive/retired-constants/math/freed), found by reference counting from the candidates
[](eq:archive/retired-constants/math/candidates). Nothing in the calling code of a retirement
changes. A model that runs indefinitely stops growing: its size is bounded by what its current
quantities, terms, samplers, proposals and pins need, by
[%full](statement:archive/retired-constants/math/freeing).

Two functions are new in `skoro::scope`, `pin` and `unpin`, for the nodes a user wants to keep
whether or not anything uses them, as
[Handles the user holds](math.svx#handles-the-user-holds) decided. The case
they exist for is a constant made at configuration time and used by every tick:

```rust
use skoro::scope::{constant, pin, unpin};

let mut window = Writer::new();
// The measurement noise, made once and kept for the run.
let noise = window.build(|| pin(constant(Real, 0.1)))?;
for k in 0.. {
    // Retiring before adding: the noise may have no consumer for a moment, and is kept.
    if k >= W {
        let retirement = window.retire(Addresses::new().under(tick_address(k - W)), observed(k - W))?;
        retirement.replace(|plan| replacement(plan))?;
    }
    window.build(|| tick(k, noise))?;
}
// Done with it: the next retirement frees it if nothing uses it.
window.build(|| unpin(noise))?;
```

Without the pin, the first retirement that removes the last term reading `noise` frees it, and the
next tick's `build` fails with `UnknownNode`, whose message now says that a retirement may have
freed the node and that pinning keeps it.

## Constraints

Read at commit `2d788a5`.

**The model** (`crates/skoro-core/src/model.rs`) is a set of persistent maps. Freeing must keep
every one of them consistent, and the new state goes beside them:

```rust
pub(crate) struct Model {
    pub(crate) nodes: HashTrieMapSync<NodeId, Arc<Node>>,
    /// The compilation graph's edges read forward: each node's consumers.
    pub(crate) consumers: HashTrieMapSync<NodeId, HashTrieSetSync<NodeId>>,
    pub(crate) addresses: RedBlackTreeMapSync<Address, NodeId>,
    pub(crate) terms: RedBlackTreeMapSync<NodeId, Arc<Term>>,
    pub(crate) scoring: HashTrieMapSync<NodeId, NodeId>,
    pub(crate) joined: HashTrieMapSync<NodeId, HashTrieSetSync<NodeId>>,
    pub(crate) proposals: HashTrieMapSync<NodeId, Arc<Proposal>>,
    pub(crate) programs: Mutex<Vec<(ProgramKey, Arc<Program>)>>,
}
```

`consumers` is exactly what reference counting needs: a node with no entry, or an empty one, has
no consumer. A clone of `Model` drops `programs`, so no program compiled before a retirement is
run against the model after it.

**What a model holds outside the edges** is spread over three places, none of them indexed by the
node held (`crates/skoro-core/src/term.rs`):

```rust
pub struct Term {
    pub log_density: NodeId,
    pub scores: Option<NodeId>,
    pub involves: Box<[NodeId]>,
    pub sampler: Option<Sampler>,
    pub(crate) bindings: Vec<(&'static str, NodeId)>,
}

pub struct Sampler {
    pub sample: Arc<dyn Sample>,
    pub inputs: Vec<QDyn>,
    pub log_density: Option<Q<Real>>,
}

pub struct Proposal {
    pub sample: Arc<dyn Sample>,
    pub inputs: Vec<QDyn>,
    pub log_density: Q<Real>,
}
```

A term's log-density is found by `terms.contains_key`. Its bindings, its sampler's nodes and a
proposal's nodes are not: asking "does anything hold node $n$" means scanning every term and
proposal. The design adds an index so that the question costs one lookup ([Design](#design)).
Bindings are added by `TermHandle::bind` and by `Retirement::replace`; a sampler is set by
`TermHandle::sampler`, which may replace an earlier one; proposals are added by
`Draft::add_proposal` and removed by retirement's `remove`.

**Retirement** (`crates/skoro-core/src/retire.rs`) builds the replacement in a scope over a
`Draft` of the snapshot, checks it, calls `remove`, adds the factor, its bindings and its
proposals, and publishes once:

```rust
// 2. The removal.
remove(&mut draft.model, &plan, &retired, &unproposed);

// 3. The replacement, its bindings, and its proposals.
let factor = draft.add_term(None, log_density).map_err(fail)?;
// ... bindings extended, proposals added ...
writer.publish(draft.model);
```

Freeing goes between the last line of step 3 and `publish`, against the model after both, as the
mathematics requires. `remove` already visits every removed node and every input of one, which
are most of the candidates; it is the natural place to collect them. A retirement of nothing
returns before the scope is opened, and so frees nothing.

**Scopes** (`crates/skoro-core/src/scope.rs`) create nodes through three `Draft` methods
(`mapped`, `constant`, `quantity`), each calling `self.model.insert`. The nodes created in the
retirement's scope, $S$, are the nodes those calls insert between the opening of the scope and its
close. Identifiers increase with time, but the model's maps are not ordered by identifier, so $S$
cannot be read off the model cheaply; it is recorded as it is made. Builders are free functions
with `#[track_caller]`, running a step on the innermost draft through `scope::build`, which
records the first failure and returns a failed handle; `pin` and `unpin` follow the same
pattern. A composite's definition is a `Draft` with `in_model == false`, in which model-level
operations are refused with `NotInAModel`.

**Handles** (`crates/skoro-core/src/q.rs`) are `Copy` identifiers bound to no writer, and stay so.
Identifiers are never reused (`crates/skoro-core/src/id.rs`), so a handle to a freed node fails
with `UnknownNode` and never names another node:

```rust
/// A handle the model does not hold: from another model, or a retired quantity.
UnknownNode { id: NodeId },
```

**The existing retirement tests** (`crates/skoro/tests/retirement.rs`) assert today that the
constant `a` of the test chain stays after the first tick is retired:

```rust
for id in [x1.id(), chain.z1.id(), chain.kept.id(), chain.a.id(), chain.terms[3]] {
    assert!(new.node(id).is_some(), "{id:?} stays");
}
```

`a` is read only by the term of $\randomPoint_0$ and its sampler, both removed, so it is now
freed. That assertion moves to the list of nodes removed. `kept`, the value $2\point_1$ that no
term reads, stays: it is no candidate, which is the locality of
[](eq:archive/retired-constants/math/freed).

**The archive** reclaimed only *private* nodes, those the graph built for itself and never handed
out, by rescanning all of them until none was childless (`remove_closure` and
`reclaim_private_nodes` in `crates/skoro-graph/src/graph.rs` at `archive-2026-10-09`). Not kept:
the restriction to private nodes, since the constants that grow here are the user's, and the
rescanning, which is quadratic in the worst case where reference counting is linear in what it
visits. Kept: the archive's `validate`, which recomputed its derived indices from the nodes and
reported the first disagreement; the held index here has the same check
(`held_index_matches_the_terms`).

## Design

Everything is in `skoro-core`. No other crate changes, except the tests and the example.

### The model's new state

```rust
// crates/skoro-core/src/model.rs
pub(crate) struct Model {
    // ... as now ...
    /// `H(M)` but the terms' log-densities: how many times each node is named by a binding, a
    /// sampler's inputs or log-density, or a proposal's inputs or log-density. A node with no
    /// entry is named by none.
    pub(crate) held: HashTrieMapSync<NodeId, u32>,
    /// `P(M)`: the pinned nodes.
    pub(crate) pinned: HashTrieSetSync<NodeId>,
    /// Pinned nodes a retirement would have freed: when one is unpinned it is among the next
    /// retirement's candidates, `U` of the candidates.
    pub(crate) spared: HashTrieSetSync<NodeId>,
}

impl Model {
    pub(crate) fn hold(&mut self, id: NodeId);     // count + 1
    pub(crate) fn release(&mut self, id: NodeId);  // count - 1, the entry removed at 0
    /// Whether the model keeps `id` outside the compilation graph's edges: `K(M)`.
    pub(crate) fn keeps(&self, id: NodeId) -> bool {
        self.terms.contains_key(&id) || self.held.contains_key(&id) || self.pinned.contains(&id)
    }
}
```

The three are persistent maps like the others, so a snapshot carries its pins, and a fork
inherits its parent's pins and changes its own afterwards. `Clone for Model` clones them.

The index is a count, not a set of holders, because a node may be named several times by one term
(two roles bound to one scale) and by several terms at once, and each naming is released
separately. The alternative, scanning every term and proposal at each retirement to compute
$H(M')$, needs no maintenance but makes every retirement linear in the model, where the core's
removal is linear in what it removes.

Every place that names a node outside the edges holds it, and every place that stops naming one
releases it:

| where | holds | releases |
| --- | --- | --- |
| `TermHandle::bind` | the bound node | |
| `TermHandle::sampler` | the new sampler's inputs and log-density | the old sampler's, if the term had one |
| `Retirement::replace`, step 3 | the factor's bindings | |
| `Draft::add_proposal` | the proposal's inputs and log-density | the replaced proposal's, if the parameter had one |
| `remove`, for each term of $\mathcal T_E$ | | its bindings and its sampler's nodes |
| `remove`, for each proposal removed | | its inputs and log-density |

Two helpers keep these symmetric: `fn named(term: &Term) -> impl Iterator<Item = NodeId>` (the
bindings and the sampler's nodes) and `fn named_by(proposal: &Proposal) -> impl Iterator<Item = NodeId>`,
used by both the holding and the releasing side and by the index's test.

### Recording the scope's nodes

```rust
// crates/skoro-core/src/scope.rs
pub(crate) struct Draft {
    pub(crate) model: Model,
    pub(crate) failure: Option<Failure>,
    in_model: bool,
    /// The nodes this draft inserted, in order: `S` when the draft is a retirement's scope.
    pub(crate) created: Vec<NodeId>,
}

impl Draft {
    /// Inserts `node` and records it in `created`.
    fn insert(&mut self, node: Node);
}
```

`mapped`, `constant` and `quantity` call `self.insert` in place of `self.model.insert`. The
`Draft` of `Writer::build` records too, and drops the list with the draft; the cost is one push per
node made. The alternative, a watermark identifier and a scan of the draft for larger
identifiers, costs a scan of the whole model at every retirement.

### Pinning

```rust
// crates/skoro-core/src/scope.rs, re-exported as skoro::scope::{pin, unpin}

/// Keeps `node` through every retirement that does not remove what it is reached from, whether
/// or not anything uses it (`UnknownNode`, `FailedOperand`, `OutsideScope`, `NotInAModel`).
/// Pinning a pinned node changes nothing.
#[track_caller]
pub fn pin<H: Into<QDyn> + Copy>(node: H) -> H;

/// Releases a pin: the node is freed by the next retirement if nothing else needs it. Unpinning a
/// node that is not pinned changes nothing.
#[track_caller]
pub fn unpin<H: Into<QDyn> + Copy>(node: H) -> H;
```

Each runs a step on the innermost draft that checks the handle with `draft.spaces(&[node])`
(which gives `OutsideScope`, `FailedOperand` and `UnknownNode`), refuses a definition's draft
with `NotInAModel`, and inserts into or removes from `model.pinned`. They return their argument,
so that `let noise = pin(constant(Real, 0.1));` reads as one statement. A quantity may be pinned;
it changes nothing, since no quantity is freed, and the pin goes when the quantity is retired.

`Snapshot` gains two readers:

```rust
impl Snapshot {
    /// Whether `id` is pinned.
    pub fn is_pinned(&self, id: impl Into<NodeId>) -> bool;
    /// Every node of the compilation graph, sorted: the order they were made in.
    pub fn nodes(&self) -> Vec<NodeId>;
}
```

`nodes` exists for the tests, which check what is left after a retirement against
[%full](statement:archive/retired-constants/math/freeing), and for any reader that reports a
model's contents.

Releasing by a guard dropped at the end of a lexical scope, considered in the mathematics'
discussion, is not built. A guard dropped has no writer to change: a pin is part of a branch's
state, published in its snapshots, while a guard is a value of the program, and nothing says which
branch its drop should change. An explicit `unpin` in a scope says exactly that.

### Freeing

`remove` returns the candidates it finds; freeing runs after the replacement is added.

```rust
// crates/skoro-core/src/retire.rs

/// Removes `R` as now, releases the held nodes of the terms and proposals it drops, drops the
/// pins and spared entries of removed nodes, and returns the candidates it found:
/// `in(R) \ R` and `H(M) \ H(M') \ R`, before the removal decides which survive.
fn remove(model: &mut Model, plan: &Plan, retired: &HashSet<NodeId>, unproposed: &HashSet<NodeId>)
    -> Vec<NodeId>;

/// Frees `F` of [](eq:archive/retired-constants/math/freed), by reference counting from
/// `candidates`: the proof's part (3).
fn free(model: &mut Model, candidates: Vec<NodeId>) {
    let mut stack = candidates;
    while let Some(id) = stack.pop() {
        let Some(node) = model.nodes.get(&id).cloned() else {
            continue; // removed, or already freed
        };
        let consumed = model.consumers.get(&id).is_some_and(|c| !c.is_empty());
        if node.is_quantity() || consumed || model.terms.contains_key(&id) || model.held.contains_key(&id) {
            continue;
        }
        if model.pinned.contains(&id) {
            model.spared.insert_mut(id);
            continue;
        }
        for input in node.inputs() {
            // drop `id` from `consumers[input]`, removing the entry when it empties
            stack.push(*input);
        }
        model.consumers.remove_mut(&id);
        model.nodes.remove_mut(&id);
    }
}
```

In `Retirement::replace`, step 3 ends with

```rust
let mut candidates = removal;                   // from `remove`
candidates.extend(draft.created.iter().copied()); // S
let released: Vec<NodeId> = draft.model.spared.iter()
    .filter(|id| !draft.model.pinned.contains(*id)).copied().collect();
for id in &released { draft.model.spared.remove_mut(id); }
candidates.extend(released);                    // U
free(&mut draft.model, candidates);
writer.publish(draft.model);
```

A node is examined each time it is pushed, and pushed once as a candidate and once for each of
its consumers that is freed; the work is linear in the candidates and the edges into the nodes
freed. A node visited while it still has a consumer is skipped, and visited again when its last
consumer is freed, which is the reference counting of the proof; since the compilation graph is
acyclic, nothing is left that only a cycle keeps.

A spared node leaves `spared` when it is released into the candidates, whatever happens to it
then: if it has gained a consumer, it is no longer one that a retirement would have freed, and if
it later loses it, it is a candidate through $\operatorname{in}(R)$ again.

### Errors

No new variant. `UnknownNode`'s documentation and message say that the node may have been freed:

```rust
/// A handle the model does not hold: from another model, retired, or freed by a retirement
/// that left it needed by nothing (`scope::pin` keeps such a node).
UnknownNode { id: NodeId },
// "the model does not hold node {id:?}: it is from another model, was retired, or was freed by
//  a retirement (pin a node to keep it)"
```

The documentation of `Retirement::replace` and of `Writer::retire` adds the freeing to what a
retirement removes, and the module documentation of `retire.rs` links the entry's mathematics.

### The example

`sliding_window` reports the number of nodes in the window, and checks that it stopped growing:
the largest count over the ticks after $2W$ is at most 1.5 times the largest over the ticks from
$W + 1$ to $2W$, the first full windows. The window's quantities vary between 33 and 50, a
ratio of 1.5, with the landmarks in view; before this entry the count grew by a factor of five
between tick 9 and tick 79, which the check fails.

## Tests

In `crates/skoro/tests/retirement.rs`, on the test chain and small models built beside it:

- `retirement_reads_the_blanket_and_removes_what_it_names` (changed): `a` moves from the nodes
  that stay to the nodes removed, since only the term of $\randomPoint_0$ and its sampler read it;
  `kept` still stays, which is the locality of [](eq:archive/retired-constants/math/freed).
- `freed_nodes_are_what_nothing_needs`: after retiring the first tick, every node of the new
  snapshot (`Snapshot::nodes`) with no consumer is a quantity, kept (a term's log-density, a
  binding, a sampler's or a proposal's node, or pinned), or had no consumer in the old snapshot.
  This is part (2) of [%full](statement:archive/retired-constants/math/freeing), checked
  directly.
- `nothing_left_refers_to_a_removed_node` (as now): its `assert_closed` walks every term,
  sampler and proposal of the new snapshot back to its roots, and must find every node it names.
  With freeing, this is part (1) of the proposition, and it passes unchanged.
- `kalman_replacement_matches_quadrature` and `part_of_a_tick_matches_the_joint_normal` (as now):
  the densities and derivatives after the retirement are the closed forms. Freeing changes no
  value, and these show it.
- `shared_constant_stays_while_a_term_reads_it`: a mean constant shared by the terms of
  $\bfz_0$ and $\bfz_1$ stays when the first tick is retired; one given to the term of $\bfz_0$
  alone is freed.
- `sampler_inputs_are_held`: a term that is kept, whose sampler reads a constant its log-density
  does not, where that constant also fed a retired term; the constant stays, and the model still
  generates. This is the case the mathematics was reopened for.
- `replacement_reusing_a_retired_constant_keeps_it`: a replacement built from a constant that only
  the retired terms read keeps it, since freeing runs against the model after the replacement.
- `unused_values_of_the_replacement_scope_are_freed`: a constant and a mapped value of $\point_1$
  made in the replacement's body and not reached from the factor or its bindings are gone after
  `replace`.
- `pinned_constant_survives_a_retirement_that_leaves_it_unused`: a pinned constant read only by
  retired terms stays, and a tick built after the retirement uses it; the same model unpinned
  fails that build with `UnknownNode`.
- `unpinned_node_is_freed_at_the_next_retirement`: a pinned constant spared by one retirement
  stays after `unpin`, and is freed by the next retirement that is not empty, though that
  retirement touches nothing near it.
- `pin_on_a_node_reached_from_a_retired_quantity_does_not_keep_it`: a pinned mapped value of
  $\randomPoint_0$ is removed with it, and the new snapshot does not report it pinned.
- `pins_belong_to_their_branch`: a fork unpins a constant its parent pinned; the same retirement
  in both frees it in the fork alone.
- `pin_refusals`: `pin` of a handle from another model (`UnknownNode`), of a failed handle
  (`FailedOperand`), with no scope open (`OutsideScope`), and inside a composite's definition
  (`NotInAModel`).
- `steady_window_holds_a_constant_number_of_nodes`: the chain extended one tick at a time for 100
  ticks, retiring the oldest after each, with a replacement of fresh constants; `Snapshot::len`
  after every retirement from the third on equals its value after the third. Before this entry it
  grows by a fixed amount at every retirement.

In `crates/skoro-core/src/retire.rs`, a unit test, since the index is private:

- `held_index_matches_the_terms`: a model whose terms bind, replace a sampler and carry
  proposals, retired twice; after every change, `held` equals the counts recomputed from every
  term's bindings and sampler and every proposal, and every node of `spared` is a node of the
  model. This is the archive's `validate`, for the one index this entry adds.

The example `sliding_window` checks that its node count stopped growing, as
[The example](#the-example) says.
