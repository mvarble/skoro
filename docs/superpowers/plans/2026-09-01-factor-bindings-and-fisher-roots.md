# Factor Bindings and Fisher Roots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `FactorRecord`/`ResidualRegistry` side table in `skoro-model` with **bindings** — labelled reference edges stored in `skoro-graph` from a log-density term to other nodes — and add the first new binding, a **Fisher information root** node, plus the assembly code (`information`, `information_root`, `information_blocks`) that reads it.

**Architecture:** Additive-first, breaking-change-isolated. Phases 1-3 add bindings, terms, `Factor`/`Handle`, and Fisher assembly *alongside* the existing record machinery so `cargo test --workspace` stays green throughout. Phase 4a is the one atomic commit that flips `Distribution::log_density` to `Distribution::factor` and every `impl Distribution` with it. Phase 4b deletes the record machinery. Phase 5 adds the Fisher-specific integration tests and updates docs.

**Tech Stack:** Rust 2024, `burn` tensors (`skoro-backend::{Backend, Tensor, TensorData}`), the `skoro-graph`/`skoro-model`/`skoro-linalg`/`skoro-normal`/`skoro` workspace crates.

**Spec:** `docs/superpowers/specs/2026-09-01-factor-bindings-and-fisher-root-spec.md` — every `R<n>` and `§<n>` reference below points there. This plan does not restate the spec's requirement text, math derivation (§4), or full acceptance-test tables (§5); it sequences them into TDD steps and gives the code for everything not already exact in the spec. **Read the cited spec section before starting each task.**

## Global Constraints

- One backend per test binary: never mix `NdArrayBackend<f32,_>` and `<f64,_>` in one test executable (§8).
- Build tensors from `TensorData` (with `.convert::<B::FloatElem>()` where the source is a host `Vec<f64>`), never `Tensor::from_floats`, which rounds through `f32` even on an `f64` backend (§8).
- `HVP` state gating: `VjpState` gates on `diffs`, `HvpState` gates on `tangents`; an `hvp` output slot must be `None` whenever the corresponding `vjp` slot is (§8, `.claude/skills/skoro-diff-operation/SKILL.md`).
- `Q.client` is `pub(crate)`; `Handle` must live in `skoro-model`, in `crates/skoro-model/src/q.rs` (§8, R13).
- Lock discipline in `add_quantity`: the write lock on `ModelServer` must be released before calling `distr.factor(&x)` and re-taken after (§8, R16).
- Bindings are never data edges: `topological_sort` and `compile_backward_from` must not read `Bindings` (R10). The collector and `Graph::remove`/`remove_closure` must always treat a binding as a reference (R7, R8, R17).
- Layouts are sized by manifold dimension, never storage (`SE3` is 6 wide, not 12) — this is already true of `TangentLayout::from_ids`; nothing here changes it, but every new size claim in this plan (e.g. the Fisher root's column count) must respect it.
- Run every phase checkpoint from the workspace root (`/home/mvarble/dev/skoro`) before starting the next phase (§6).
- Commit messages: the bodies below are complete; end each with the attribution trailer your own session requires. None is pre-written into this plan.

---

## Phase 1 — `skoro-graph`: bindings and terms (additive)

Implements R1, R5–R11 and the tests in spec §5.1 (`crates/skoro-graph/tests/graph_mutation.rs`, new `mod bindings`). Checkpoint at the end of the phase: `cargo test -p skoro-graph -p skoro-core` and `cargo build --workspace` both pass (§6, Phase 1).

### Task 1: `skoro-core` — `NodeStillBound` replaces `NodeIsRecorded` (R1)

**Files:**
- Modify: `crates/skoro-core/src/error.rs` (the `NodeIsRecorded { id: Id }` variant at line 62 and its `Display` arm at line 166)
- Modify: `crates/skoro-core/tests/error.rs` (lines 61-64 and 174)
- Modify: `crates/skoro-model/src/model.rs:239-241` (`ModelServer::remove`'s registry check — temporary shim, see below)
- Modify: `crates/skoro-model/tests/residual.rs:180-181` and its doc comment at `crates/skoro-model/tests/compile_fail/remove_closure_is_not_a_client_op.rs:5`

**Interfaces:**
- Produces: `SkoroError::NodeStillBound { id: Id, by: Id }`, which every later phase's binding-refusal code returns.

This task has no new acceptance test of its own (the tests that exercise `NodeStillBound` come in Task 4); it is a rename that must keep the workspace compiling because `skoro-model` still references the old variant name until Phase 4b deletes the registry.

- [ ] **Step 1: Rename the error variant**

In `crates/skoro-core/src/error.rs`, replace:

```rust
    /// The node cannot be removed because a recorded factor names it — deleting it would
    /// take a term out of the objective, and leave the record naming a node that is gone.
    /// Retire the factor with a closure removal instead, which takes the referents.
    NodeIsRecorded { id: Id },
```

with:

```rust
    /// The node cannot be removed because a binding still names it as its target — deleting
    /// it would leave the referrer pointing at nothing. Retire the referrer with a closure
    /// removal instead, which takes the referents.
    NodeStillBound { id: Id, by: Id },
```

and its `Display` arm:

```rust
            SkoroError::NodeIsRecorded { id } => {
                write!(
                    f,
                    "cannot remove node {id:?}; a recorded factor names it \
                     (retire the factor with a closure removal instead)"
                )
            }
```

becomes:

```rust
            SkoroError::NodeStillBound { id, by } => {
                write!(
                    f,
                    "cannot remove node {id:?}; it is still bound by {by:?} \
                     (retire the referrer with a closure removal instead)"
                )
            }
```

- [ ] **Step 2: Fix `crates/skoro-core/tests/error.rs`**

Replace the message-table entry at line 61-64:

```rust
        (
            SkoroError::NodeIsRecorded { id },
            "a recorded factor names it",
        ),
```

with:

```rust
        (
            SkoroError::NodeStillBound { id, by: id },
            "still bound by",
        ),
```

Replace the id-containment sweep entry at line 174, `SkoroError::NodeIsRecorded { id },`, with `SkoroError::NodeStillBound { id, by: id },`.

- [ ] **Step 3: Temporary shim in `skoro-model`**

In `crates/skoro-model/src/model.rs:239-241`:

```rust
        if let Some(id) = ids.iter().find(|id| self.registry.names(**id)) {
            return Err(SkoroError::NodeIsRecorded { id: *id });
        }
```

becomes (temporary — Phase 4b deletes this whole `if` block along with the registry):

```rust
        if let Some(id) = ids.iter().find(|id| self.registry.names(**id)) {
            return Err(SkoroError::NodeStillBound { id: *id, by: *id });
        }
```

In `crates/skoro-model/tests/residual.rs:180-181`, change `Err(SkoroError::NodeIsRecorded { .. })` to `Err(SkoroError::NodeStillBound { .. })`. Update the doc comment mentioning `NodeIsRecorded` at `crates/skoro-model/tests/compile_fail/remove_closure_is_not_a_client_op.rs:5` to say `NodeStillBound`.

- [ ] **Step 4: Verify**

Run: `cargo test -p skoro-core -p skoro-model`
Expected: PASS (the `residual.rs` and `error.rs` suites still hold; nothing about behavior changed, only the variant's name and its two fields).

- [ ] **Step 5: Commit**

```bash
git add crates/skoro-core/src/error.rs crates/skoro-core/tests/error.rs \
  crates/skoro-model/src/model.rs crates/skoro-model/tests/residual.rs \
  crates/skoro-model/tests/compile_fail/remove_closure_is_not_a_client_op.rs
git commit -m "$(cat <<'EOF'
Rename SkoroError::NodeIsRecorded to NodeStillBound{id,by}

Prepares for bindings replacing the FactorRecord registry (R1). The
registry's refusal check is patched to pass by:id as a temporary shim
until Phase 4b deletes the registry outright.
EOF
)"
```

### Task 2: `Bindings` type and `Graph` accessors (R5, R6, R9 partial)

**Files:**
- Create: `crates/skoro-graph/src/binding.rs`
- Modify: `crates/skoro-graph/src/lib.rs` (add `pub mod binding;`)
- Modify: `crates/skoro-graph/src/graph.rs` (new private field `bindings: Bindings`, four new public methods, `validate_indices` extension)
- Test: `crates/skoro-graph/tests/graph_mutation.rs` (new `mod bindings`, first two tests)

**Interfaces:**
- Produces: `skoro_graph::binding::{Role, Bindings}`; `Graph::{bind, bindings_of, bound, referrers_of}`.
- Consumes: nothing new — `Graph::new()`'s `#[derive(Default)]` already default-inits new struct fields.

- [ ] **Step 1: Write the failing tests**

In `crates/skoro-graph/tests/graph_mutation.rs`, add at the end of the file (after the existing `mod degenerate`):

```rust
mod bindings {
    use super::*;
    use skoro_graph::binding::Role;

    const R1: Role = Role("r1");
    const R2: Role = Role("r2");

    #[test]
    fn bind_refuses_a_missing_endpoint() {
        let mut graph = Graph::<B>::new();
        let a = graph.add_parameter(Real);
        let fresh = bogus_id();

        assert!(matches!(
            graph.bind(a, R1, fresh),
            Err(SkoroError::NodeNotFound { id }) if id == fresh
        ));
        assert!(matches!(
            graph.bind(fresh, R1, a),
            Err(SkoroError::NodeNotFound { id }) if id == fresh
        ));
        graph.validate_indices().unwrap();
    }

    #[test]
    fn bind_is_readable_in_order() {
        let mut graph = Graph::<B>::new();
        let a = graph.add_parameter(Real);
        let b = graph.add_parameter(Real);
        let c = graph.add_parameter(Real);

        graph.bind(a, R1, b).unwrap();
        graph.bind(a, R2, c).unwrap();
        graph.bind(a, R1, c).unwrap();

        assert_eq!(graph.bindings_of(a), &[(R1, b), (R2, c), (R1, c)]);
        assert_eq!(graph.bound(a, R1), vec![b, c]);
        assert_eq!(graph.referrers_of(b), vec![a]);
        graph.validate_indices().unwrap();
    }
}
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `cargo test -p skoro-graph --test graph_mutation bindings`
Expected: FAIL to compile — `skoro_graph::binding` does not exist, and `Graph::bind`/`bindings_of`/`bound`/`referrers_of` are not methods.

- [ ] **Step 3: Write `crates/skoro-graph/src/binding.rs`**

```rust
//! Labelled reference edges from a graph node to others, stored beside the arena.
//!
//! A binding `(from, role, to)` means "`to` plays `role` for `from`". Bindings are not data
//! edges — the compiler never reads them (R10) — but they *are* references: [`Graph::remove`]
//! refuses to take a bound target out from under its referrer, and the collector marks a
//! bound target reachable whenever its referrer is.

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
```

- [ ] **Step 4: Declare the module**

In `crates/skoro-graph/src/lib.rs`, add `pub mod binding;` next to the other `pub mod` lines (alphabetical: after `pub mod compiler;`'s neighbors — place it before `pub mod compiler;` since `binding` sorts first).

- [ ] **Step 5: Add the field and the four methods to `Graph`**

In `crates/skoro-graph/src/graph.rs`, add the import `use crate::binding::{Bindings, Role};` beside the other `use crate::` lines, add the field to the struct (after `private_nodes: HashSet<Id>,`):

```rust
    /// Labelled reference edges from a log-density term to other nodes (R5/R6). Read by
    /// inference algorithms and by the collector; never by the compiler (R10).
    bindings: Bindings,
```

`Graph::new()` (`graph.rs:123-132`) builds the struct field by field; add `bindings: Bindings::default(),` after `private_nodes: HashSet::new(),`.

Add the four methods, near `pub fn contains`/`pub fn node` (around line 136-151):

```rust
    /// Record `(from, role, to)`. Errors [`SkoroError::NodeNotFound`] if `from` is missing
    /// (checked first) or `to` is. Duplicate identical triples are allowed and stored twice —
    /// nothing here deduplicates. No cycle check: a binding is a reference, not a data edge.
    pub fn bind(&mut self, from: Id, role: Role, to: Id) -> SkoroResult<()> {
        if !self.nodes.contains_key(&from) {
            return Err(SkoroError::NodeNotFound { id: from });
        }
        if !self.nodes.contains_key(&to) {
            return Err(SkoroError::NodeNotFound { id: to });
        }
        self.bindings.bind(from, role, to);
        self.debug_validate_indices();
        Ok(())
    }

    /// Every `(role, to)` pair `from` has bound, in binding order.
    pub fn bindings_of(&self, from: Id) -> &[(Role, Id)] {
        self.bindings.of(from)
    }

    /// The `to`s `from` has bound under `role`, in binding order.
    pub fn bound(&self, from: Id, role: Role) -> Vec<Id> {
        self.bindings.bound(from, role)
    }

    /// Every node that has bound `to` under any role.
    pub fn referrers_of(&self, to: Id) -> Vec<Id> {
        self.bindings.referrers_of(to).collect()
    }
```

- [ ] **Step 6: Extend `validate_indices` (R9, first half)**

Add a crate-private iterator to `Bindings` (in `binding.rs`), since `by_source` is private:

```rust
    /// Every `(from, [(role, to), ..])` pair, for validation only.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (Id, &[(Role, Id)])> + '_ {
        self.by_source.iter().map(|(from, pairs)| (*from, pairs.as_slice()))
    }
```

Then in `crates/skoro-graph/src/graph.rs`, inside `validate_indices` (`graph.rs:295-397`), add before the final `Ok(())`:

```rust
        for (from, pairs) in self.bindings.iter() {
            if !self.nodes.contains_key(&from) {
                return Err(format!("a binding names {from:?} as a source, which is not a node"));
            }
            for (_, to) in pairs {
                if !self.nodes.contains_key(to) {
                    return Err(format!("a binding names {to:?} as a target, which is not a node"));
                }
            }
        }
        self.bindings.validate()?;
```

- [ ] **Step 7: Run the tests**

Run: `cargo test -p skoro-graph --test graph_mutation bindings`
Expected: PASS — `bind_refuses_a_missing_endpoint` and `bind_is_readable_in_order` both green.

- [ ] **Step 8: Run the full crate suite to catch drift**

Run: `cargo test -p skoro-graph`
Expected: PASS (nothing else touched `Graph`'s field list or `validate_indices` in a way existing tests exercise).

- [ ] **Step 9: Commit**

```bash
git add crates/skoro-graph/src/binding.rs crates/skoro-graph/src/lib.rs \
  crates/skoro-graph/src/graph.rs crates/skoro-graph/tests/graph_mutation.rs
git commit -m "$(cat <<'EOF'
Add Bindings and Graph::{bind,bindings_of,bound,referrers_of}

Labelled reference edges from a term to other nodes, stored in the
graph rather than a side table (R5/R6). validate_indices now also
checks every binding endpoint exists and that Bindings' own reverse
index agrees with its forward one (R9, first half).
EOF
)"
```

### Task 3: `remove` / `remove_closure` honor bindings (R7, R8)

**Files:**
- Modify: `crates/skoro-graph/src/graph.rs` (`remove` at line ~1266, `remove_closure` at line ~221)
- Test: `crates/skoro-graph/tests/graph_mutation.rs` (`mod bindings`, three more tests)

**Interfaces:**
- Consumes: `Graph::{bindings_of, referrers_of}` from Task 2, `SkoroError::NodeStillBound` from Task 1.

- [ ] **Step 1: Write the failing tests**

Append to `mod bindings` in `crates/skoro-graph/tests/graph_mutation.rs`:

```rust
    #[test]
    fn remove_refuses_a_bound_target() {
        let mut graph = Graph::<B>::new();
        let a = graph.add_parameter(Real);
        let b = graph.add_parameter(Real);
        graph.bind(a, R1, b).unwrap();

        assert!(matches!(
            graph.remove(&[b]),
            Err(SkoroError::NodeStillBound { id, by }) if id == b && by == a
        ));
        graph.remove(&[a, b]).unwrap();
        graph.validate_indices().unwrap();
    }

    #[test]
    fn remove_source_clears_the_reference() {
        let mut graph = Graph::<B>::new();
        let a = graph.add_parameter(Real);
        let b = graph.add_parameter(Real);
        graph.bind(a, R1, b).unwrap();

        graph.remove(&[a]).unwrap();
        assert!(graph.referrers_of(b).is_empty());
        graph.remove(&[b]).unwrap();
        graph.validate_indices().unwrap();
    }

    #[test]
    fn remove_closure_takes_referrers() {
        let mut graph = Graph::<B>::new();
        let a = graph.add_parameter(Real);
        let b = graph.add_parameter(Real);
        graph.bind(a, R1, b).unwrap();

        let removed = graph.remove_closure(&[b]).unwrap();
        assert_eq!(removed.len(), 2, "b and its referrer a");
        assert!(!graph.contains(a));
        assert!(!graph.contains(b));
        graph.validate_indices().unwrap();
    }

    #[test]
    fn validate_indices_holds_after_binding_mutations() {
        let mut graph = Graph::<B>::new();
        let a = graph.add_parameter(Real);
        let b = graph.add_parameter(Real);
        let c = graph.add_parameter(Real);
        graph.bind(a, R1, b).unwrap();
        graph.bind(a, R2, c).unwrap();
        graph.bind(a, R1, c).unwrap();
        graph.validate_indices().unwrap();

        graph.remove(&[a, b, c]).unwrap();
        graph.validate_indices().unwrap();
    }
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `cargo test -p skoro-graph --test graph_mutation bindings`
Expected: FAIL — `remove_refuses_a_bound_target` and `remove_closure_takes_referrers` fail because `remove`/`remove_closure` do not yet consult bindings (they will currently just succeed / not extend the closure).

- [ ] **Step 3: `Graph::remove` — refuse and clear bindings**

In `crates/skoro-graph/src/graph.rs`, `pub fn remove` (currently starting ~line 1266): add the binding-referrer check right after the existing `NodeStillReferenced` loop and before `let mut out = Vec::with_capacity(ids.len());`:

```rust
        for id in ids {
            if let Some(referrer) = self.referrers_of(*id).into_iter().find(|r| !ids.contains(r)) {
                return Err(SkoroError::NodeStillBound { id: *id, by: referrer });
            }
        }
```

Then, inside the removal loop, right after `self.private_nodes.remove(id);` (which already runs per removed node), add:

```rust
            self.bindings.remove_source(*id);
            self.bindings.remove_target(*id);
```

- [ ] **Step 4: `Graph::remove_closure` — extend the walk through referrers**

In `remove_closure` (currently ~line 221), the closure-building `while let Some(id) = queue.pop_front()` loop currently does:

```rust
            if let Some(children) = self.children.get(&id) {
                queue.extend(children.iter().copied());
            }
```

Add immediately after:

```rust
            queue.extend(self.referrers_of(id));
```

- [ ] **Step 5: Run the tests**

Run: `cargo test -p skoro-graph --test graph_mutation bindings`
Expected: PASS, all four `mod bindings` tests plus the two from Task 2.

- [ ] **Step 6: Run the full crate suite**

Run: `cargo test -p skoro-graph`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add crates/skoro-graph/src/graph.rs crates/skoro-graph/tests/graph_mutation.rs
git commit -m "$(cat <<'EOF'
Graph::remove/remove_closure treat bindings as references

remove refuses a batch that leaves a bound target's referrer outside
it (NodeStillBound), and clears both binding directions for every
removed node. remove_closure's downward walk now also follows
referrers_of, so retiring a bound target takes its referrer with it
(R7/R8).
EOF
)"
```

### Task 4: `Term`/`TermKind` and `Graph::{terms, log_density_terms}` (R11)

**Files:**
- Create: `crates/skoro-graph/src/term.rs`
- Modify: `crates/skoro-graph/src/lib.rs` (`mod term;`, private — re-exported from `graph`)
- Modify: `crates/skoro-graph/src/graph.rs` (`pub use crate::term::{Term, TermKind};` near the top, two new methods)
- Test: `crates/skoro-graph/tests/graph_mutation.rs` (`mod bindings`, two more tests)

**Interfaces:**
- Produces: `skoro_graph::graph::{Term, TermKind}`, `Graph::{terms, log_density_terms}` — Phase 3's `ModelClientOps::terms`/`FactorInfo` and Phase 3's `information.rs` both consume these.

- [ ] **Step 1: Write the failing tests**

Append to `mod bindings` (`stochastic`, `inert`, `summed` and `bogus_id` are the helpers already defined at the top of `graph_mutation.rs`, lines 27-60; `points`, `scalar` and `into_vec` come from `tests/common/mod.rs`):

```rust
    #[test]
    fn terms_classify_proposal_and_weight() {
        let mut graph = Graph::<B>::new();
        let stoch = stochastic(&mut graph, vec![]);
        let density = inert(&mut graph, vec![stoch]);
        graph.mark_log_density(density, stoch).unwrap();

        let obs = stochastic(&mut graph, vec![]);
        let obs_density = inert(&mut graph, vec![obs]);
        graph.mark_log_density(obs_density, obs).unwrap();
        graph.observe::<_, Real, _>(obs, &1.0f64).unwrap();

        let free_standing = graph.add_parameter(Real);
        let free_density = inert(&mut graph, vec![free_standing]);
        graph.mark_log_likelihood(free_density).unwrap();

        let terms = graph.terms();
        assert_eq!(
            terms.iter().map(|t| t.density).collect::<Vec<_>>(),
            vec![density, obs_density, free_density],
            "marking order"
        );
        assert_eq!(
            terms.iter().map(|t| t.kind).collect::<Vec<_>>(),
            vec![TermKind::Proposal, TermKind::Weight, TermKind::Weight]
        );

        graph.observe::<_, Real, _>(stoch, &2.0f64).unwrap();
        assert_eq!(graph.terms()[0].kind, TermKind::Weight);
        graph.validate_indices().unwrap();
    }

    #[test]
    fn log_density_terms_sums_only_the_named_terms() {
        let mut graph = Graph::<B>::new();
        let stoch = stochastic(&mut graph, vec![]);
        let t1 = inert(&mut graph, vec![stoch]);
        graph.mark_log_density(t1, stoch).unwrap();
        let free_standing = graph.add_parameter(Real);
        let t2 = summed(&mut graph, vec![free_standing]);
        graph.mark_log_likelihood(t2).unwrap();

        // `stoch` is a leaf the compiled program reads from `points`, so it needs a value too.
        let points = points([(free_standing, 3.0), (stoch, 0.5)]);
        let whole = graph.log_density(&points).unwrap();
        let split = (into_vec(graph.log_density_terms(&points, &[t1]).unwrap())[0]
            + into_vec(graph.log_density_terms(&points, &[t2]).unwrap())[0])
            - into_vec(whole)[0];
        approx::assert_abs_diff_eq!(split, 0.0, epsilon = 1e-12);

        let mapped = inert(&mut graph, vec![free_standing]);
        assert!(matches!(
            graph.log_density_terms(&points, &[mapped]),
            Err(SkoroError::NotLogLikelihood { id }) if id == mapped
        ));
    }
```

`points` (`tests/common/mod.rs:61`) builds a `ValueMap` from `(Id, f64)` pairs, and `approx` is already a dev-dependency of `skoro-graph`, so nothing needs adding to `common/mod.rs` or `Cargo.toml`.

- [ ] **Step 2: Run the tests to see them fail**

Run: `cargo test -p skoro-graph --test graph_mutation bindings`
Expected: FAIL to compile — `TermKind`, `graph.terms()`, `graph.log_density_terms()` do not exist yet.

- [ ] **Step 3: Write `crates/skoro-graph/src/term.rs`**

```rust
//! A term is a [`Node::LogLikelihood`](crate::node::Node::LogLikelihood) that is currently in
//! [`Graph::log_likelihoods`](crate::graph::Graph::log_likelihoods) — one summand of the
//! model's objective.

use skoro_core::id::Id;

/// Whether a term scores a live stochastic quantity (its removal changes the *proposal*) or
/// not (an observed density, or a free-standing factor — its removal changes only the
/// *weight*).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TermKind {
    Proposal,
    Weight,
}

/// One summand of the model's log-density, as read off [`Graph::terms`](crate::graph::Graph::terms).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Term {
    /// The log-likelihood node.
    pub density: Id,
    /// The quantity this term is the log-density of, if any — `None` for a free-standing
    /// factor.
    pub scored: Option<Id>,
    pub kind: TermKind,
}
```

- [ ] **Step 4: Declare and re-export the module**

In `crates/skoro-graph/src/lib.rs`, add `mod term;` (private — the public surface is the re-export from `graph`).

In `crates/skoro-graph/src/graph.rs`, near the top-level `use crate::` block, add:

```rust
pub use crate::term::{Term, TermKind};
```

- [ ] **Step 5: Implement the two `Graph` methods**

Add near `pub fn log_density` (~line 1373):

```rust
    /// One [`Term`] per id in [`log_likelihoods`](Self::log_likelihoods), in that order.
    pub fn terms(&self) -> Vec<Term> {
        self.log_likelihoods
            .iter()
            .map(|&density| {
                let scored = self.stochastic_of(density);
                let kind = match scored {
                    Some(s) if matches!(self.nodes.get(&s), Some(Node::Stochastic(_))) => {
                        TermKind::Proposal
                    }
                    _ => TermKind::Weight,
                };
                Term { density, scored, kind }
            })
            .collect()
    }

    /// [`log_density`](Self::log_density) restricted to `terms`: every id must be in
    /// [`log_likelihoods`](Self::log_likelihoods) (else [`SkoroError::NotLogLikelihood`]), and
    /// an empty slice yields zeros of shape `[batch, 1]`, batch taken from `points` exactly as
    /// [`log_density`](Self::log_density) does.
    pub fn log_density_terms(&self, points: &ValueMap<B>, terms: &[Id]) -> SkoroResult<Point<B>> {
        for id in terms {
            if !self.log_likelihoods.contains(id) {
                return Err(SkoroError::NotLogLikelihood { id: *id });
            }
        }
        let compiled_sequence = self.compile_backward_from(terms);
        let batch_size = if let Some((_, point)) = points.iter().next() {
            point.dims()[0]
        } else {
            1
        };
        let forward_output = compiled_sequence.forward(ForwardInput {
            points,
            options: ForwardOptions {
                batch_size,
                use_model_samplers: false,
            },
        })?;
        let mut log_density = Point::zeros([batch_size, 1], &Default::default());
        for id in terms {
            log_density = log_density
                + forward_output
                    .evaluations
                    .get(id)
                    .expect("a validated term should be evaluated")
                    .0
                    .clone()
                    .unwrap_float();
        }
        Ok(log_density)
    }
```

- [ ] **Step 6: Run the tests**

Run: `cargo test -p skoro-graph --test graph_mutation bindings`
Expected: PASS, all 8 tests in `mod bindings`.

- [ ] **Step 7: Run the full crate suite and the Phase 1 checkpoint**

Run: `cargo test -p skoro-graph -p skoro-core`
Expected: PASS.

Run: `cargo build --workspace`
Expected: PASS — this is the Phase 1 checkpoint (§6). `skoro-model` still compiles against the `NodeStillBound` shim from Task 1.

- [ ] **Step 8: Commit**

```bash
git add crates/skoro-graph/src/term.rs crates/skoro-graph/src/lib.rs \
  crates/skoro-graph/src/graph.rs crates/skoro-graph/tests/graph_mutation.rs \
  crates/skoro-graph/tests/common/mod.rs
git commit -m "$(cat <<'EOF'
Add Term/TermKind and Graph::{terms,log_density_terms}

A Term classifies a log-likelihood node as Proposal (scores a live
Stochastic quantity) or Weight (everything else), re-derived from the
node kind on every call rather than cached. log_density_terms is
log_density restricted to a named subset of terms (R11).
EOF
)"
```

**Phase 1 checkpoint (already run in Task 4, Step 7):** `cargo test -p skoro-graph -p skoro-core` and `cargo build --workspace` both pass.

---

## Phase 2 — `skoro-linalg`: `FixedLinearMap` (additive)

Implements R2–R4 and the tests in spec §5.2 (`crates/skoro-linalg/tests/linear_map.rs`, new file). Checkpoint: `cargo test -p skoro-linalg` (§6, Phase 2).

Before starting, re-read `.claude/skills/skoro-diff-operation/SKILL.md` (already loaded once during planning) for the `Hvp` two-gate rule, and `crates/skoro-operation/src/ops.rs:644-706` (`LiteralScalarMultiplication`) for the exact `hvp` gating this op must copy: `hvp` returns `Some(omega.zeros_like())` when the vjp state is `true`, `None` otherwise — never compute a nonzero second derivative for a linear map.

### Task 5: `FixedLinearMap` and the `linear_map` sugar (R2–R4)

**Files:**
- Create: `crates/skoro-linalg/src/matrix/linear_map.rs`
- Modify: `crates/skoro-linalg/src/matrix/mod.rs` (add `mod linear_map; pub use linear_map::*;`)
- Test: `crates/skoro-linalg/tests/linear_map.rs` (new)

**Interfaces:**
- Produces: `skoro_linalg::matrix::{FixedLinearMap, linear_map}` — Phase 3's `information_blocks`, Phase 3.7's `gaussian_root_matrix`/`information_root`, and Phase 3.7's `JointNormal::factor` all consume `FixedLinearMap`/`linear_map`.
- Consumes: `skoro_model::map::register_map1` (existing), `ContinuousQuantity` (existing).

- [ ] **Step 1: Write the failing tests**

Create `crates/skoro-linalg/tests/linear_map.rs`. Follow the backend/tensor-helper conventions already used by `crates/skoro-linalg`'s other integration tests (check `crates/skoro-linalg/tests/` for an existing `common`/fixture module and reuse it rather than redefining `type B` and tensor helpers from scratch — if none exists, define locally as below, matching the `skoro-graph` convention of `NdArrayBackend<f64, i32>` and `TensorData`-built tensors):

```rust
use skoro_backend::{NdArrayBackend, Tensor, TensorData};
use skoro_core::error::SkoroError;
use skoro_model::{
    ModelClientOps,
    hash_map_builders::points,
    jacobian::{jacobian_jvp, jacobian_vjp},
    model::ModelClient,
    test_utils::{into_vec, test_hvp_numerically, test_jvp_numerically, test_vjp_numerically},
};
use skoro_linalg::matrix::{linear_map, FixedLinearMap};
use skoro_riemann::euclidean::EuclideanD;

type B = NdArrayBackend<f64, i32>;

fn row(values: &[f64]) -> Tensor<B, 2> {
    let n = values.len();
    Tensor::<B, 1>::from_data(TensorData::new(values.to_vec(), [n]), &Default::default())
        .reshape([1, n])
}

/// `K = [[1, 2], [3, 4], [5, 6]]`, row-major, `3x2`.
fn k() -> FixedLinearMap<B> {
    FixedLinearMap::from_row_major(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &Default::default())
}

#[test]
fn forward_matches_the_host_product() {
    let model = ModelClient::<B>::default();
    let x = model.add_parameter(EuclideanD(2));
    let y = linear_map(&x, EuclideanD(3), k()).unwrap();

    let values = points().with_quantity(&x, row(&[1.0, 1.0])).build();
    let out = into_vec(model.evaluate(&values, &y).unwrap());
    // K @ [1,1]^T = [3, 7, 11]
    assert_eq!(out, vec![3.0, 7.0, 11.0]);
}

#[test]
fn jacobian_is_the_matrix() {
    let model = ModelClient::<B>::default();
    let x = model.add_parameter(EuclideanD(2));
    let y = linear_map(&x, EuclideanD(3), k()).unwrap();

    let values = points().with_quantity(&x, row(&[1.0, 1.0])).build();
    let inputs = model.tangent_layout(&[x.id()]).unwrap();
    let outputs = model.tangent_layout(&[y.id()]).unwrap();

    let jvp = into_vec(jacobian_jvp(&model, &values, &inputs, &outputs).unwrap().reshape([3, 2]));
    let vjp = into_vec(jacobian_vjp(&model, &values, &inputs, &outputs).unwrap().reshape([3, 2]));
    let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    for (got, want) in [(&jvp, &expected), (&vjp, &expected)] {
        for (g, w) in got.iter().zip(want) {
            approx::assert_abs_diff_eq!(g, w, epsilon = 1e-12);
        }
    }
}

#[test]
fn hvp_is_zero() {
    let model = ModelClient::<B>::default();
    let x = model.add_parameter(EuclideanD(2));
    let y = linear_map(&x, EuclideanD(3), k()).unwrap();

    let values = points().with_quantity(&x, row(&[1.0, 1.0])).build();
    let tangents = [(x.id(), row(&[1.0, 0.0]))].into_iter().collect();
    let ord1 = [(y.id(), Tensor::<B, 3>::ones([1, 3, 1], &Default::default()))]
        .into_iter()
        .collect();
    let out = model
        .evaluate_hvp(&values, &tangents, &ord1, &ord1, &y)
        .unwrap();
    // No second-order cotangent should land on x: a linear map has zero Hessian.
    assert!(out.ord2_cotangents.get(&x.id()).is_none_or(|t| into_vec(t.clone()).iter().all(|v| *v == 0.0)));
}

#[test]
fn shape_mismatch_is_reported() {
    let model = ModelClient::<B>::default();
    let x = model.add_parameter(EuclideanD(2));
    // k() is 3x2; a 4-wide output needs 4 rows, and the check reports rows first.
    assert!(matches!(
        linear_map(&x, EuclideanD(4), k()),
        Err(SkoroError::ShapeMismatch { expected: 4, found: 3 })
    ));
}
```

The exact field/method names on the `evaluate_hvp` result (`ord2_cotangents` above) and on `PointHvpEval`/`Q::evaluate_hvp` need checking against `crates/skoro-graph/src/graph.rs`'s `PointHvpEval` definition and an existing HVP test (e.g. `crates/skoro-model/tests/hvp_diff_gating.rs`) before this compiles — read that file first and match its exact call shape; the assertion here should end up checking that the HVP-mode cotangent flowing back to `x` is exactly zero (or absent), which is the observable form of "linear map ⇒ zero Hessian" per the SKILL.md note that a wrongly-nonzero HVP is wasted work, not wrong output, so prefer asserting on the *value* being zero rather than on the flag being unset if both are inspectable.

- [ ] **Step 2: Run the tests to see them fail**

Run: `cargo test -p skoro-linalg --test linear_map`
Expected: FAIL to compile — `FixedLinearMap`/`linear_map` do not exist.

- [ ] **Step 3: Write `crates/skoro-linalg/src/matrix/linear_map.rs`**

```rust
//! A fixed (non-differentiable) linear map `y = K x`, for turning a host-computed matrix into
//! a graph node without a distribution being able to add constants of its own (`Q.client` is
//! `pub(crate)` — see `crates/skoro-model/src/q.rs`).

use skoro_backend::{Backend, CotangentFrame, Point, Tensor, TensorData};
use skoro_core::error::{SkoroError, SkoroResult};
use skoro_core::kind::Continuous;
use skoro_graph::quantity::ContinuousQuantity;
use skoro_model::{map::register_map1, q::Q};
use skoro_riemann::Manifold;
use skoro_operation::{
    Input,
    diff::{DiffModeState, DiffOperation, DiffOperationForward},
};

/// `K`, shape `[out, in]`, fixed at construction.
#[derive(Debug, Clone)]
pub struct FixedLinearMap<B: Backend> {
    matrix: Tensor<B, 2>,
}

impl<B: Backend> FixedLinearMap<B> {
    pub fn new(matrix: Tensor<B, 2>) -> Self {
        Self { matrix }
    }

    /// Build `K` from row-major host data, through `TensorData` — never `Tensor::from_floats`,
    /// which rounds through `f32` even on an `f64` backend.
    pub fn from_row_major(rows: usize, cols: usize, data: &[f64], device: &B::Device) -> Self {
        let tensor = Tensor::<B, 2>::from_data(
            TensorData::new(data.to_vec(), [rows, cols]).convert::<B::FloatElem>(),
            device,
        );
        Self::new(tensor)
    }

    fn dims(&self) -> [usize; 2] {
        self.matrix.dims()
    }

    /// `K` broadcast to `[batch, out, in]` — `K` itself carries no batch dimension, so every
    /// frame-shaped mode has to repeat it explicitly; burn's `matmul` does not implicitly
    /// broadcast a batch-1 operand against a wider one.
    fn batched(&self, batch: usize) -> Tensor<B, 3> {
        self.matrix.clone().unsqueeze_dim::<3>(0).repeat_dim(0, batch)
    }

    /// `Kᵀ` broadcast to `[batch, in, out]`.
    fn batched_transpose(&self, batch: usize) -> Tensor<B, 3> {
        self.matrix
            .clone()
            .transpose()
            .unsqueeze_dim::<3>(0)
            .repeat_dim(0, batch)
    }
}

impl<B: Backend> DiffOperation<B> for FixedLinearMap<B> {
    type VjpState = bool;
    type HvpState = ();

    fn forward(
        &self,
        inputs: Vec<Input<B>>,
        diff_state: DiffModeState<B>,
    ) -> DiffOperationForward<B, Self::VjpState, Self::HvpState> {
        let inputs: Vec<Point<B>> = inputs.into_iter().map(Input::unwrap_float).collect();
        let [x] = inputs.try_into().unwrap();
        let batch = x.dims()[0];
        // Plain rank-2 matmul: x is [batch, in] as a literal matrix, Kᵀ is [in, out] — no
        // batching broadcast needed here, unlike the frame-shaped modes below.
        let y = x.clone().matmul(self.matrix.clone().transpose());
        match diff_state {
            DiffModeState::ForwardOnly => DiffOperationForward::forward_only(y),
            DiffModeState::Jvp(tvecs) => {
                let [dx] = tvecs.try_into().unwrap();
                let dy = dx.map(|dx| self.batched(batch).matmul(dx));
                DiffOperationForward::jvp(y, dy)
            }
            DiffModeState::Vjp(diffs) => DiffOperationForward::vjp(y, diffs[0]),
            DiffModeState::Hvp { diffs, tangents } => {
                let [dx] = tangents.try_into().unwrap();
                let dy = dx.map(|dx| {
                    self.batched(batch)
                        .matmul(dx.unsqueeze_dim(2))
                        .squeeze_dim(2)
                });
                DiffOperationForward::hvp(y, dy, diffs[0], ())
            }
        }
    }

    fn vjp(
        &self,
        state: &Self::VjpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        let batch = omega.dims()[0];
        vec![state.then(|| self.batched_transpose(batch).matmul(omega))]
    }

    fn hvp(
        &self,
        state: &Self::VjpState,
        _: &Self::HvpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        vec![state.then(|| omega.zeros_like())]
    }
}

/// Register `x` mapped through a fixed linear map, checking `map`'s shape against `x`'s and
/// `out`'s manifold dimensions first.
pub fn linear_map<B, I, O>(x: &Q<B, I>, out: O, map: FixedLinearMap<B>) -> SkoroResult<Q<B, O>>
where
    B: Backend,
    I: ContinuousQuantity<B>,
    O: ContinuousQuantity<B>,
{
    let [rows, cols] = map.dims();
    let out_dim = out.dimension();
    let in_dim = x.get_quantity_type::<Continuous>().dimension();
    if rows != out_dim {
        return Err(SkoroError::ShapeMismatch {
            expected: out_dim,
            found: rows,
        });
    }
    if cols != in_dim {
        return Err(SkoroError::ShapeMismatch {
            expected: in_dim,
            found: cols,
        });
    }
    register_map1(out, map, x)
}
```

- [ ] **Step 4: Wire up the module**

In `crates/skoro-linalg/src/matrix/mod.rs`:

```rust
mod linear_map;
mod matmul;
mod transpose;
mod utils;

pub use linear_map::*;
pub use matmul::*;
pub use transpose::*;
pub use utils::*;
```

- [ ] **Step 5: Run the tests**

Run: `cargo test -p skoro-linalg --test linear_map`
Expected: PASS. If `hvp_is_zero` or the exact `evaluate_hvp` call shape needed adjustment in Step 1, iterate there first — do not weaken the assertion to something that would pass with a nonzero Hessian.

- [ ] **Step 6: Numerically verify the derivatives (SKILL.md checklist item 10)**

Add one more test to `crates/skoro-linalg/tests/linear_map.rs`, using the harness in `skoro_model::test_utils` (the crate's dev-dependencies already enable the `testing` feature; the functions are imported in Step 1):

```rust
#[test]
fn derivatives_agree_with_finite_differences() {
    let model = ModelClient::<B>::default();
    let x = model.add_parameter(EuclideanD(2));
    let y = linear_map(&x, EuclideanD(3), k()).unwrap();
    let point = points().with_quantity(&x, row(&[0.3, -1.2])).build();
    test_jvp_numerically(&x, &y, point.clone(), 1e-6, 1e-8);
    test_vjp_numerically(&x, &y, point.clone(), 1e-6, 1e-8);
    test_hvp_numerically(&x, &y, point, 1e-6, 1e-8);
}
```

A one-input op has no mixed blocks, so `test_hvp_cross_numerically` does not apply.

- [ ] **Step 7: Run the Phase 2 checkpoint**

Run: `cargo test -p skoro-linalg`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add crates/skoro-linalg/src/matrix/linear_map.rs crates/skoro-linalg/src/matrix/mod.rs \
  crates/skoro-linalg/tests/linear_map.rs
git commit -m "$(cat <<'EOF'
Add FixedLinearMap and the linear_map sugar

A backend-generic DiffOperation for y = Kx with a host-fixed K,
needed because a Distribution cannot add its own constant nodes
(Q.client is pub(crate)) but the Gaussian Fisher root (R24) and the
joint-factor whitening root (R26) both need one (R2-R4).
EOF
)"
```

**Phase 2 checkpoint (already run in Task 5, Step 7):** `cargo test -p skoro-linalg` passes.

---

## Phase 3 — `skoro-model`: roles, handles, factors, Fisher assembly (additive)

Implements R12, R13, R14's `Factor` struct (the `Distribution` trait's `log_density` method stays as-is until Phase 4a), R15, R16's `add_factor`/`bind` (not yet the new `add_quantity`), R17, R19, R21, R22, R23, and the tests in spec §5.4 (`crates/skoro-model/tests/terms.rs`), §5.5 (`crates/skoro-model/tests/jacobian.rs`, new tests added to the existing file), and the `add_factor_*` tests of §5.3 (added to `crates/skoro-model/tests/residual.rs` for now — Phase 4b is what renames that file to `bindings.rs` and removes the record-based tests). Checkpoint: `cargo test --workspace` passes with the record machinery still present (§6, Phase 3).

**Watch out — a temporary method-name collision.** `ModelBuilderOps::factors(&self) -> Vec<FactorRecord>` (old, still present) and `ModelClientOps::factors(&self) -> Vec<FactorInfo>` (new, added in Task 11) are both implemented for `ModelClient<B>`. Since `ModelBuilderOps: ModelClientOps` is a supertrait relationship, Rust does **not** prefer one over the other for an unqualified `model.factors()` call when both traits are in scope — it is a genuine `E0034` ambiguous-method-call error. Any existing test file that imports both traits (check with `grep -rln "ModelBuilderOps" crates/skoro-model/tests crates/skoro-normal/tests crates/skoro/tests` — `residual.rs` and `stacked_tangent.rs` both do) and calls unqualified `.factors()` will stop compiling the moment Task 11 lands. Fix those call sites in the same commit as Task 11: change `model.factors()` to `ModelBuilderOps::factors(&model)` wherever the old `FactorRecord`-returning meaning is intended. This is temporary — Phase 4b's Task 20 deletes `factors` from `ModelBuilderOps` entirely, at which point every one of those call sites reverts to plain `.factors()` (now unambiguously the new `FactorInfo` meaning). Do the revert as part of Task 20, not before.

### Task 6: `roles.rs` (R12)

**Files:**
- Create: `crates/skoro-model/src/roles.rs`
- Modify: `crates/skoro-model/src/lib.rs` (`pub mod roles;`)

**Interfaces:**
- Produces: `skoro_model::roles::{Role, ARGUMENT, RESIDUAL, NORMALIZER_PARENT, LOCATION, WHITENING, FISHER_ROOT}`, consumed by every task from here through Phase 5.

No test of its own — these are constants, exercised indirectly by every later test that binds a role.

- [ ] **Step 1: Write `crates/skoro-model/src/roles.rs`**

```rust
//! The binding roles this crate defines. See `docs/superpowers/specs/2026-09-01-factor-bindings-and-fisher-root-spec.md` §2/§3.4 for the full contract table.

pub use skoro_graph::binding::Role;

/// The term's local views are expressed over these nodes, in this order. Cardinality ≥ 1 per
/// term; nodes must be distinct and continuous.
pub const ARGUMENT: Role = Role("argument");

/// An `EuclideanD` node `r` with `−log p = ½‖r‖² + n`. Cardinality 0 or 1 per term.
pub const RESIDUAL: Role = Role("residual");

/// Every node the normalizer `n` depends on. Any cardinality per term.
pub const NORMALIZER_PARENT: Role = Role("normalizer_parent");

/// `μ` in `r = L⁻¹(x ⊖ μ)`. Cardinality 0 or 1 per term.
pub const LOCATION: Role = Role("location");

/// `L` in `r = L⁻¹(x ⊖ μ)`, with `Σ = LLᵀ`. Cardinality 0 or 1 per term.
pub const WHITENING: Role = Role("whitening");

/// `EuclideanMatrixD(k, Σ dim(ARGUMENT))` with `CᵀC` equal to the expected information over the
/// arguments, in `ARGUMENT` order. Cardinality 0 or 1 per term.
pub const FISHER_ROOT: Role = Role("fisher_root");
```

- [ ] **Step 2: Declare the module**

In `crates/skoro-model/src/lib.rs`, add `pub mod roles;` next to the other `pub mod` lines.

- [ ] **Step 3: Verify**

Run: `cargo build -p skoro-model`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add crates/skoro-model/src/roles.rs crates/skoro-model/src/lib.rs
git commit -m "$(cat <<'EOF'
Add binding role constants (R12)

ARGUMENT, RESIDUAL, NORMALIZER_PARENT, LOCATION, WHITENING,
FISHER_ROOT — the vocabulary every later task binds against.
EOF
)"
```

### Task 7: `Handle<B>` (R13)

**Files:**
- Modify: `crates/skoro-model/src/q.rs` (new `Handle` type)

**Interfaces:**
- Consumes: `crate::model::NodeDropGuard` (already `pub(crate)`, defined in `model.rs`).
- Produces: `skoro_model::q::Handle`, consumed by `Factor` (Task 8) and `FactorInfo`/`ModelServer::add_factor` (Task 9).

No standalone test — `Handle` is exercised through `Factor::bind` in Task 8's tests.

- [ ] **Step 1: Add `Handle` to `crates/skoro-model/src/q.rs`**

Add `use crate::model::NodeDropGuard;` to the existing `use crate::model::{ModelClient, NodeDropGuard};` import (it already imports `NodeDropGuard` — check the top of the file; if it does not, add it). Then append near the end of the file, before the closing comment about `mark_likelihood`:

```rust
/// A cloneable, type-erased reference to a quantity — what [`Factor::bind`](crate::distribution::Factor::bind)
/// stores instead of a `Q<B, T>`, since a `Factor` collects bindings of heterogeneous quantity
/// types in one `Vec`.
///
/// Carries the same drop guard as the `Q` it was built from, so the bound node is kept alive by
/// the collector for exactly as long as *some* handle to it — `Q` or `Handle` — survives.
pub struct Handle<B: Backend> {
    pub(crate) id: Id,
    pub(crate) client: ModelClient<B>,
    pub(crate) guard: Arc<NodeDropGuard>,
}

impl<B: Backend> Handle<B> {
    /// The identifier of the node this handle names.
    #[inline(always)]
    pub fn id(&self) -> Id {
        self.id
    }
}

impl<B: Backend, T> From<&Q<B, T>> for Handle<B> {
    fn from(q: &Q<B, T>) -> Self {
        Self {
            id: q.id,
            client: q.client.clone(),
            guard: q.guard.clone(),
        }
    }
}

impl<B: Backend> Clone for Handle<B> {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            client: self.client.clone(),
            guard: self.guard.clone(),
        }
    }
}

impl<B: Backend> std::fmt::Debug for Handle<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Handle")
            .field("id", &self.id)
            .field("server_id", &self.client.server_id)
            .finish_non_exhaustive()
    }
}
```

- [ ] **Step 2: Verify**

Run: `cargo build -p skoro-model`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add crates/skoro-model/src/q.rs
git commit -m "$(cat <<'EOF'
Add Handle<B>: a type-erased, cloneable Q reference (R13)

Factor collects bindings of heterogeneous quantity types in one Vec,
which Q<B, T>'s type parameter does not allow; Handle erases T while
keeping the same drop guard, so a bound node stays alive exactly as
long as some handle — Q or Handle — to it survives.
EOF
)"
```

### Task 8: `Factor<B>` struct (R14, struct only — trait unchanged until Phase 4a)

**Files:**
- Modify: `crates/skoro-model/src/distribution.rs` (add `Factor`, leave `Distribution::log_density` as-is)

**Interfaces:**
- Consumes: `crate::q::Handle` (Task 7), `skoro_graph::binding::Role`.
- Produces: `skoro_model::distribution::Factor`, consumed by `ModelServer::add_factor` (Task 9).

No standalone test yet — `Factor::new`/`bind` are exercised by Task 9's `add_factor_*` tests.

- [ ] **Step 1: Add `Factor` to `crates/skoro-model/src/distribution.rs`**

Add imports `use skoro_graph::binding::Role;` and `use crate::q::Handle;` at the top, then append:

```rust
/// What a distribution (or a free-standing builder) hands back to describe one log-density
/// term: the term itself, plus every labelled binding it wants recorded once the term is
/// marked. The `Handle`s are load-bearing — they are the roots that keep a bound node (a
/// residual, a Fisher root) alive between the moment this is built and the moment
/// [`ModelServer::add_factor`](crate::model::ModelServer::add_factor) records its bindings.
pub struct Factor<B: Backend> {
    pub log_density: Q<B, Real>,
    pub bindings: Vec<(Role, Handle<B>)>,
}

impl<B: Backend> Factor<B> {
    pub fn new(log_density: Q<B, Real>) -> Self {
        Self {
            log_density,
            bindings: Vec::new(),
        }
    }

    /// Queue `(role, q)` to be bound once this factor is recorded.
    pub fn bind<T>(mut self, role: Role, q: &Q<B, T>) -> Self {
        self.bindings.push((role, Handle::from(q)));
        self
    }
}
```

- [ ] **Step 2: Verify**

Run: `cargo build -p skoro-model`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add crates/skoro-model/src/distribution.rs
git commit -m "$(cat <<'EOF'
Add the Factor<B> struct (R14, struct only)

A log-density term plus the labelled bindings its builder wants
recorded. Distribution::log_density is untouched here — Phase 4a
replaces it with a Distribution::factor method returning this type.
EOF
)"
```

### Task 9: `ModelServer::{bind, add_factor}`, `ModelClient::{bind, add_factor}` (R15, R16's `add_factor`/`bind` only)

**Files:**
- Modify: `crates/skoro-model/src/model.rs` (two new `ModelServer` methods, two new `ModelClient` methods)
- Modify: `crates/skoro-model/src/client_ops.rs` (`ModelBuilderOps` gains `add_factor`/`bind`, alongside the existing `record_factor`/`factors`/etc. — not yet removed)
- Test: `crates/skoro-model/tests/residual.rs` (new tests, per §5.3's `add_factor_*` rows)

**Interfaces:**
- Consumes: `Factor<B>` (Task 8), `roles::ARGUMENT` (Task 6), `Graph::bind` (Phase 1 Task 2).
- Produces: `ModelServer::{bind, add_factor}`, `ModelClient::{bind, add_factor}`, `ModelBuilderOps::{bind, add_factor}` — consumed by `JointNormal::factor`/`add_joint_factor` in Phase 4a (Task 18).

- [ ] **Step 1: Write the failing tests**

Append to `crates/skoro-model/tests/residual.rs`, inside a new `mod bindings_via_add_factor` (keep the existing `mod records`/`mod rejections`/`mod compile_fail` untouched — Phase 4b renames the whole file):

```rust
mod bindings_via_add_factor {
    use super::*;
    use skoro_model::{
        distribution::Factor,
        roles::{ARGUMENT, RESIDUAL},
    };

    #[test]
    fn add_factor_marks_and_binds() {
        let model = ModelClient::<B>::default();
        let a = model.add_parameter(Real);
        let b = model.add_parameter(Real);
        let term = register_map1(Real, LiteralScalarMultiplication::new(-0.5), &a).unwrap();
        let r = model.add_parameter(EuclideanD(2));

        let factor = Factor::new(term.clone()).bind(RESIDUAL, &r);
        model.add_factor(factor, &[a.id(), b.id()]).unwrap();

        let info = ModelClientOps::factors(&model);
        assert_eq!(info.len(), 1);
        assert_eq!(info[0].scored, None);
        assert_eq!(info[0].kind, skoro_graph::graph::TermKind::Weight);
        assert_eq!(info[0].arguments, vec![a.id(), b.id()]);
        assert_eq!(ModelClientOps::bound(&model, term.id(), RESIDUAL), vec![r.id()]);
    }

    #[test]
    fn add_factor_needs_an_argument() {
        let model = ModelClient::<B>::default();
        let a = model.add_parameter(Real);
        let term = register_map1(Real, LiteralScalarMultiplication::new(-0.5), &a).unwrap();

        assert!(matches!(
            model.add_factor(Factor::new(term), &[]),
            Err(SkoroError::ShapeMismatch { expected: 1, found: 0 })
        ));
    }
}
```

These tests read `ModelClientOps::{factors, bound}`, which Task 11 adds. Implement this task's code (Steps 2-5), then Task 11, then run both tasks' tests together; committing the two tasks as one commit is fine.

- [ ] **Step 2: Implement `ModelServer::bind`**

In `crates/skoro-model/src/model.rs`, near `mark_likelihood` (~line 178):

```rust
    /// Record a binding: `to` plays `role` for `from`. Delegates to
    /// [`Graph::bind`](skoro_graph::graph::Graph::bind).
    pub fn bind(&mut self, from: Id, role: Role, to: Id) -> SkoroResult<()> {
        self.graph.bind(from, role, to)
    }
```

Add `use skoro_graph::binding::Role;` and `use crate::roles::ARGUMENT;` to the top of `model.rs`.

- [ ] **Step 3: Implement `ModelServer::add_factor`**

```rust
    /// Mark `factor.log_density` as a free-standing likelihood term, then bind `ARGUMENT` to
    /// each of `arguments` in order, then every `(role, handle)` `factor` declared. Refuses an
    /// empty `arguments` before mutating anything — a term with no argument has no local view.
    pub fn add_factor(&mut self, factor: &Factor<B>, arguments: &[Id]) -> SkoroResult<()> {
        if arguments.is_empty() {
            return Err(SkoroError::ShapeMismatch {
                expected: 1,
                found: 0,
            });
        }
        self.mark_likelihood(&factor.log_density)?;
        let density = factor.log_density.id;
        for &arg in arguments {
            self.bind(density, ARGUMENT, arg)?;
        }
        for (role, handle) in &factor.bindings {
            self.bind(density, *role, handle.id)?;
        }
        Ok(())
    }
```

Add `use crate::distribution::Factor;` to `model.rs`'s imports.

- [ ] **Step 4: `ModelClient::{bind, add_factor}`**

Near `ModelClient::mark_likelihood` (~line 893):

```rust
    /// Record a binding. See [`ModelServer::bind`].
    pub fn bind(&self, from: Id, role: Role, to: Id) -> SkoroResult<()> {
        self.server.write().unwrap().bind(from, role, to)
    }

    /// Attach a free-standing likelihood factor, with its declared bindings. See
    /// [`ModelServer::add_factor`].
    pub fn add_factor(&self, factor: Factor<B>, arguments: &[Id]) -> SkoroResult<Q<B, Real>> {
        self.server.write().unwrap().add_factor(&factor, arguments)?;
        Ok(factor.log_density)
    }
```

- [ ] **Step 5: `ModelBuilderOps` gains `bind`/`add_factor` (alongside the old methods, not replacing them yet)**

In `crates/skoro-model/src/client_ops.rs`, add to the `ModelBuilderOps` trait definition (after `stacked_tangent`, before `record_factor`):

```rust
    /// Attach a free-standing likelihood factor to real-valued term with its declared
    /// bindings, and bind `ARGUMENT` to each of `arguments` in order.
    fn add_factor(&self, factor: Factor<B>, arguments: &[Id]) -> SkoroResult<Q<B, Real>>;

    /// Record a binding: `to` plays `role` for `from`.
    fn bind(&self, from: Id, role: Role, to: Id) -> SkoroResult<()>;
```

and to the `impl<B: Backend> ModelBuilderOps<B> for ModelClient<B>` block:

```rust
    #[inline(always)]
    fn add_factor(&self, factor: Factor<B>, arguments: &[Id]) -> SkoroResult<Q<B, Real>> {
        ModelClient::add_factor(self, factor, arguments)
    }

    #[inline(always)]
    fn bind(&self, from: Id, role: Role, to: Id) -> SkoroResult<()> {
        ModelClient::bind(self, from, role, to)
    }
```

Add `use crate::distribution::Factor;` and `use skoro_graph::binding::Role;` to `client_ops.rs`'s imports.

- [ ] **Step 6: Run the tests (after Task 11 lands — see Step 1's note)**

Run: `cargo test -p skoro-model --test residual bindings_via_add_factor`
Expected: PASS once Task 11's `ModelClientOps::{factors, bound}` exist.

- [ ] **Step 7: Commit**

```bash
git add crates/skoro-model/src/model.rs crates/skoro-model/src/client_ops.rs \
  crates/skoro-model/tests/residual.rs
git commit -m "$(cat <<'EOF'
Add ModelServer/ModelClient::{bind,add_factor} (R15/R16 partial)

add_factor marks a free-standing term and binds ARGUMENT plus every
role the caller's Factor declared; bind is the raw single-triple
primitive. Both land on ModelBuilderOps alongside the still-present
record_factor/factors/residual_layout — Phase 4b removes those.
EOF
)"
```

### Task 10: `ModelServer::cleanup` roots bound targets (R17)

**Files:**
- Modify: `crates/skoro-model/src/model.rs` (`cleanup`, ~line 434-492)
- Test: `crates/skoro-model/tests/residual.rs`, `mod bindings_via_add_factor` (one more test)

**Interfaces:**
- Consumes: `Graph::bindings_of` (Phase 1 Task 2).

- [ ] **Step 1: Write the failing test**

Append to `mod bindings_via_add_factor`:

```rust
    #[test]
    fn a_bound_node_survives_cleanup() {
        let model = ModelClient::<B>::default();
        let a = model.add_parameter(Real);
        let term = register_map1(Real, LiteralScalarMultiplication::new(-0.5), &a).unwrap();
        let r = model.add_parameter(EuclideanD(2));
        let r_id = r.id();

        let factor = Factor::new(term).bind(RESIDUAL, &r);
        model.add_factor(factor, &[a.id()]).unwrap();
        drop(r);

        model.cleanup();
        assert!(model.server_call(|s| s.graph().contains(r_id)));
    }
```

- [ ] **Step 2: Run the test to see it fail**

Run: `cargo test -p skoro-model --test residual a_bound_node_survives_cleanup`
Expected: FAIL — `cleanup`'s mark phase does not yet follow bindings, so `r` (whose only `Q` handle was just dropped) is swept.

- [ ] **Step 3: Update the root set and the mark loop**

In `crates/skoro-model/src/model.rs`, `pub fn cleanup` (~line 434), **leave the root set alone**, including the `roots.extend(self.registry.factors()...)` block. It is not redundant yet: `facade::add_gaussian_quantity` still builds a second residual whose `Q` is dropped at the end of that function, so the record is that node's only reference until Task 16 rebuilds the Gaussian factor on bindings. Task 17 deletes the block.

Then, inside the mark `while let Some(id) = queue.pop_front()` loop, after the existing `stochastic_of` block:

```rust
            // The scored quantity of a density, which need not be among its parents.
            if let Some(stochastic) = self.graph.stochastic_of(id)
                && marked.insert(stochastic)
            {
                queue.push_back(stochastic);
            }
```

add:

```rust
            for (_, to) in self.graph.bindings_of(id) {
                if marked.insert(*to) {
                    queue.push_back(*to);
                }
            }
```

**Verify the sweep still runs as one batch call.** `cleanup`'s sweep already computes the whole `garbage` set and calls `self.graph.remove(&ordered)` exactly once (not per-node) — this is what makes `demotion_sweeps_the_bindings_with_the_density` (Task 19, Phase 4b) work: a demoted density (retired to `Mapped`, off `log_likelihoods`, unmarked) and its bound residual (also unmarked, since nothing else marks it) both land in `garbage` and are removed together, so `remove`'s `NodeStillBound` check never fires on either — the referrer is in the same batch. Do not change the sweep to remove nodes one at a time; confirm this by reading the current `remove(&ordered)` call at the end of `cleanup` and leaving it untouched.

- [ ] **Step 4: Run the test**

Run: `cargo test -p skoro-model --test residual bindings_via_add_factor`
Expected: PASS.

- [ ] **Step 5: Run the full `skoro-model` suite**

Run: `cargo test -p skoro-model`
Expected: PASS.

- [ ] **Step 6: Run `cargo test -p skoro-normal` too, since Gaussian factors are still registry-based here**

Run: `cargo test -p skoro-normal`
Expected: PASS (the record roots kept in Step 3 are what keep `add_gaussian_quantity`'s second residual alive here).

- [ ] **Step 7: Commit**

```bash
git add crates/skoro-model/src/model.rs crates/skoro-model/tests/residual.rs
git commit -m "$(cat <<'EOF'
cleanup marks binding targets reachable (R17)

The collector's mark phase now also walks graph.bindings_of(id) for
every marked node, so a bound residual/root survives with its
referrer even after every Q naming it directly is dropped. The
record roots stay until Phase 4b deletes the registry.
EOF
)"
```

### Task 11: `ModelClientOps` gains `factors`/`bindings_of`/`bound`/`terms`/`log_proposal`/`log_weight`, and `FactorInfo` (R19)

**Files:**
- Modify: `crates/skoro-model/src/client_ops.rs` (trait methods, impl, `FactorInfo` struct)
- Test: `crates/skoro-model/tests/residual.rs` (`mod bindings_via_add_factor`, finish the tests started in Task 9; add `factors_are_in_term_order`)

**Interfaces:**
- Consumes: `Graph::{terms, bindings_of, bound}` (Phase 1 Task 2 & 4), `roles::ARGUMENT`.
- Produces: `ModelClientOps::{factors, bindings_of, bound, terms, log_proposal, log_weight}`, `FactorInfo` — consumed by `least_squares::residual_layout` (Task 12), `information.rs` (Task 14), and every Phase 4/5 test.

- [ ] **Step 1: Finish the tests from Task 9, and add one more**

`add_factor_marks_and_binds` and `add_factor_needs_an_argument` (written in Task 9) now compile against real methods. Append one more to `mod bindings_via_add_factor`:

```rust
    #[test]
    fn factors_are_in_term_order() {
        let model = ModelClient::<B>::default();
        let mut densities = Vec::new();
        for i in 0..3 {
            let a = model.add_parameter(Real);
            let term =
                register_map1(Real, LiteralScalarMultiplication::new(-0.5 - i as f64), &a)
                    .unwrap();
            model
                .add_factor(Factor::new(term.clone()), &[a.id()])
                .unwrap();
            densities.push(term.id());
        }
        let info = ModelClientOps::factors(&model);
        assert_eq!(
            info.iter().map(|f| f.density).collect::<Vec<_>>(),
            densities
        );
    }
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `cargo test -p skoro-model --test residual bindings_via_add_factor`
Expected: FAIL to compile — `ModelClientOps::factors`/`bound` do not exist.

- [ ] **Step 3: `FactorInfo` and the trait methods**

In `crates/skoro-model/src/client_ops.rs`, add imports `use skoro_graph::graph::{Term, TermKind};` and `use crate::roles::{ARGUMENT, Role};` (note: `Role` was already added in Task 9). Add near the top, before the `ModelClientOps` trait:

```rust
/// One term of the model's objective, as read back through [`ModelClientOps::factors`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FactorInfo {
    pub density: Id,
    pub scored: Option<Id>,
    pub kind: TermKind,
    pub arguments: Vec<Id>,
}
```

Add to the `ModelClientOps` trait (after `log_density_of`):

```rust
    /// One [`FactorInfo`] per term, in term order.
    fn factors(&self) -> Vec<FactorInfo>;

    /// Every `(role, to)` pair `id` has bound, in binding order.
    fn bindings_of(&self, id: Id) -> Vec<(Role, Id)>;

    /// The `to`s `id` has bound under `role`, in binding order.
    fn bound(&self, id: Id, role: Role) -> Vec<Id>;

    /// One [`Term`] per id in the graph's likelihood terms, in marking order.
    fn terms(&self) -> Vec<Term>;

    /// The sum of every [`TermKind::Proposal`] term's log-density at `points`.
    fn log_proposal(&self, points: &ValueMap<B>) -> SkoroResult<Point<B>>;

    /// The sum of every [`TermKind::Weight`] term's log-density at `points`.
    fn log_weight(&self, points: &ValueMap<B>) -> SkoroResult<Point<B>>;
```

Add to `impl<B: Backend> ModelClientOps<B> for ModelClient<B>`:

```rust
    #[inline(always)]
    fn factors(&self) -> Vec<FactorInfo> {
        self.server_call(|server| {
            server
                .graph()
                .terms()
                .into_iter()
                .map(|term| FactorInfo {
                    density: term.density,
                    scored: term.scored,
                    kind: term.kind,
                    arguments: server.graph().bound(term.density, ARGUMENT),
                })
                .collect()
        })
    }

    #[inline(always)]
    fn bindings_of(&self, id: Id) -> Vec<(Role, Id)> {
        self.server_call(|server| server.graph().bindings_of(id).to_vec())
    }

    #[inline(always)]
    fn bound(&self, id: Id, role: Role) -> Vec<Id> {
        self.server_call(|server| server.graph().bound(id, role))
    }

    #[inline(always)]
    fn terms(&self) -> Vec<Term> {
        self.server_call(|server| server.graph().terms())
    }

    #[inline(always)]
    fn log_proposal(&self, points: &ValueMap<B>) -> SkoroResult<Point<B>> {
        let ids: Vec<Id> = ModelClientOps::terms(self)
            .into_iter()
            .filter(|t| t.kind == TermKind::Proposal)
            .map(|t| t.density)
            .collect();
        self.server_call(|server| server.graph().log_density_terms(points, &ids))
    }

    #[inline(always)]
    fn log_weight(&self, points: &ValueMap<B>) -> SkoroResult<Point<B>> {
        let ids: Vec<Id> = ModelClientOps::terms(self)
            .into_iter()
            .filter(|t| t.kind == TermKind::Weight)
            .map(|t| t.density)
            .collect();
        self.server_call(|server| server.graph().log_density_terms(points, &ids))
    }
```

- [ ] **Step 4: Fix the ambiguity this creates at existing `.factors()` call sites**

Per this phase's "Watch out" note: `grep -rn "\.factors()" crates/skoro-model/tests crates/skoro-normal/tests crates/skoro-normal/src crates/skoro/tests` and change every unqualified `model.factors()` (or `client.factors()`) call that means the *old* `FactorRecord` API to `ModelBuilderOps::factors(&model)` (adjust the variable name per call site). `crates/skoro-model/tests/residual.rs`'s existing `mod records`/`mod rejections` tests and `crates/skoro/tests/stacked_tangent.rs`'s `mod records`/`mod retirement` are the ones to check; `crates/skoro-normal/src/facade.rs`'s `gaussian_blocks` also calls `client.factors()` internally — qualify it there too: `GaussianBlock::all(&ModelBuilderOps::factors(client))`.

- [ ] **Step 5: Run the tests**

Run: `cargo test -p skoro-model --test residual`
Expected: PASS, both `mod records`/`mod rejections` (now qualified) and `mod bindings_via_add_factor`.

- [ ] **Step 6: Run the full workspace**

Run: `cargo test --workspace`
Expected: PASS (this is a good point to also run the Phase 3 checkpoint early, since Tasks 12-14 build on this one but do not change existing behavior).

- [ ] **Step 7: Commit**

```bash
git add crates/skoro-model/src/client_ops.rs crates/skoro-model/tests/residual.rs \
  crates/skoro-normal/src/facade.rs crates/skoro/tests/stacked_tangent.rs
git commit -m "$(cat <<'EOF'
Add ModelClientOps::{factors,bindings_of,bound,terms,log_proposal,log_weight} (R19)

FactorInfo is the binding-based read side of a term: density, scored,
kind and arguments (bound(density, ARGUMENT)). Existing .factors()
call sites meaning the old FactorRecord API are qualified as
ModelBuilderOps::factors(..) for the duration of the coexistence
window (removed again in Phase 4b, Task 20).
EOF
)"
```

### Task 12: `least_squares::residual_layout` (R21)

**Files:**
- Create: `crates/skoro-model/src/least_squares.rs`
- Modify: `crates/skoro-model/src/lib.rs` (`pub mod least_squares;`)
- Test: `crates/skoro-model/tests/residual.rs` (`mod bindings_via_add_factor`, one more test)

**Interfaces:**
- Consumes: `ModelClientOps::{factors, bound}` (Task 11), `roles::RESIDUAL`, `TangentLayout` (existing), `ModelClientOps::tangent_layout` (existing).

- [ ] **Step 1: Write the failing test**

Append to `mod bindings_via_add_factor`:

```rust
    #[test]
    fn residual_layout_follows_term_order_and_dimension() {
        let model = ModelClient::<B>::default();
        let a = model.add_parameter(Real);
        let term_a = register_map1(Real, LiteralScalarMultiplication::new(-0.5), &a).unwrap();
        let r2 = model.add_parameter(EuclideanD(2));
        model
            .add_factor(Factor::new(term_a).bind(RESIDUAL, &r2), &[a.id()])
            .unwrap();

        let b = model.add_parameter(Real);
        let term_b = register_map1(Real, LiteralScalarMultiplication::new(-0.5), &b).unwrap();
        let r3 = model.add_parameter(EuclideanD(3));
        model
            .add_factor(Factor::new(term_b).bind(RESIDUAL, &r3), &[b.id()])
            .unwrap();

        let layout = skoro_model::least_squares::residual_layout(&model).unwrap();
        assert_eq!(layout.total(), 5);
        assert_eq!(layout.ids().collect::<Vec<_>>(), vec![r2.id(), r3.id()]);
    }
```

- [ ] **Step 2: Run the test to see it fail**

Run: `cargo test -p skoro-model --test residual residual_layout_follows_term_order_and_dimension`
Expected: FAIL — `skoro_model::least_squares` does not exist.

- [ ] **Step 3: Write `crates/skoro-model/src/least_squares.rs`**

```rust
//! Reading the rows of a least-squares `J` off the model's bindings.

use skoro_backend::Backend;
use skoro_core::{
    error::{SkoroError, SkoroResult},
    id::Id,
};
use skoro_graph::tangent_layout::TangentLayout;

use crate::client_ops::ModelClientOps;
use crate::roles::RESIDUAL;

/// A layout over every term's `RESIDUAL` binding, in term order. A term with no residual is
/// skipped; a term with more than one is [`SkoroError::ShapeMismatch`].
pub fn residual_layout<B: Backend, C: ModelClientOps<B>>(client: &C) -> SkoroResult<TangentLayout> {
    let mut ids: Vec<Id> = Vec::new();
    for factor in client.factors() {
        let residuals = client.bound(factor.density, RESIDUAL);
        match residuals.len() {
            0 => continue,
            1 => ids.push(residuals[0]),
            n => {
                return Err(SkoroError::ShapeMismatch {
                    expected: 1,
                    found: n,
                });
            }
        }
    }
    client.tangent_layout(&ids)
}
```

- [ ] **Step 4: Declare the module**

In `crates/skoro-model/src/lib.rs`, add `pub mod least_squares;`.

- [ ] **Step 5: Run the test**

Run: `cargo test -p skoro-model --test residual residual_layout_follows_term_order_and_dimension`
Expected: PASS.

- [ ] **Step 6: Run the full `skoro-model` suite**

Run: `cargo test -p skoro-model`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add crates/skoro-model/src/least_squares.rs crates/skoro-model/src/lib.rs \
  crates/skoro-model/tests/residual.rs
git commit -m "$(cat <<'EOF'
Add least_squares::residual_layout (R21)

The bindings-based replacement for ModelServer::residual_layout:
collects bound(density, RESIDUAL) per term, in term order.
EOF
)"
```

### Task 13: Jacobian rows may be leaves (R22)

**Files:**
- Modify: `crates/skoro-model/src/jacobian.rs` (delete `check_rows`, add the leaf-row fill)
- Test: `crates/skoro-model/tests/jacobian.rs` (existing file — add per spec §5.5; also fix or remove the existing `NotMapped`-expecting test at line ~231, since R22 deletes that error path)

**Interfaces:**
- Consumes: nothing new.
- Produces: the corrected `jacobian_jvp`/`jacobian_vjp`, consumed by `information_blocks` (Task 14).

- [ ] **Step 1: Read the existing test this changes**

`a_leaf_row_is_rejected` (`jacobian.rs:219-236`) asserts that a leaf row is an error. R22 makes it the identity instead, so Step 5 deletes that test once the two below cover the same fixture.

- [ ] **Step 2: Write the two new failing tests from spec §5.5**

Append to `crates/skoro-model/tests/jacobian.rs`, reusing its `Model` fixture: `x` is the first column of `inputs()`, so its identity block is the first entry of its row, and `D_IN` is 8.

```rust
/// A row that is a leaf and also a column is the identity on its own columns, in both paths.
#[test]
fn a_leaf_row_that_is_a_column_is_the_identity() {
    let model = Model::new();
    let points = model.points();
    let inputs = model.inputs();
    let outputs = layout(&model.client, &[model.u.id(), model.x.id()]);

    for j in [
        jacobian_jvp(&model.client, &points, &inputs, &outputs).unwrap(),
        jacobian_vjp(&model.client, &points, &inputs, &outputs).unwrap(),
    ] {
        assert_eq!(j.dims(), [BATCH, 2, D_IN]);
        let j = j.into_data().to_vec::<f64>().unwrap();
        for batch in 0..BATCH {
            // Row 1 of each batch is `x`'s row: `∂x/∂x = 1` in column 0, zero elsewhere.
            let row = &j[batch * 2 * D_IN + D_IN..batch * 2 * D_IN + 2 * D_IN];
            assert_eq!(row, &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        }
    }
}

/// A row that is a leaf but not a column is zero, in both paths — not an error.
#[test]
fn a_leaf_row_that_is_not_a_column_is_zero() {
    let model = Model::new();
    let c = model.client.add_constant(Real, &3.0f64);
    let points = model.points();
    let inputs = model.inputs();
    let outputs = layout(&model.client, &[model.u.id(), c.id()]);

    for j in [
        jacobian_jvp(&model.client, &points, &inputs, &outputs).unwrap(),
        jacobian_vjp(&model.client, &points, &inputs, &outputs).unwrap(),
    ] {
        assert_eq!(j.dims(), [BATCH, 2, D_IN]);
        let j = j.into_data().to_vec::<f64>().unwrap();
        for batch in 0..BATCH {
            let row = &j[batch * 2 * D_IN + D_IN..batch * 2 * D_IN + 2 * D_IN];
            assert_eq!(row, &[0.0; 8]);
        }
    }
}
```

- [ ] **Step 3: Run the tests to see them fail**

Run: `cargo test -p skoro-model --test jacobian`
Expected: FAIL — both new tests hit `NotMapped` from the current `check_rows`, while the existing `a_leaf_row_is_rejected` still passes.

- [ ] **Step 4: Delete `check_rows` and add the leaf-row fill**

In `crates/skoro-model/src/jacobian.rs`, delete the `check_rows` function (lines 72-90) entirely, and its two call sites (`check_rows(&out.primal, outputs)?;` in both `jacobian_jvp` and `jacobian_vjp`).

Add, above `jacobian_jvp`:

```rust
/// Overwrite the rows of `j` (`[batch, Out.total(), In.total()]`) belonging to any output
/// block that is a leaf (absent from `primal.evaluations` — a `Parameter`, `Stochastic`,
/// `Constant` or `Observed` node): the identity on its own columns if it also has an input
/// block, zero otherwise (already the case; nothing to do there).
fn fill_leaf_rows<B: Backend>(
    j: Tensor<B, 3>,
    primal: &ForwardOutput<B>,
    outputs: &TangentLayout,
    inputs: &TangentLayout,
    batch: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    let mut j = j;
    for r in outputs.blocks() {
        if primal.evaluations.get(&r.id).is_some() {
            continue;
        }
        let Some(c) = inputs.block(r.id) else {
            continue;
        };
        let identity = Tensor::<B, 2>::eye(r.dimension, device)
            .unsqueeze_dim::<3>(0)
            .repeat_dim(0, batch);
        j = j.slice_assign(
            [
                0..batch,
                r.offset..r.offset + r.dimension,
                c.offset..c.offset + c.dimension,
            ],
            identity,
        );
    }
    j
}
```

Update `jacobian_jvp`'s body:

```rust
pub fn jacobian_jvp<B: Backend, C: ModelClientOps<B>>(
    client: &C,
    points: &ValueMap<B>,
    inputs: &TangentLayout,
    outputs: &TangentLayout,
) -> SkoroResult<Tensor<B, 3>> {
    let (batch, device) = batch_and_device(points);
    let seeds = inputs.identity_frames(batch, &device);
    let ids: Vec<Id> = outputs.ids().collect();
    let out = client.evaluate_ids_jvp(points, &seeds, &ids)?;
    let j = outputs.flatten_frames(&out.tangents, batch, inputs.total(), &device)?;
    Ok(fill_leaf_rows(j, &out.primal, outputs, inputs, batch, &device))
}
```

and `jacobian_vjp`'s:

```rust
pub fn jacobian_vjp<B: Backend, C: ModelClientOps<B>>(
    client: &C,
    points: &ValueMap<B>,
    inputs: &TangentLayout,
    outputs: &TangentLayout,
) -> SkoroResult<Tensor<B, 3>> {
    let (batch, device) = batch_and_device(points);
    let seeds = outputs.identity_frames(batch, &device);
    let ids: Vec<Id> = outputs.ids().collect();
    let out = client.evaluate_ids_vjp(points, &seeds, &ids)?;
    let transposed = inputs.flatten_frames(&out.cotangents, batch, outputs.total(), &device)?;
    let j = transposed.swap_dims(1, 2);
    Ok(fill_leaf_rows(j, &out.primal, outputs, inputs, batch, &device))
}
```

If `Tensor::slice_assign`'s exact signature differs from `fn slice_assign<const D2: usize>(self, ranges: [Range<usize>; D2], values: Self) -> Self` (check the compiler error), adjust the call to match — the semantics to preserve are: overwrite exactly the `[0..batch] × [r.offset..r.offset+r.dimension] × [c.offset..c.offset+c.dimension]` sub-block of `j` with the batched identity, leaving everything else in `j` untouched.

Update the module doc comment (the `# Structural constraint` section, lines 14-45) to drop the "rows must be Mapped/LogDensity" claim and state instead: rows may be any continuous node; a leaf row is the identity on its own columns (if it has one) and zero elsewhere.

- [ ] **Step 5: Fix the now-obsolete `NotMapped` test**

Delete `a_leaf_row_is_rejected` (`jacobian.rs:219-236`): it asserted the behaviour R22 removes, and `a_leaf_row_that_is_a_column_is_the_identity` covers the same fixture. Drop the `SkoroError` import if nothing else in the file uses it.

- [ ] **Step 6: Run the tests**

Run: `cargo test -p skoro-model --test jacobian`
Expected: PASS.

- [ ] **Step 7: Run the full `skoro-model` suite**

Run: `cargo test -p skoro-model`
Expected: PASS — check for any other test relying on the old `NotMapped` behavior via `grep -rn "NotMapped" crates/skoro-model/tests`.

- [ ] **Step 8: Commit**

```bash
git add crates/skoro-model/src/jacobian.rs crates/skoro-model/tests/jacobian.rs
git commit -m "$(cat <<'EOF'
Jacobian rows may be leaves, not just Mapped nodes (R22)

Deletes check_rows/NotMapped from jacobian_jvp/jacobian_vjp. A row
absent from the primal's evaluations (Parameter/Stochastic/Constant/
Observed) is now filled with the identity on its own columns when it
is also an input column, and left zero otherwise — needed so a
Fisher root's arguments (which include leaves like x and mu) can be
Jacobian rows.
EOF
)"
```

### Task 14: Fisher assembly — `information.rs` (R23)

**Files:**
- Create: `crates/skoro-model/src/information.rs`
- Modify: `crates/skoro-model/src/lib.rs` (`pub mod information;`)

**Interfaces:**
- Consumes: `ModelClientOps::{factors, bound, tangent_layout, evaluate_ids}` (existing + Task 11), `jacobian::jacobian_vjp` (Task 13), `roles::FISHER_ROOT`.
- Produces: `information::{InformationBlock, information_blocks, information_root, information}`, consumed by Phase 5's `skoro-normal`/`skoro` tests. No acceptance test targets this module directly until Phase 5 (§5.7/§5.9) — those are the only tests that can exercise it end-to-end, since they need a real `FISHER_ROOT` binding, which `Normal::factor`/`JointNormal::factor` (Phase 4a) provide. Implement it now per the additive-phase plan (§6 lists R23 under Phase 3), but its first real test coverage is Phase 5.

- [ ] **Step 1: Write `crates/skoro-model/src/information.rs`**

```rust
//! Assembling the Fisher information (or its root) from every term's `FISHER_ROOT` binding.

use skoro_backend::{Backend, Tensor};
use skoro_core::{
    error::{SkoroError, SkoroResult},
    id::Id,
};
use skoro_graph::tangent_layout::TangentLayout;
use skoro_operation::ValueMap;
use skoro_riemann::matrix::EuclideanMatrixD;

use crate::client_ops::ModelClientOps;
use crate::jacobian::jacobian_vjp;
use crate::roles::FISHER_ROOT;

/// One term's contribution to the stacked Fisher root, `[batch, k, free.total()]`.
pub struct InformationBlock<B: Backend> {
    pub factor: Id,
    pub root: Tensor<B, 3>,
}

/// Batch size and device, chosen the same way `jacobian.rs` does.
fn batch_and_device<B: Backend>(points: &ValueMap<B>) -> (usize, B::Device) {
    match points.values().next() {
        Some(point) => (point.dims()[0], point.device()),
        None => (1, B::Device::default()),
    }
}

/// One [`InformationBlock`] per term that carries a `FISHER_ROOT` binding, skipping every term
/// that does not.
pub fn information_blocks<B: Backend, C: ModelClientOps<B>>(
    client: &C,
    points: &ValueMap<B>,
    free: &TangentLayout,
) -> SkoroResult<Vec<InformationBlock<B>>> {
    let (batch, device) = batch_and_device(points);
    let mut blocks = Vec::new();
    for f in client.factors() {
        let roots = client.bound(f.density, FISHER_ROOT);
        let root_id = match roots.len() {
            0 => continue,
            1 => roots[0],
            n => {
                return Err(SkoroError::ShapeMismatch {
                    expected: 1,
                    found: n,
                });
            }
        };
        if f.arguments.is_empty() {
            return Err(SkoroError::ShapeMismatch {
                expected: 1,
                found: 0,
            });
        }
        let args = client.tangent_layout(&f.arguments)?;
        let j = jacobian_vjp(client, points, free, &args)?;

        let out = client.evaluate_ids(points, &[root_id])?;
        let (value, qty) = match out.evaluations.get(&root_id) {
            Some(pair) => pair,
            None => out
                .constants
                .get(&root_id)
                .ok_or(SkoroError::NodeNotFound { id: root_id })?,
        };
        let matrix_type = qty
            .as_any()
            .downcast_ref::<EuclideanMatrixD>()
            .ok_or(SkoroError::QuantityTypeMismatch { id: root_id })?;
        let (k, cols) = (matrix_type.0, matrix_type.1);
        if cols != args.total() {
            return Err(SkoroError::LayoutDimensionMismatch {
                expected: args.total(),
                found: cols,
            });
        }
        let root_batch = value.unwrap_float_ref().dims()[0];
        let mut root = value.unwrap_float_ref().clone().reshape([root_batch, k, cols]);
        if root_batch == 1 && batch > 1 {
            root = root.repeat_dim(0, batch);
        }

        let block = root.matmul(j);
        blocks.push(InformationBlock {
            factor: f.density,
            root: block,
        });
    }
    Ok(blocks)
}

/// The blocks concatenated along dim 1: `[batch, Σk, free.total()]`. `ShapeMismatch { expected:
/// 1, found: 0 }` if there are no blocks.
pub fn information_root<B: Backend, C: ModelClientOps<B>>(
    client: &C,
    points: &ValueMap<B>,
    free: &TangentLayout,
) -> SkoroResult<Tensor<B, 3>> {
    let blocks = information_blocks(client, points, free)?;
    if blocks.is_empty() {
        return Err(SkoroError::ShapeMismatch {
            expected: 1,
            found: 0,
        });
    }
    Ok(Tensor::cat(
        blocks.into_iter().map(|b| b.root).collect(),
        1,
    ))
}

/// `Aᵀ A` for the stacked root `A`: `[batch, free.total(), free.total()]`.
pub fn information<B: Backend, C: ModelClientOps<B>>(
    client: &C,
    points: &ValueMap<B>,
    free: &TangentLayout,
) -> SkoroResult<Tensor<B, 3>> {
    let a = information_root(client, points, free)?;
    Ok(a.clone().swap_dims(1, 2).matmul(a))
}
```

`Input::unwrap_float_ref` (`crates/skoro-core/src/any_tensor.rs:50`) returns `&Tensor<B, 2>`, and `DynQuantityType::as_any` (`crates/skoro-graph/src/quantity.rs:224`) returns the wrapped quantity type itself, so the `EuclideanMatrixD` downcast succeeds.

- [ ] **Step 2: Declare the module**

In `crates/skoro-model/src/lib.rs`, add `pub mod information;`.

- [ ] **Step 3: Verify it builds**

Run: `cargo build -p skoro-model`
Expected: PASS. There is no runnable test for this module yet (see the task header) — Phase 5's `crates/skoro-normal/tests/information.rs` and `crates/skoro/tests/stacked_tangent.rs`'s `joint_information_is_the_precision` are what exercise it, once Phase 4a gives `Normal`/`JointNormal` a `FISHER_ROOT` binding to read.

- [ ] **Step 4: Commit**

```bash
git add crates/skoro-model/src/information.rs crates/skoro-model/src/lib.rs
git commit -m "$(cat <<'EOF'
Add information/information_root/information_blocks (R23)

Reads every term's FISHER_ROOT binding, evaluates it, and multiplies
by that term's VJP Jacobian over its own arguments to produce
[batch, k, free.total()] blocks; information_root stacks them,
information returns their Gram matrix. Untested until Phase 4a gives
a distribution a FISHER_ROOT to bind.
EOF
)"
```

**Phase 3 checkpoint:** `cargo test --workspace` passes with the record machinery still present.

- [ ] Run: `cargo test --workspace`
- [ ] Expected: PASS.

---

## Phase 4a — the breaking step

Implements R14's trait change, R16's new `add_quantity`, R24–R26, spec §5.6 (`crates/skoro-model/tests/teardown.rs`), spec §5.4 (`crates/skoro-model/tests/terms.rs`), and adapts every `impl Distribution` in the workspace. `facade::add_gaussian_quantity` may temporarily still call `record_factor` internally (its body is rewritten fully in Phase 4b). Checkpoint: `cargo test --workspace` (§6, Phase 4a).

Task 15 (R24, pure math + a `Tril`/`FixedLinearMap` composition) has no dependency on the `Distribution` trait and is genuinely additive — do it first, on its own commit. Task 16 is the one place in this plan that **must** land as a single commit: `Distribution::factor` replacing `Distribution::log_density` breaks every `impl Distribution` in the workspace simultaneously (`grep -rn "impl.*Distribution<" crates` — confirmed sites: `crates/skoro-normal/src/distribution.rs:168` (`Normal`), `crates/skoro-model/tests/discrete_ops.rs:262` (`PointMass`), `crates/skoro-model/tests/teardown.rs:170,340` (`LazyDistr`, `DetachedDistr`), `crates/skoro-model/tests/plumbing.rs:71` (`DummyDistribution`)); do not attempt to split it across commits or the workspace will not build in between.

### Task 15: `gaussian_root_matrix` and `information_root` (R24)

**Files:**
- Create: `crates/skoro-normal/src/information.rs`
- Modify: `crates/skoro-normal/src/lib.rs` (`pub mod information;`)
- Test: `crates/skoro-normal/tests/information.rs` (new file — start it now with just the `gaussian_root_matrix` unit check; the rest of this file is Phase 5, Task 24)

**Interfaces:**
- Consumes: `skoro_linalg::matrix::{FixedLinearMap, linear_map}` (Phase 2), `skoro_linalg::tril::Tril`.
- Produces: `skoro_normal::information::{gaussian_root_matrix, information_root}`, consumed by `Normal::factor` (Task 17).

- [ ] **Step 1: Write a failing hand-computed check for `gaussian_root_matrix(2)`**

Create `crates/skoro-normal/tests/information.rs`:

```rust
use skoro_normal::information::gaussian_root_matrix;

/// `d = 2`: `c_rows = d + d² = 6`, `c_cols = 2d + d(d+1)/2 = 7`, so `rows = 42`, `cols = 3` —
/// this shape alone catches an off-by-one in `c_rows`/`c_cols` before any graph is involved.
///
/// For `W = [[1, 0], [0.5, 1]]` (packed `[w00, w10, w11] = [1, 0.5, 1]`), hand-derive a few
/// representative entries of `C` (via `vec(C) = K · vec(W)`) rather than the whole 42-entry
/// matrix — enough to catch a sign or transpose error in each of the two blocks §4.1 describes.
#[test]
fn gaussian_root_matrix_matches_the_hand_derivation_at_d_2() {
    let (rows, cols, k) = gaussian_root_matrix(2);
    assert_eq!((rows, cols), (42, 3));

    let w = [1.0, 0.5, 1.0]; // [w00, w10, w11], packed row-major lower-triangular
    let vec_c: Vec<f64> = (0..rows)
        .map(|r| (0..cols).map(|c| k[r * cols + c] * w[c]).sum())
        .collect();
    let c_cols = 7;
    let at = |row: usize, col: usize| vec_c[row * c_cols + col];

    // (x, mu) block: C = [W, -W, 0], rows 0..2.
    approx::assert_abs_diff_eq!(at(0, 0), 1.0, epsilon = 1e-12); // W[0][0]
    approx::assert_abs_diff_eq!(at(0, 2), -1.0, epsilon = 1e-12); // -W[0][0]
    approx::assert_abs_diff_eq!(at(1, 0), 0.5, epsilon = 1e-12); // W[1][0]
    approx::assert_abs_diff_eq!(at(1, 1), 1.0, epsilon = 1e-12); // W[1][1]
    approx::assert_abs_diff_eq!(at(1, 3), -1.0, epsilon = 1e-12); // -W[1][1]
    approx::assert_abs_diff_eq!(at(0, 4), 0.0, epsilon = 1e-12); // zero block over L

    // L block: rows 2..6 index (r, c) row-major; cols 4..7 index q = packed(a, b).
    let s = std::f64::consts::FRAC_1_SQRT_2;
    // row (r=0, c=0) -> C row 2; S_0[0][0] = sqrt(2) * w00.
    approx::assert_abs_diff_eq!(at(2, 4), 2.0f64.sqrt() * 1.0, epsilon = 1e-12);
    // row (r=1, c=1) -> C row 5; S_2[1][1] = sqrt(2) * w11.
    approx::assert_abs_diff_eq!(at(5, 6), 2.0f64.sqrt() * 1.0, epsilon = 1e-12);
    // row (r=0, c=1) -> C row 3; S_0[0][1] = w10 / sqrt(2), S_1[0][1] = w11 / sqrt(2).
    approx::assert_abs_diff_eq!(at(3, 4), s * 0.5, epsilon = 1e-12);
    approx::assert_abs_diff_eq!(at(3, 5), s * 1.0, epsilon = 1e-12);
    // Symmetry the construction claims: row (r=1, c=0) -> C row 4 should equal row (0,1)'s C_L
    // entries, since S_q is symmetric by construction.
    assert_eq!(at(4, 4), at(3, 4));
    assert_eq!(at(4, 5), at(3, 5));
}
```

- [ ] **Step 2: Run the test to see it fail**

Run: `cargo test -p skoro-normal --test information`
Expected: FAIL to compile — `skoro_normal::information` does not exist.

- [ ] **Step 3: Write `crates/skoro-normal/src/information.rs`**

```rust
//! The Gaussian Fisher root over `[x, μ, L]` — see
//! `docs/superpowers/specs/2026-09-01-factor-bindings-and-fisher-root-spec.md` §4.1 for the
//! derivation this implements exactly.

use skoro_backend::Backend;
use skoro_core::error::SkoroResult;
use skoro_linalg::matrix::{FixedLinearMap, linear_map};
use skoro_linalg::tril::Tril;
use skoro_model::q::Q;
use skoro_riemann::matrix::EuclideanMatrixD;

/// `K` with `vec(C) = K · vec(W)`, `W = L⁻¹` packed as a `Tril`, `C` the Gaussian Fisher root
/// over `[x, μ, L]`. Returns `(rows, cols, row-major data)` with `rows = (d + d²)(2d + d(d+1)/2)`,
/// `cols = d(d+1)/2`.
pub fn gaussian_root_matrix(d: usize) -> (usize, usize, Vec<f64>) {
    let d_l = d * (d + 1) / 2;
    let c_cols = 2 * d + d_l;
    let c_rows = d + d * d;
    let rows = c_rows * c_cols;
    let cols = d_l;
    let mut k = vec![0.0_f64; rows * cols];
    let packed = |i: usize, j: usize| i * (i + 1) / 2 + j;
    let cidx = |r: usize, c: usize| r * c_cols + c;

    for i in 0..d {
        for j in 0..=i {
            k[cidx(i, j) * cols + packed(i, j)] += 1.0;
            k[cidx(i, d + j) * cols + packed(i, j)] += -1.0;
        }
    }

    let s = std::f64::consts::FRAC_1_SQRT_2;
    for a in 0..d {
        for b in 0..=a {
            let q = packed(a, b);
            for r in 0..d {
                for c in 0..d {
                    let row = cidx(d + r * d + c, 2 * d + q);
                    if c == b && a <= r {
                        k[row * cols + packed(r, a)] += s;
                    }
                    if r == b && a <= c {
                        k[row * cols + packed(c, a)] += s;
                    }
                }
            }
        }
    }

    (rows, cols, k)
}

/// The Gaussian Fisher root as a graph node: `EuclideanMatrixD(d + d², 2d + d(d+1)/2)`, built
/// from `w = L⁻¹` (packed `Tril`) through the fixed linear map [`gaussian_root_matrix`]
/// describes.
pub fn information_root<B: Backend>(w: &Q<B, Tril>, d: usize) -> SkoroResult<Q<B, EuclideanMatrixD>> {
    let (rows, cols, data) = gaussian_root_matrix(d);
    let map = FixedLinearMap::from_row_major(rows, cols, &data, &Default::default());
    let d_l = d * (d + 1) / 2;
    linear_map(w, EuclideanMatrixD(d + d * d, 2 * d + d_l), map)
}
```

- [ ] **Step 4: Declare the module**

In `crates/skoro-normal/src/lib.rs`, add `pub mod information;`.

- [ ] **Step 5: Run the test**

Run: `cargo test -p skoro-normal --test information`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/skoro-normal/src/information.rs crates/skoro-normal/src/lib.rs \
  crates/skoro-normal/tests/information.rs
git commit -m "$(cat <<'EOF'
Add gaussian_root_matrix and information_root (R24)

The fixed linear map K with vec(C) = K vec(W) that turns a packed
Tril whitening into the Gaussian Fisher root over [x, mu, L], per
spec section 4.1. Pure math plus a FixedLinearMap composition — no
dependency on the Distribution trait, so this lands before the
breaking change in Task 16.
EOF
)"
```

### Task 16: the atomic breaking commit — `Distribution::factor`, `add_quantity`, `Normal::factor`, `JointNormal::factor` (R14, R16, R25, R26, §5.6)

This is one commit. Every step below is staged together; do not run `cargo test` expecting green until all of them are done, because the crate will not compile in between (the trait signature and every implementor change together).

**Files:**
- Modify: `crates/skoro-model/src/distribution.rs` (`Distribution::log_density` → `Distribution::factor`)
- Modify: `crates/skoro-model/src/model.rs` (`ModelClient::add_quantity`)
- Modify: `crates/skoro-model/tests/discrete_ops.rs` (`PointMass`, line 262)
- Modify: `crates/skoro-model/tests/plumbing.rs` (`DummyDistribution`, line 71)
- Modify: `crates/skoro-model/tests/teardown.rs` (`LazyDistr` line 170, `DetachedDistr` line 340 — spec §5.6)
- Modify: `crates/skoro-normal/src/distribution.rs` (`Normal::factor`, R25)
- Modify: `crates/skoro-normal/src/joint.rs` (`JointNormal::factor`, `add_joint_factor`, R26)
- Modify: any test that calls `add_joint_factor` and asserts on the old `FactorRecord`/`registry` reads it just stopped populating — see Step 8.

**Interfaces:**
- Produces: `Distribution::factor(&self, x: &Q<B, Self::Qty>) -> SkoroResult<Factor<B>>` (replacing `log_density`), the new `ModelClient::add_quantity` (binds `ARGUMENT` + every declared role instead of calling `mark_density` alone).

- [ ] **Step 1: Flip the `Distribution` trait**

In `crates/skoro-model/src/distribution.rs`, replace:

```rust
    /// Build the log-density node scoring `x` under this distribution.
    ///
    /// Fallible because a distribution's parameters and its quantity can disagree
    /// dimensionally — a covariance factor sized for a different manifold, say — which
    /// is ordinary user error rather than an internal fault.
    fn log_density(&self, x: &Q<B, Self::Qty>) -> SkoroResult<Q<B, Real>>;
```

with:

```rust
    /// Build the factor — log-density term plus its declared bindings — scoring `x` under
    /// this distribution.
    ///
    /// Fallible because a distribution's parameters and its quantity can disagree
    /// dimensionally — a covariance factor sized for a different manifold, say — which
    /// is ordinary user error rather than an internal fault.
    fn factor(&self, x: &Q<B, Self::Qty>) -> SkoroResult<Factor<B>>;
```

- [ ] **Step 2: Rewrite `ModelClient::add_quantity` (R16)**

In `crates/skoro-model/src/model.rs`, replace the `add_quantity` body (~line 832-851):

```rust
    pub(crate) fn add_quantity<K, D>(&self, distr: D) -> SkoroResult<Q<B, D::Qty>>
    where
        K: Kind<B>,
        D: Distribution<B, K>,
    {
        // add stochastic quantity
        let mut server = self.server.write().unwrap();
        let id = server.add_stochastic(&distr)?;
        let x = Q::new(id, self.clone(), NodeDropGuard::new(id, &server.handles));

        // maybe slow, but currently the price for ergonomic interface: we need to drop the server
        // to add the log-density term (or else deadlock will occur).
        drop(server);
        let density = distr.log_density(&x)?;

        // again, slow... but now we mark the mapped quantity as a density.
        let mut server = self.server.write().unwrap();
        server.mark_density(&density, &x)?;
        Ok(x)
    }
```

with:

```rust
    pub(crate) fn add_quantity<K, D>(&self, distr: D) -> SkoroResult<Q<B, D::Qty>>
    where
        K: Kind<B>,
        D: Distribution<B, K>,
    {
        // add stochastic quantity
        let mut server = self.server.write().unwrap();
        let id = server.add_stochastic(&distr)?;
        let x = Q::new(id, self.clone(), NodeDropGuard::new(id, &server.handles));

        // The lock must be dropped before calling distr.factor(&x), or the distribution's own
        // graph-building calls (which take the write lock themselves) deadlock.
        drop(server);
        let factor = distr.factor(&x)?;

        let mut server = self.server.write().unwrap();
        server.mark_density(&factor.log_density, &x)?;
        let density = factor.log_density.id;
        server.bind(density, ARGUMENT, x.id())?;
        for &parent in distr.parent_ids() {
            server.bind(density, ARGUMENT, parent)?;
        }
        for (role, handle) in &factor.bindings {
            server.bind(density, *role, handle.id)?;
        }
        Ok(x)
    }
```

Add `use crate::roles::ARGUMENT;` to `model.rs`'s imports (it may already be there from Task 9).

- [ ] **Step 3: Adapt `PointMass` (`crates/skoro-model/tests/discrete_ops.rs:262`)**

Change:

```rust
        fn log_density(&self, x: &Q<B, Boolean>) -> SkoroResult<Q<B, Real>> {
            x.to_real()? * 0.0f32
        }
```

to:

```rust
        fn factor(&self, x: &Q<B, Boolean>) -> SkoroResult<Factor<B>> {
            Ok(Factor::new((x.to_real()? * 0.0f32)?))
        }
```

Add `use skoro_model::distribution::Factor;` to the test file's imports (check whether `Distribution` is already imported the same way — match that style).

- [ ] **Step 4: Adapt `DummyDistribution` (`crates/skoro-model/tests/plumbing.rs:71`)**

Change:

```rust
    fn log_density(&self, x: &Q<B, Real>) -> SkoroResult<Q<B, Real>> {
        register_map1(Real, DummyOperation, x)
    }
```

to:

```rust
    fn factor(&self, x: &Q<B, Real>) -> SkoroResult<Factor<B>> {
        Ok(Factor::new(register_map1(Real, DummyOperation, x)?))
    }
```

Add the same `Factor` import.

- [ ] **Step 5: Adapt `LazyDistr` and `DetachedDistr` (`crates/skoro-model/tests/teardown.rs`, spec §5.6)**

`LazyDistr` (line ~170):

```rust
        fn log_density(&self, x: &Q<B, Real>) -> SkoroResult<Q<B, Real>> {
            register_map1(Real, LiteralScalarMultiplication::new(2.0), x)
        }
```

becomes:

```rust
        fn factor(&self, x: &Q<B, Real>) -> SkoroResult<Factor<B>> {
            Ok(Factor::new(register_map1(
                Real,
                LiteralScalarMultiplication::new(2.0),
                x,
            )?))
        }
```

`DetachedDistr` (line ~340):

```rust
        fn log_density(&self, _x: &Q<B, Real>) -> SkoroResult<Q<B, Real>> {
            register_map1(Real, LiteralScalarMultiplication::new(2.0), &self.elsewhere)
        }
```

becomes:

```rust
        fn factor(&self, _x: &Q<B, Real>) -> SkoroResult<Factor<B>> {
            Ok(Factor::new(register_map1(
                Real,
                LiteralScalarMultiplication::new(2.0),
                &self.elsewhere,
            )?))
        }
```

Add `use skoro_model::distribution::Factor;` to `teardown.rs`'s imports. Per spec §5.6, also remove the mention of `NlsStructure` at line 11 of that file (a doc-comment reference to a bound that no longer exists in this design) — read the surrounding paragraph and rephrase it to talk about `ModelBuilderOps` directly instead, since `NlsStructure` was never implemented in this codebase (grep confirms no hits outside that one comment).

- [ ] **Step 6: `Normal::factor` (R25)**

In `crates/skoro-normal/src/distribution.rs`, add `use skoro_graph::binding::Role;` and `use skoro_model::{distribution::Factor, roles::{RESIDUAL, LOCATION, WHITENING, NORMALIZER_PARENT, FISHER_ROOT}};` and `use crate::information::information_root;` (add `Continuous` to the existing `skoro_core::kind::Continuous` import if not already present — it is, since `Distribution<B, Continuous>` is already implemented here). Replace `log_density`:

```rust
    fn log_density(&self, x: &Q<B, M>) -> SkoroResult<Q<B, Real>> {
        let white_tangent = self.whitened_residual(x)?;
        let log_det = self.std_dev.invert()?.diag()?.abs()?.log()?.sum()?;
        let quadratic = (white_tangent.norm_squared()? * -0.5)?;
        (quadratic + self.constant)? + log_det
    }
```

with:

```rust
    fn factor(&self, x: &Q<B, M>) -> SkoroResult<Factor<B>> {
        let d = self.quantity_type.dimension();
        let n = self.std_dev.get_quantity_type::<Continuous>().0;
        if n != d {
            return Err(SkoroError::ShapeMismatch {
                expected: d,
                found: n,
            });
        }
        let w = self.std_dev.invert()?;
        let r = w.vector_product(&self.mean.inverse_retract(x)?)?;
        let log_det = w.diag()?.abs()?.log()?.sum()?;
        let density = ((r.norm_squared()? * -0.5)? + self.constant)? + log_det;
        let root = information_root(&w, d)?;
        Ok(Factor::new(density?)
            .bind(RESIDUAL, &r)
            .bind(LOCATION, &self.mean)
            .bind(WHITENING, &self.std_dev)
            .bind(NORMALIZER_PARENT, &self.std_dev)
            .bind(FISHER_ROOT, &root))
    }
```

Note `density` is computed as `SkoroResult<Q<B, Real>>` by the same `?`-chained expression the old `log_density` used (mirroring its exact arithmetic, just bound to a name before wrapping in `Factor::new`), and `w` is built exactly once here (the old code called `self.std_dev.invert()` twice — once in `whitened_residual`, once again for `log_det` — R25 fixes that redundancy by building `w` once and reusing it for both `r` and `log_det`, so `whitened_residual` is no longer called from inside `factor`; keep `whitened_residual` itself as a public method unchanged, since other code and Phase 4b's `structure.rs` still use it directly). Add `use skoro_core::error::SkoroError;` if not already imported (it is, via `SkoroResult`'s crate — check whether `SkoroError` itself needs a separate `use`).

- [ ] **Step 7: `JointNormal::factor` and `add_joint_factor` (R26)**

In `crates/skoro-normal/src/joint.rs`, add `use skoro_linalg::matrix::linear_map; use skoro_model::{distribution::Factor, roles::{RESIDUAL, NORMALIZER_PARENT, FISHER_ROOT}}; use skoro_riemann::matrix::EuclideanMatrixD;`. Add a new method to the `impl<B: Backend> JointNormal<B> where B::FloatElem: ...` block (same bound as `residual`/`log_density`), after `normalizer_parents`:

```rust
    /// The scatter matrix `P`, shape `[D·D, D(D+1)/2]`, with `P[i·D + j, i(i+1)/2 + j] = 1` for
    /// `j ≤ i`, zero elsewhere — unpacks a `Tril(D)` into its dense `D×D` row-major form.
    fn scatter_matrix(d: usize) -> (usize, usize, Vec<f64>) {
        let cols = d * (d + 1) / 2;
        let rows = d * d;
        let mut p = vec![0.0_f64; rows * cols];
        for i in 0..d {
            for j in 0..=i {
                let packed = i * (i + 1) / 2 + j;
                p[(i * d + j) * cols + packed] = 1.0;
            }
        }
        (rows, cols, p)
    }

    /// The factor: `RESIDUAL` is the stacked-tangent residual, `NORMALIZER_PARENT` is the
    /// whitening, and `FISHER_ROOT` is the whitening itself unpacked to a dense
    /// `EuclideanMatrixD(D, D)` — since `r = Wδ`, `CᵀC = WᵀW = Λ` exactly for `C = W` dense.
    pub fn factor<C: ModelBuilderOps<B>>(&self, client: &C) -> SkoroResult<Factor<B>> {
        let residual = self.residual(client)?;
        // Built from the same residual node as the residual binding, so the two cannot
        // disagree — and the stacked tangent is built once rather than twice.
        let density = ((residual.norm_squared()? * -0.5)? + self.normalizer)?;
        let d = client.tangent_layout(&self.ids)?.total();
        let (rows, cols, data) = Self::scatter_matrix(d);
        let map = skoro_linalg::matrix::FixedLinearMap::from_row_major(
            rows,
            cols,
            &data,
            &Default::default(),
        );
        let root = linear_map(&self.whitening, EuclideanMatrixD(d, d), map)?;
        Ok(Factor::new(density)
            .bind(RESIDUAL, &residual)
            .bind(NORMALIZER_PARENT, &self.whitening)
            .bind(FISHER_ROOT, &root))
    }
```

Replace `add_joint_factor`'s body:

```rust
pub fn add_joint_factor<B, C>(client: &C, joint: &JointNormal<B>) -> SkoroResult<Q<B, Real>>
where
    B: Backend,
    C: ModelBuilderOps<B>,
    B::FloatElem: std::ops::Div<B::FloatElem, Output = B::FloatElem>
        + std::ops::Mul<B::FloatElem, Output = B::FloatElem>
        + std::ops::Add<B::FloatElem, Output = B::FloatElem>
        + std::ops::Sub<B::FloatElem, Output = B::FloatElem>
        + std::ops::Neg<Output = B::FloatElem>
        + std::ops::AddAssign<B::FloatElem>,
{
    client.add_factor(joint.factor(client)?, joint.ids())
}
```

- [ ] **Step 8: Fix existing tests that observed `add_joint_factor`'s old registry side effect**

`add_joint_factor` no longer calls `record_factor` — any existing test asserting on `ModelBuilderOps::factors()`/`model.residual_layout()` after calling it will now see an empty registry. Run:

```bash
grep -rln "add_joint_factor" crates/skoro-normal/tests crates/skoro/tests
```

For each hit (expected: `crates/skoro-normal/tests/window.rs`, `crates/skoro/tests/stacked_tangent.rs`), read the test, and for any assertion reading `model.factors()`/`ModelBuilderOps::factors(&model)`/`model.residual_layout()` immediately after an `add_joint_factor` call, replace it with the new-API equivalent:

| Old (registry-based) | New (bindings-based) |
|---|---|
| `model.factors().len()` / `ModelBuilderOps::factors(&model).len()` | `ModelClientOps::factors(&model).len()` |
| `model.factors()[0].density` | `ModelClientOps::factors(&model)[0].density` |
| `model.factors()[0].residual` | `ModelClientOps::bound(&model, ModelClientOps::factors(&model)[0].density, RESIDUAL)[0]` |
| `model.residual_layout()` | `skoro_model::least_squares::residual_layout(&model)` |

`crates/skoro/tests/stacked_tangent.rs`'s `mod records`/`mod retirement` (read in full during Phase 4b's Task 22 regardless — do the minimal subset of that rewrite here that is needed to keep this commit's `cargo test --workspace` green; the rest of that file's cleanup is Task 22's job) and `crates/skoro-normal/tests/window.rs` are the two files most likely to need this. If a test only checks `graph_node_count` bounds (not factor contents), it needs no change — the binding-based path keeps nodes alive exactly as well as the registry did (Task 10 verified this).

- [ ] **Step 8b: Add `crates/skoro-model/tests/terms.rs` (spec §5.4)**

It needs `Distribution::factor`, which is why it lands in this commit rather than in Phase 3:

```rust
//! The `log p = log q + log w` split: a proposal term scores a live stochastic quantity; every
//! other term (an observed quantity's density, a free-standing factor) is a weight term.

use skoro_backend::{NdArrayBackend, Point, Tensor, TensorData};
use skoro_core::{error::SkoroResult, id::Id, kind::Continuous};
use skoro_graph::graph::TermKind;
use skoro_model::{
    ModelBuilderOps, ModelClientOps,
    distribution::{Distribution, Factor, Sampler},
    hash_map_builders::points,
    map::register_map1,
    model::ModelClient,
    q::Q,
    test_utils::into_vec,
};
use skoro_operation::{Input, ops::LiteralScalarMultiplication};
use skoro_riemann::real::Real;

type B = NdArrayBackend<f64, i32>;

fn row(values: &[f64]) -> Tensor<B, 2> {
    let n = values.len();
    Tensor::<B, 1>::from_data(TensorData::new(values.to_vec(), [n]), &Default::default())
        .reshape([1, n])
}

fn scalar(t: Tensor<B, 2>) -> f64 {
    into_vec(t)[0]
}

#[derive(Debug, Clone)]
struct NeverSampler;

impl Sampler<B, Continuous> for NeverSampler {
    fn sample(&self, _: usize, _: Vec<Input<B>>) -> Point<B> {
        unimplemented!("the tests never generate")
    }
}

/// `log p(x) = 2x`, over no parents.
struct Linear;

impl Distribution<B, Continuous> for Linear {
    type Qty = Real;
    type Sampler = NeverSampler;

    fn quantity_type(&self) -> Real {
        Real
    }
    fn sampler(&self) -> NeverSampler {
        NeverSampler
    }
    fn parent_ids(&self) -> &[Id] {
        &[]
    }
    fn factor(&self, x: &Q<B, Real>) -> SkoroResult<Factor<B>> {
        Ok(Factor::new(register_map1(
            Real,
            LiteralScalarMultiplication::new(2.0),
            x,
        )?))
    }
}

#[test]
fn proposal_plus_weight_is_the_log_density() {
    let model = ModelClient::<B>::default();
    let x = model.add_quantity(Linear).unwrap();
    let y = model.add_quantity(Linear).unwrap();
    y.observe(&1.5f64).unwrap();
    let a = model.add_parameter(Real);
    let term = register_map1(Real, LiteralScalarMultiplication::new(-3.0), &a).unwrap();
    model.add_factor(Factor::new(term), &[a.id()]).unwrap();

    let values = points()
        .with_quantity(&x, row(&[0.25]))
        .with_quantity(&a, row(&[2.0]))
        .build();
    let whole = scalar(model.log_density(&values).unwrap());
    let split = scalar(model.log_proposal(&values).unwrap())
        + scalar(model.log_weight(&values).unwrap());
    approx::assert_abs_diff_eq!(whole, split, epsilon = 1e-12);
}

#[test]
fn observing_moves_a_term_to_the_weight() {
    let model = ModelClient::<B>::default();
    let x = model.add_quantity(Linear).unwrap();
    let density = model.log_density_of(&x).unwrap();
    let kind_of = |model: &ModelClient<B>| {
        model
            .terms()
            .into_iter()
            .find(|t| t.density == density)
            .unwrap()
            .kind
    };

    assert_eq!(kind_of(&model), TermKind::Proposal);
    x.observe(&0.5f64).unwrap();
    assert_eq!(kind_of(&model), TermKind::Weight);
}
```

- [ ] **Step 9: Build, then run the whole workspace**

Run: `cargo build --workspace`
Expected: FAIL first (iterate through the compiler's error list — every `impl Distribution` site, every `Factor` import) until it passes.

Run: `cargo test --workspace`
Expected: PASS. This is the Phase 4a checkpoint (§6).

- [ ] **Step 10: Commit**

```bash
git add crates/skoro-model/src/distribution.rs crates/skoro-model/src/model.rs \
  crates/skoro-model/tests/discrete_ops.rs crates/skoro-model/tests/plumbing.rs \
  crates/skoro-model/tests/teardown.rs crates/skoro-normal/src/distribution.rs \
  crates/skoro-normal/src/joint.rs crates/skoro-normal/tests/window.rs \
  crates/skoro-model/tests/terms.rs crates/skoro/tests/stacked_tangent.rs
git commit -m "$(cat <<'EOF'
Distribution::factor replaces log_density (R14/R16/R25/R26)

The breaking step: every Distribution now returns a Factor (its
log-density plus declared bindings) instead of a bare log-density
node. add_quantity binds ARGUMENT to the scored quantity and every
parent, then every role the factor declared. Normal binds RESIDUAL,
LOCATION, WHITENING, NORMALIZER_PARENT and FISHER_ROOT; JointNormal
binds RESIDUAL, NORMALIZER_PARENT and FISHER_ROOT (its whitening
unpacked dense, since r = Wδ makes CᵀC = WᵀW = Λ directly).
add_joint_factor now calls add_factor instead of record_factor, so
every existing caller reading the old registry after it is updated
to the new bindings-based reads.
EOF
)"
```

**Phase 4a checkpoint (already run in Task 16, Step 9):** `cargo test --workspace` passes.

---

## Phase 4b — delete the record machinery

Implements R18, R20, R27, R28, the rest of §5.3 (renaming `tests/residual.rs` to `tests/bindings.rs` and finishing its table), §5.8, §5.9, and — because the checkpoint's own `grep` demands it even though the spec's §6 phase list omits it — R29 (the umbrella crate's prelude). Checkpoint: `cargo test --workspace` and `grep -rn "FactorRecord\|ResidualRegistry\|record_factor\|residual_layout()\|add_gaussian_quantity\|NodeIsRecorded" crates readme.md` returns nothing (§6, Phase 4b).

### Task 17: Delete `residual.rs`, the registry, and every trait method that read it (R18, R20)

**Files:**
- Delete: `crates/skoro-model/src/residual.rs`
- Modify: `crates/skoro-model/src/lib.rs` (drop `pub mod residual;`, `pub use residual::{FactorRecord, ResidualRegistry};`)
- Modify: `crates/skoro-model/src/model.rs` (drop the `registry` field and every method that reads/writes it: `record_factor`, `registry`, `residual_layout`, `prune_registry`, `debug_validate_state`; drop the `registry.names` check in `remove`; drop the `roots.extend(self.registry...)` block in `cleanup` if Task 10 re-added it)
- Modify: `crates/skoro-model/src/client_ops.rs` (drop `record_factor`, `factors` (the old `Vec<FactorRecord>` one — already superseded by `ModelClientOps::factors`), `residual_layout`, `mark_likelihood` from `ModelBuilderOps` and its impl)
- Modify: `crates/skoro-model/Cargo.toml` (drop the `crossbeam-queue` dependency if `cargo machete` or a `grep -rn crossbeam crates/skoro-model/src` shows it now unused)
- Modify: `crates/skoro-core/src/error.rs` — no change needed here (R1 already renamed the variant; nothing else references it)

**Interfaces:**
- Consumes: nothing new.
- Produces: `ModelBuilderOps` reduced to exactly `add_quantity, add_factor, bind, stacked_tangent, remove_closure` (R20).

- [ ] **Step 1: Delete the file and its declarations**

```bash
git rm crates/skoro-model/src/residual.rs
```

In `crates/skoro-model/src/lib.rs`, remove `pub mod residual;` and `pub use residual::{FactorRecord, ResidualRegistry};`.

- [ ] **Step 2: Strip `ModelServer`**

In `crates/skoro-model/src/model.rs`:
- Remove the `registry: ResidualRegistry` field from the `ModelServer` struct and its doc comment.
- Remove `registry: ResidualRegistry::new(),` from `ModelServer::new`.
- Remove `pub fn record_factor`, `pub fn registry`, `pub fn residual_layout`, `fn prune_registry`, `fn debug_validate_state` in their entirety.
- Remove every `self.prune_registry();` call site (in `demote_to_parameter`, `remove`, `remove_closure`).
- In `remove`, delete the now-dead check:
  ```rust
          if let Some(id) = ids.iter().find(|id| self.registry.names(**id)) {
              return Err(SkoroError::NodeStillBound { id: *id, by: *id });
          }
  ```
  (the shim from Task 1, Step 3 — `Graph::remove`'s own binding check, added in Phase 1 Task 3, now does this job for real).
- In `cleanup`, remove the `roots.extend(self.registry.factors()...)` block (Task 10 kept it; this is where it goes).
- Remove `use crate::residual::{FactorRecord, ResidualRegistry};` from the imports.

- [ ] **Step 3: Strip `client_ops.rs`**

In `crates/skoro-model/src/client_ops.rs`:
- Remove `record_factor`, `factors` (the `Vec<FactorRecord>` one), `residual_layout`, `mark_likelihood` from the `ModelBuilderOps` trait definition and from `impl<B: Backend> ModelBuilderOps<B> for ModelClient<B>`.
- Remove `use crate::residual::FactorRecord;`.
- Confirm the resulting trait reads exactly:
  ```rust
  pub trait ModelBuilderOps<B: Backend>: ModelClientOps<B> {
      fn add_quantity<K, D>(&self, distr: D) -> SkoroResult<Q<B, D::Qty>>
      where
          K: Kind<B>,
          D: Distribution<B, K>;
      fn add_factor(&self, factor: Factor<B>, arguments: &[Id]) -> SkoroResult<Q<B, Real>>;
      fn bind(&self, from: Id, role: Role, to: Id) -> SkoroResult<()>;
      fn stacked_tangent(&self, at: &PointMap<B>, ids: &[Id]) -> SkoroResult<Q<B, EuclideanD>>;
      fn remove_closure(&self, roots: &[Id]) -> SkoroResult<Vec<Node<B>>>;
  }
  ```
  (R20). Delete `ModelClient::mark_likelihood`'s trait-facing wrapper if `client_ops.rs` had one that only existed to serve the trait method just removed — but keep the `pub(crate)` inherent `ModelClient::mark_likelihood`/`ModelServer::mark_likelihood` in `model.rs`, since `ModelServer::add_factor` (Task 9) still calls it internally.

- [ ] **Step 4: Revert the `.factors()` qualification from Task 11**

`grep -rn "ModelBuilderOps::factors" crates` and change every hit back to plain `.factors()` — with the old trait method gone, `ModelClientOps::factors` is the only candidate again, so the qualification is no longer needed (and would now fail to compile, since `ModelBuilderOps::factors` no longer exists).

- [ ] **Step 5: Drop the now-unused dependency, if any**

```bash
grep -rn "crossbeam" crates/skoro-model/src
```

If nothing left references `crossbeam_queue`, remove the `crossbeam-queue` line from `crates/skoro-model/Cargo.toml`'s `[dependencies]`.

- [ ] **Step 6: Build — expect a long error list, work through it**

Run: `cargo build --workspace`
Expected: FAIL, repeatedly, at every remaining use of `FactorRecord`/`ResidualRegistry`/`record_factor`/`residual_layout()`/`registry()`. These are exactly the sites Tasks 19-22 fix (facade.rs, structure.rs, and the test files under §5.8/§5.9) — for *this* task, fix only the sites inside `skoro-model` itself (its own `tests/` directory: `mod records`/`mod rejections` in the now-doomed `residual.rs` — which Task 22 renames/rewrites — leave those failing for now if they are the only remaining error source, and proceed to Task 18 immediately rather than half-fixing `residual.rs` twice). If `cargo build -p skoro-model` (library only, not tests) passes, that is enough to move to Task 18; the full-workspace build does not need to pass again until Task 22 is done.

- [ ] **Step 7: Commit**

```bash
git add -A crates/skoro-model
git commit -m "$(cat <<'EOF'
Delete the FactorRecord/ResidualRegistry side table (R18/R20)

Bindings (Phase 1-3) fully supersede it: cleanup, remove and
remove_closure all reason about references through Graph's bindings
now, not a parallel registry. ModelBuilderOps is reduced to
add_quantity, add_factor, bind, stacked_tangent, remove_closure.
Downstream crates (skoro-normal, skoro, and skoro-model's own tests)
are updated in the following commits.
EOF
)"
```

### Task 18: Delete `facade::add_gaussian_quantity`/`gaussian_blocks`, keep the `constant_covariance_*` helpers (R27)

**Files:**
- Modify: `crates/skoro-normal/src/facade.rs`

**Interfaces:**
- Produces: `facade.rs` reduced to the three `constant_covariance_*` helpers.

- [ ] **Step 1: Delete the two functions**

In `crates/skoro-normal/src/facade.rs`, delete `add_gaussian_quantity` and `gaussian_blocks` in their entirety (both call the now-gone `record_factor`/`ModelBuilderOps::factors`). Delete the now-unused imports this leaves behind (`skoro_model::{FactorRecord, ModelBuilderOps, ...}` — keep `ModelClientOps` since the `constant_covariance_*` helpers use it; drop `FactorRecord`, `ModelBuilderOps`, `crate::distribution::Normal`, `crate::structure::GaussianBlock` if nothing else in the file needs them).

Update the module doc comment (lines 1-14) to drop the description of `add_gaussian_quantity`/recording, replacing it with a short note that a Gaussian quantity now enters a model through `client.add_quantity(Normal::new(mean, std_dev))` directly (R27's own wording).

- [ ] **Step 2: Build**

Run: `cargo build -p skoro-normal`
Expected: FAIL at every remaining call site of `add_gaussian_quantity`/`gaussian_blocks` — these are Task 21's job (the §5.8 test files). Confirm the *library* target (`cargo build -p skoro-normal --lib`) passes before moving on.

- [ ] **Step 3: Commit**

```bash
git add crates/skoro-normal/src/facade.rs
git commit -m "$(cat <<'EOF'
Delete facade::add_gaussian_quantity and gaussian_blocks (R27)

A Gaussian quantity now enters a model through
client.add_quantity(Normal::new(mean, std_dev)) directly, which
builds its Factor (bindings and all) through the Distribution trait.
The constant_covariance_* helpers are untouched.
EOF
)"
```

### Task 19: `GaussianBlock::{of, all}` on the new API (R28)

**Files:**
- Modify: `crates/skoro-normal/src/structure.rs`

**Interfaces:**
- Consumes: `ModelClientOps::{bound, factors}` (Task 11), `FactorInfo` (Task 11), `roles::{RESIDUAL, LOCATION, WHITENING}`.
- Produces: `GaussianBlock::{of, all}` on the new signatures, consumed by Task 21's `crates/skoro-normal/tests/structure.rs` rewrite.

- [ ] **Step 1: Rewrite `structure.rs`**

Replace the whole file's body (keep the `GaussianBlock` struct's fields unchanged — `density`, `residual`, `mean`, `whitening`):

```rust
//! The Gaussian view of a factor's bindings.
//!
//! A [`Normal`](crate::distribution::Normal) factor's negative log-density is `½‖r‖²` plus a
//! normalizer, with `r(x) = L⁻¹ (x ⊖ μ)`. What distinguishes it from a merely
//! least-squares-shaped factor is that the mean and the whitening are reachable
//! **separately** — which is what a filter needs, since `H = ∂μ/∂x` and `R = L Lᵀ` are
//! different objects. [`GaussianBlock`] is the view of a term whose bindings carry all three
//! of `RESIDUAL`, `LOCATION` and `WHITENING`.

use skoro_backend::Backend;
use skoro_core::id::Id;

use crate::client_ops::{FactorInfo, ModelClientOps};
use crate::roles::{LOCATION, RESIDUAL, WHITENING};

/// One factor of a hierarchical Gaussian model, read off its bindings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GaussianBlock {
    pub density: Id,
    pub residual: Id,
    pub mean: Id,
    pub whitening: Id,
}

impl GaussianBlock {
    /// The Gaussian view of `factor`, or `None` unless it carries exactly one `RESIDUAL`, one
    /// `LOCATION` and one `WHITENING` binding.
    pub fn of<B: Backend, C: ModelClientOps<B>>(client: &C, factor: &FactorInfo) -> Option<Self> {
        let residual = single(client.bound(factor.density, RESIDUAL))?;
        let mean = single(client.bound(factor.density, LOCATION))?;
        let whitening = single(client.bound(factor.density, WHITENING))?;
        Some(Self {
            density: factor.density,
            residual,
            mean,
            whitening,
        })
    }

    /// The Gaussian blocks among the model's terms, in term order.
    pub fn all<B: Backend, C: ModelClientOps<B>>(client: &C) -> Vec<Self> {
        client
            .factors()
            .iter()
            .filter_map(|factor| Self::of(client, factor))
            .collect()
    }
}

fn single(ids: Vec<Id>) -> Option<Id> {
    match ids.as_slice() {
        [id] => Some(*id),
        _ => None,
    }
}
```

Note this module lives in `skoro_model`'s own re-export path in the spec's wording ("Add `crates/skoro-model/src/structure.rs`"? — re-check: R28 is listed under §3.7 "skoro-normal", so this file stays at `crates/skoro-normal/src/structure.rs` as before; only its *implementation* changes). Confirm `crate::client_ops::FactorInfo` resolves — `FactorInfo` is defined in `skoro_model::client_ops` (Task 11); import it as `use skoro_model::client_ops::FactorInfo;` and `use skoro_model::ModelClientOps;` (this file is in `skoro-normal`, a different crate from `skoro-model`) — the `use crate::...` lines drafted above are wrong for a cross-crate import; fix to:

```rust
use skoro_model::{ModelClientOps, client_ops::FactorInfo, roles::{LOCATION, RESIDUAL, WHITENING}};
```

- [ ] **Step 2: Build**

Run: `cargo build -p skoro-normal --lib`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add crates/skoro-normal/src/structure.rs
git commit -m "$(cat <<'EOF'
GaussianBlock::{of,all} read RESIDUAL/LOCATION/WHITENING bindings (R28)

Replaces the FactorRecord-based view with one built from
ModelClientOps::bound against a FactorInfo — of() requires exactly
one of each of the three roles, all() filters the model's terms
through it.
EOF
)"
```

### Task 20: Umbrella crate prelude (R29 — folded in here because Phase 4b's checkpoint grep requires it)

**Files:**
- Modify: `crates/skoro/src/lib.rs`

**Interfaces:**
- Produces: the prelude every downstream test (including Task 22's) imports from.

- [ ] **Step 1: Update the prelude**

In `crates/skoro/src/lib.rs`'s `pub mod prelude`, remove:

```rust
    // ----- what a builder records about its least-squares factors -----
    pub use skoro_model::model::{ModelClient, ModelServer};
    pub use skoro_model::q::Q;
    pub use skoro_model::{FactorRecord, ResidualRegistry};
    pub use skoro_normal::structure::GaussianBlock;
```

replacing the `FactorRecord`/`ResidualRegistry` line, and add the new re-exports. The block becomes:

```rust
    // ----- the model interface -----
    pub use skoro_model::{ModelBuilderOps, ModelClientOps};
    pub use skoro_model::model::{ModelClient, ModelServer};
    pub use skoro_model::q::{Handle, Q};

    // ----- what a builder declares about its factors -----
    pub use skoro_graph::graph::{Term, TermKind};
    pub use skoro_model::client_ops::FactorInfo;
    pub use skoro_model::distribution::Factor;
    pub use skoro_model::roles::{self, Role};
    pub use skoro_normal::structure::GaussianBlock;

    // ----- Fisher information and least-squares reads -----
    pub use skoro_model::information::{
        InformationBlock, information, information_blocks, information_root,
    };
    pub use skoro_model::least_squares::residual_layout;
```

This replaces both the existing `// ----- the model interface -----` line and the `// ----- what a builder records about its least-squares factors -----` block (`crates/skoro/src/lib.rs:76-82`). Add `FixedLinearMap, linear_map` to the existing `pub use skoro_linalg::matrix::{MatmulOps, TransposeOps};` line. Update the module-level doc comment at the top of `lib.rs` if it mentions `add_gaussian_quantity`, to describe `client.add_quantity(Normal::new(mean, std_dev))` instead, and drop `add_gaussian_quantity`/`gaussian_blocks` from the facade re-export:

```rust
    // ----- Gaussian factor builders -----
    pub use skoro_normal::facade::{
        constant_covariance_diagonal, constant_covariance_identity, constant_covariance_isotropic,
    };
    pub use skoro_normal::joint::add_joint_factor;
```

- [ ] **Step 2: Build**

Run: `cargo build -p skoro --lib`
Expected: PASS. The doctest in `lib.rs`'s top-level module doc (the `model.add_parameter`/`add_constant` example) should still compile unchanged, since it does not touch anything removed.

Run: `cargo test -p skoro --doc`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add crates/skoro/src/lib.rs
git commit -m "$(cat <<'EOF'
Update the umbrella prelude for bindings (R29)

Removes FactorRecord/ResidualRegistry/add_gaussian_quantity; adds
Role, roles, Factor, FactorInfo, Handle, Term, TermKind, the
information module, least_squares::residual_layout, and
FixedLinearMap/linear_map.
EOF
)"
```

### Task 21: `residual.rs` → `bindings.rs` — finish spec §5.3's table

**Files:**
- Rename: `crates/skoro-model/tests/residual.rs` → `crates/skoro-model/tests/bindings.rs`
- Modify: `crates/skoro-model/tests/compile_fail/remove_closure_is_not_a_client_op.rs` (keep as-is per spec — only the `NodeStillBound` wording from Task 1 changed)

**Interfaces:**
- Consumes: everything from Phase 1-4a: `Bindings`, `Term`, `ModelClientOps::{factors, bound, terms}`, `ModelBuilderOps::{add_factor, bind}`, `Factor`, `roles::*`.

This is the file spec §5.3 fully describes; Task 9-12 already added `mod bindings_via_add_factor` with 5 of its 9 rows (`add_factor_marks_and_binds`, `add_factor_needs_an_argument`, `a_bound_node_survives_cleanup`, `factors_are_in_term_order`, `residual_layout_follows_term_order_and_dimension`). This task: (a) renames the file, (b) deletes the now-dead `mod records`/`mod rejections` (the `FactorRecord`-based tests — their subject matter no longer exists), (c) adds the remaining 4 rows using a **local test distribution** per spec §5.3's header, and (d) merges `mod bindings_via_add_factor`'s tests into the file's top level (or a `mod bindings` — match whichever the file's remaining `mod compile_fail` expects as a sibling).

- [ ] **Step 1: Rename and prune**

```bash
git mv crates/skoro-model/tests/residual.rs crates/skoro-model/tests/bindings.rs
```

Delete `mod records` and `mod rejections` from `bindings.rs` in their entirety (their fixture `a_factor` and every test in them referenced `FactorRecord`, which no longer exists). Keep `mod compile_fail` (still tests that `remove_closure` is unreachable from `ModelClientOps` — unrelated to the registry). Keep the file's module doc comment but rewrite it to describe bindings instead of records, along the lines of: "A binding is a labelled reference edge... recorded once when a distribution is added, or explicitly via `bind`/`add_factor`."

- [ ] **Step 2: Add the local test distribution the remaining 4 tests need**

Each integration-test binary is its own crate, so `bindings.rs` defines its own sampler and distribution (mirroring `LazySampler` in `teardown.rs:50-56`). Adjust the file's imports (drop `FactorRecord`; add `Id`, `SkoroResult`, `Point`, `Continuous`, `Distribution`, `Factor`, `Sampler`, `ARGUMENT`, `RESIDUAL`) and add near the top:

```rust
#[derive(Debug, Clone)]
struct BoundSampler;

impl Sampler<B, Continuous> for BoundSampler {
    fn sample(&self, _batch_size: usize, _parameters: Vec<Input<B>>) -> Point<B> {
        unimplemented!("the tests never generate")
    }
}

/// A `Real` conditional over one parent whose factor binds a caller-supplied residual node —
/// enough to observe every binding `add_quantity` makes, with no arithmetic to get wrong.
struct BoundDistr {
    residual: Q<B, EuclideanD>,
    ids: [Id; 1],
}

impl Distribution<B, Continuous> for BoundDistr {
    type Qty = Real;
    type Sampler = BoundSampler;

    fn quantity_type(&self) -> Real {
        Real
    }
    fn sampler(&self) -> BoundSampler {
        BoundSampler
    }
    fn parent_ids(&self) -> &[Id] {
        &self.ids
    }
    fn factor(&self, x: &Q<B, Real>) -> SkoroResult<Factor<B>> {
        let density = register_map1(Real, LiteralScalarMultiplication::new(2.0), x)?;
        Ok(Factor::new(density).bind(RESIDUAL, &self.residual))
    }
}

/// `x ~ BoundDistr` over parent `p`, with `residual` bound to `x`'s term.
fn bound_model() -> (ModelClient<B>, Q<B, Real>, Q<B, Real>, Q<B, EuclideanD>) {
    let model = ModelClient::<B>::default();
    let p = model.add_parameter(Real);
    let residual = model.add_parameter(EuclideanD(1));
    let x = model
        .add_quantity(BoundDistr {
            residual: residual.clone(),
            ids: [p.id()],
        })
        .unwrap();
    (model, x, p, residual)
}
```

`Sampler` is `skoro_model::distribution::Sampler` (re-exported from `skoro_core`).

- [ ] **Step 3: Write the 4 remaining tests**

```rust
mod bindings {
    use super::*;

    #[test]
    fn add_quantity_binds_the_arguments() {
        let (model, x, p, _residual) = bound_model();
        let density = model.log_density_of(&x).unwrap();
        assert_eq!(model.bound(density, ARGUMENT), vec![x.id(), p.id()]);
    }

    #[test]
    fn add_quantity_binds_what_the_factor_declared() {
        let (model, x, _p, residual) = bound_model();
        let density = model.log_density_of(&x).unwrap();
        assert_eq!(model.bound(density, RESIDUAL), vec![residual.id()]);
    }

    #[test]
    fn remove_refuses_a_bound_node() {
        let (model, x, _p, residual) = bound_model();
        let density = model.log_density_of(&x).unwrap();
        assert!(matches!(
            model.remove(&[residual.id()]),
            Err(SkoroError::NodeStillBound { id, by }) if id == residual.id() && by == density
        ));
    }

    /// The demoted density and its bound residual leave in the same sweep: the density is no
    /// longer a root once retired, and nothing else references the residual once the test's own
    /// handle is gone. Both land in one `remove` batch, so `NodeStillBound` never fires.
    #[test]
    fn demotion_sweeps_the_bindings_with_the_density() {
        let (model, x, _p, residual) = bound_model();
        let residual_id = residual.id();
        drop(residual);

        model.demote_to_parameter(&x).unwrap();

        assert!(!model.server_call(|s| s.graph().contains(residual_id)));
        assert!(model.factors().is_empty());
    }
}
```

Fold the 5 tests already written in Tasks 9-12 (`add_factor_marks_and_binds`, `add_factor_needs_an_argument`, `a_bound_node_survives_cleanup`, `factors_are_in_term_order`, `residual_layout_follows_term_order_and_dimension` — currently under `mod bindings_via_add_factor`) into this same `mod bindings` block (rename the `mod` to match, or keep two sibling `mod`s if that reads more clearly — either satisfies the spec, which only requires the 9 tests exist somewhere in this file).

- [ ] **Step 4: Run and iterate**

Run: `cargo test -p skoro-model --test bindings`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/skoro-model/tests/bindings.rs
git rm --cached crates/skoro-model/tests/residual.rs 2>/dev/null || true
git commit -m "$(cat <<'EOF'
Rename residual.rs to bindings.rs; finish spec section 5.3's table

Drops the FactorRecord-based mod records/mod rejections. Adds the
remaining four bindings tests (add_quantity_binds_the_arguments,
add_quantity_binds_what_the_factor_declared, remove_refuses_a_bound_node,
demotion_sweeps_the_bindings_with_the_density) alongside the five
already written in Tasks 9-12.
EOF
)"
```

### Task 22: Adapt `skoro-normal`'s and `skoro`'s remaining tests (§5.8, the rest of §5.9)

**Files:**
- Modify: `crates/skoro-normal/tests/structure.rs`, `facade.rs`, `window.rs`, `retire.rs`, `demote.rs`
- Modify: `crates/skoro/tests/stacked_tangent.rs` (`mod records`, `mod retirement` — whatever Task 16 Step 8 did not already cover)

**Interfaces:**
- Consumes: `ModelClientOps::{factors, bound}`, `least_squares::residual_layout`, `GaussianBlock::all` (Task 19), `roles::{RESIDUAL, LOCATION, WHITENING}`, `client.add_quantity(Normal::new(mean, std_dev))` in place of `add_gaussian_quantity`.

This task is "adapt mechanically" per spec §5.8's own framing — read each file first (none were read while writing this plan; the mapping below is exact but the surrounding fixture code is not reproduced here), then apply the substitution table, keeping every behavior spec §5.8 lists as "must remain asserted":

| Old call | New call |
|---|---|
| `add_gaussian_quantity(&client, &mean, &std_dev)` | `client.add_quantity(Normal::new(mean.clone(), std_dev.clone()))` |
| `client.factors()` / `ModelBuilderOps::factors(&client)` | `ModelClientOps::factors(&client)` (returns `Vec<FactorInfo>`, not `Vec<FactorRecord>` — field names differ: `.density`, `.scored`, `.kind`, `.arguments`, no `.residual`/`.location`/`.whitening` directly) |
| `record.residual` | `client.bound(record.density, RESIDUAL)[0]` (or via `GaussianBlock::of`) |
| `record.location` / `record.whitening` | `client.bound(record.density, LOCATION)[0]` / `client.bound(record.density, WHITENING)[0]` (or via `GaussianBlock`) |
| `client.residual_layout()` | `skoro_model::least_squares::residual_layout(&client)` |
| `gaussian_blocks(&client)` | `GaussianBlock::all(&client)` |
| `FactorRecord::new(..).with_refinement(..)` (in any joint-factor test) | not applicable — joint factors already went through `add_joint_factor`/`Factor` in Task 16 |

- [ ] **Step 1: `structure.rs` — the behaviors spec §5.8 requires stay asserted**

Read the current file. Per §5.8: `whitened_residual` must still be the `r` in `−log p = ½‖r‖² + n` for `Real`, `EuclideanD` (non-diagonal `L`), and `SE3` (residual width 6); the builder must bind the residual, location, whitening, and normalizer parent it built; the bound density must be the model's own term. Rewrite each test's construction step from `add_gaussian_quantity` to `client.add_quantity(Normal::new(..))`, and its assertions from `FactorRecord` fields to `GaussianBlock::of(&client, &factor_info)` (or direct `bound(..)` calls) per the table above. `Normal::whitened_residual` itself is unchanged (Task 16 kept it public) — these tests likely call it directly on a `Normal` instance to get the "expected" `r`, independent of what the model recorded, so that half of each test probably needs no change at all.

- [ ] **Step 2: `retire.rs` — the behaviors spec §5.8 requires stay asserted**

Per §5.8: demoting a Gaussian quantity must remove its term from `factors()` (now `ModelClientOps::factors`), its rows from `residual_layout` (now `least_squares::residual_layout`), and (after `cleanup()`) its residual from the graph; `remove_closure(&[x])` must take the factor and its bindings; client-tier `remove` of a bound residual must be `Err(NodeStillBound)` (was `Err(NodeIsRecorded)` — the rename from Task 1, already applied at line 131 there since that file was in the earlier grep hit list; confirm it now reads `NodeStillBound`).

- [ ] **Step 3: `window.rs` — the behaviors spec §5.8 requires stay asserted**

Per §5.8: the sliding window must grow and retire without leaking, using `add_quantity` and `add_joint_factor`. Replace `add_gaussian_quantity(..)` with `model.add_quantity(Normal::new(..))`, `model.factors().len()` with `ModelClientOps::factors(&model).len()`, and `model.residual_layout().unwrap().total()` with `least_squares::residual_layout(&model).unwrap().total()` (`window.rs:133-142`). The steady-state assertion (`window.rs:147-153`) compares one tick's `graph_node_count` with the previous tick's, so it needs no numeric change even though the per-tick node count differs now: each Gaussian gains one root node and loses the facade's duplicate residual subgraph and its duplicate `invert`.

- [ ] **Step 4: `facade.rs` (test) / `demote.rs` — adapt mechanically**

Per §5.8: "adapt mechanically." Apply the substitution table; no additional behavior is called out for these two beyond what the table already covers.

- [ ] **Step 5: `crates/skoro/tests/stacked_tangent.rs` — finish what Task 16 Step 8 started**

Re-read `mod records` and `mod retirement` in full now (Task 16 only patched the minimum needed to keep `cargo test --workspace` green). Per spec §5.9: adapt `FactorInfo.arguments == ids`, `bound(density, RESIDUAL)` is the residual — finish converting every remaining `model.factors()[0].residual`/`.normalizer_parents` read to the `bound(density, ROLE)` form.

- [ ] **Step 6: Run and iterate**

Run: `cargo test -p skoro-normal`
Expected: PASS once Steps 1-4 are done.

Run: `cargo test -p skoro`
Expected: PASS once Step 5 is done.

- [ ] **Step 7: The Phase 4b checkpoint**

Run: `cargo test --workspace`
Expected: PASS.

Run:
```bash
grep -rn "FactorRecord\|ResidualRegistry\|record_factor\|residual_layout()\|add_gaussian_quantity\|NodeIsRecorded" crates readme.md
```
Expected: no output. (`residual_layout()` — the *function call* with no arguments, i.e. the old `ModelServer`/`ModelClient` inherent method — should be gone; `least_squares::residual_layout(&client)` will still legitimately match a bare `residual_layout(` grep without the `()`, so the exact pattern above with the trailing parens is what the spec's checkpoint uses and what must return nothing.)

- [ ] **Step 8: Commit**

```bash
git add crates/skoro-normal/tests/structure.rs crates/skoro-normal/tests/facade.rs \
  crates/skoro-normal/tests/window.rs crates/skoro-normal/tests/retire.rs \
  crates/skoro-normal/tests/demote.rs crates/skoro/tests/stacked_tangent.rs
git commit -m "$(cat <<'EOF'
Adapt skoro-normal/skoro tests to the bindings API (5.8, 5.9)

Mechanical substitution: add_gaussian_quantity -> add_quantity(Normal::new(..)),
FactorRecord fields -> bound(density, ROLE), residual_layout() ->
least_squares::residual_layout, gaussian_blocks -> GaussianBlock::all.
Every behavior spec section 5.8 lists as must-remain-asserted is
preserved. This is the Phase 4b checkpoint: no reference to the
deleted record machinery remains anywhere in crates/ or readme.md.
EOF
)"
```

**Phase 4b checkpoint (already run in Task 22, Step 7):** `cargo test --workspace` passes; the record-machinery grep returns nothing.

---

## Phase 5 — Fisher tests and docs

Implements spec §5.7 (`crates/skoro-normal/tests/information.rs` — finishing the file Task 15 started), the remaining row of §5.9 (`joint_information_is_the_precision` in `crates/skoro/tests/stacked_tangent.rs`), and R30 (`readme.md`). Final checkpoint: `cargo build --workspace`, `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo doc --workspace --no-deps` (§6, Phase 5).

**A note on the code below.** Every test here composes APIs from earlier phases (`Normal::factor`'s bindings, `information`/`information_root`, `jacobian_vjp`, `Dense`/`spd_solve`) that are individually verified by that phase's own tests, but this is the *first* code that exercises them together end-to-end. Treat the exact receiver style of `PointwiseOps`/`QAdd`/`MatmulOps` methods (`&self` vs. owned, operator overload vs. named method) as needing a one-line fix-up against the compiler if it differs from what is written — read `crates/skoro-pointwise/src/lib.rs` and `crates/skoro-model/src/ops.rs` first if the first compile attempt disagrees, rather than guessing twice. The *values* being compared (the mathematical claims) are what must not change.

### Task 23: `crates/skoro-normal/tests/information.rs` — finish spec §5.7's table

**Files:**
- Modify: `crates/skoro-normal/tests/information.rs` (created in Task 15 with one test; add the remaining 8 rows of spec §5.7's table)

**Interfaces:**
- Consumes: `skoro_model::information::{information, information_root}`, `skoro_model::jacobian::jacobian_vjp`, `Normal::factor`'s `FISHER_ROOT` binding (Task 16), `skoro_dense::{dense::Dense, factor::spd_solve}`.

- [ ] **Step 1: Expand the file's imports**

At the top of `crates/skoro-normal/tests/information.rs`, alongside the existing `use skoro_normal::information::gaussian_root_matrix;`, add:

```rust
use skoro_backend::{NdArrayBackend, Tensor, TensorData};
use skoro_core::error::SkoroError;
use skoro_dense::{dense::Dense, factor::spd_solve};
use skoro_linalg::{matrix::MatmulOps, tril::Tril};
use skoro_model::{
    ModelBuilderOps, ModelClientOps,
    hash_map_builders::points,
    information::{information, information_root},
    jacobian::jacobian_vjp,
    model::ModelClient,
    test_utils::into_vec,
};
use skoro_normal::distribution::Normal;
use skoro_pointwise::PointwiseOps;
use skoro_riemann::{
    euclidean::{EuclideanD, VectorD},
    matrix::{EuclideanMatrixD, MatrixD},
};
use skoro_se3::quantity::SE3;

type B = NdArrayBackend<f64, i32>;

fn row(values: &[f64]) -> Tensor<B, 2> {
    let n = values.len();
    Tensor::<B, 1>::from_data(TensorData::new(values.to_vec(), [n]), &Default::default())
        .reshape([1, n])
}

/// Unpack a row-major-packed `Tril(d)` vector into a dense `d x d` `Dense`.
fn unpack_tril(packed: &[f64], d: usize) -> Dense {
    let mut m = vec![0.0; d * d];
    let mut k = 0;
    for i in 0..d {
        for j in 0..=i {
            m[i * d + j] = packed[k];
            k += 1;
        }
    }
    Dense::from_row_major(d, d, m).unwrap()
}

/// `Σ⁻¹` for the covariance packed as `l_packed`, via the host `spd_solve` route §4.2 names.
fn precision_from_tril(l_packed: &[f64], d: usize) -> Dense {
    let l = unpack_tril(l_packed, d);
    let sigma = l.matmul(&l.transpose()).unwrap();
    spd_solve(&sigma, &Dense::identity(d)).unwrap()
}

fn assert_dense_close(got: &Dense, want: &Dense, epsilon: f64) {
    assert_eq!((got.rows(), got.cols()), (want.rows(), want.cols()));
    for (g, w) in got.data().iter().zip(want.data()) {
        approx::assert_abs_diff_eq!(g, w, epsilon = epsilon);
    }
}

/// Elementwise sum; `Dense` has no addition of its own.
fn add_dense(a: &Dense, b: &Dense) -> Dense {
    let data: Vec<f64> = a.data().iter().zip(b.data()).map(|(x, y)| x + y).collect();
    Dense::from_row_major(a.rows(), a.cols(), data).unwrap()
}
```

- [ ] **Step 2: `observed_gaussian_information_is_h_transpose_precision_h`**

```rust
#[test]
fn observed_gaussian_information_is_h_transpose_precision_h() {
    let model = ModelClient::<B>::default();
    let theta = model.add_parameter(EuclideanD(2));
    let a = model.add_constant(
        EuclideanMatrixD(3, 2),
        &MatrixD::new(3, 2, &[0.5, 0.2, 0.1, 0.6, 0.3, 0.4]),
    );
    let c = model.add_constant(EuclideanD(3), &VectorD::new(&[1.0, 1.0, 1.0]));
    // A*theta_val + c > 0 for the theta this test evaluates at.
    let linear = a.matmul(&theta).unwrap();
    let mu = (linear + c).unwrap().log().unwrap();

    let l_packed = [1.2, 0.1, 0.9, -0.2, 0.3, 1.1];
    let l = model.add_constant(Tril(3), &VectorD::new(&l_packed));
    let y = model.add_quantity(Normal::new(mu.clone(), l)).unwrap();
    y.observe(&VectorD::new(&[0.9, 1.0, 1.1])).unwrap();

    let theta_val = points().with_quantity(&theta, row(&[0.3, -0.2])).build();
    let free = model.tangent_layout(&[theta.id()]).unwrap();

    let info = information(&model, &theta_val, &free).unwrap();
    let outputs = model.tangent_layout(&[mu.id()]).unwrap();
    let h = Dense::from_tensor(jacobian_vjp(&model, &theta_val, &free, &outputs).unwrap()).unwrap();
    let sigma_inv = precision_from_tril(&l_packed, 3);
    let expected = h.transpose().matmul(&sigma_inv).unwrap().matmul(&h).unwrap();

    assert_dense_close(&Dense::from_tensor(info).unwrap(), &expected, 1e-9);

    let root = information_root(&model, &theta_val, &free).unwrap();
    assert_eq!(root.dims(), [1, 3 + 9, 2]);
}
```

- [ ] **Step 3: `gaussian_prior_information_is_the_precision`**

```rust
#[test]
fn gaussian_prior_information_is_the_precision() {
    let model = ModelClient::<B>::default();
    let mu = model.add_constant(EuclideanD(2), &VectorD::new(&[0.5, -0.5]));
    let l_packed = [1.4, 0.3, 0.8];
    let l = model.add_constant(Tril(2), &VectorD::new(&l_packed));
    let x = model.add_quantity(Normal::new(mu, l)).unwrap();

    let values = points().with_quantity(&x, row(&[0.2, 0.1])).build();
    let free = model.tangent_layout(&[x.id()]).unwrap();
    let info = information(&model, &values, &free).unwrap();

    let expected = precision_from_tril(&l_packed, 2);
    assert_dense_close(&Dense::from_tensor(info).unwrap(), &expected, 1e-9);
}
```

- [ ] **Step 4: `covariance_information_matches_the_trace_formula`**

```rust
#[test]
fn covariance_information_matches_the_trace_formula() {
    let model = ModelClient::<B>::default();
    let mu = model.add_constant(EuclideanD(2), &VectorD::new(&[0.0, 0.0]));
    let l = model.add_parameter(Tril(2));
    let l_val = [1.5, -0.4, 0.8];
    let x = model.add_quantity(Normal::new(mu, l.clone())).unwrap();
    x.observe(&VectorD::new(&[0.3, -0.6])).unwrap();

    let values = points().with_quantity(&l, row(&l_val)).build();
    let free = model.tangent_layout(&[l.id()]).unwrap();
    let info = information(&model, &values, &free).unwrap();

    // Host F_LL per spec section 4.2: dSigma_q = E_q L^T + L E_q^T,
    // F_LL[q][q'] = 1/2 trace(Sigma^-1 dSigma_q Sigma^-1 dSigma_q').
    let d = 2;
    let l_dense = unpack_tril(&l_val, d);
    let sigma = l_dense.matmul(&l_dense.transpose()).unwrap();
    let sigma_inv = spd_solve(&sigma, &Dense::identity(d)).unwrap();
    let d_l = d * (d + 1) / 2;
    let mut expected = vec![0.0; d_l * d_l];
    let packed = |i: usize, j: usize| i * (i + 1) / 2 + j;
    let e = |a: usize, b: usize| {
        let mut m = vec![0.0; d * d];
        m[a * d + b] = 1.0;
        Dense::from_row_major(d, d, m).unwrap()
    };
    for a in 0..d {
        for b in 0..=a {
            for a2 in 0..d {
                for b2 in 0..=a2 {
                    let dsq = e(a, b)
                        .matmul(&l_dense.transpose())
                        .unwrap();
                    let dsq_t = l_dense.matmul(&e(a, b).transpose()).unwrap();
                    let dsq = add_dense(&dsq, &dsq_t);
                    let dsq2 = {
                        let x = e(a2, b2).matmul(&l_dense.transpose()).unwrap();
                        let y = l_dense.matmul(&e(a2, b2).transpose()).unwrap();
                        add_dense(&x, &y)
                    };
                    let trace = sigma_inv
                        .matmul(&dsq)
                        .unwrap()
                        .matmul(&sigma_inv)
                        .unwrap()
                        .matmul(&dsq2)
                        .unwrap();
                    let t: f64 = (0..d).map(|i| trace.get(i, i)).sum();
                    expected[packed(a, b) * d_l + packed(a2, b2)] = 0.5 * t;
                }
            }
        }
    }
    let expected = Dense::from_row_major(d_l, d_l, expected).unwrap();
    assert_dense_close(&Dense::from_tensor(info).unwrap(), &expected, 1e-9);
}
```

`add_dense` is the helper from Step 1.

- [ ] **Step 5: `covariance_information_matches_the_score_covariance`**

```rust
#[test]
fn covariance_information_matches_the_score_covariance() {
    let model = ModelClient::<B>::default();
    let mu = model.add_constant(EuclideanD(2), &VectorD::new(&[0.0, 0.0]));
    let l = model.add_parameter(Tril(2));
    let l_val = [1.5, -0.4, 0.8];
    let x = model.add_quantity(Normal::new(mu, l.clone())).unwrap();

    // Draw x ~ N(mu, L Lᵀ) n times at the fixed L, one row per draw. A Parameter must be
    // supplied at the running batch, so L is repeated across the rows.
    let n = 40_000;
    let l_batched = Tensor::<B, 2>::from_data(
        TensorData::new(l_val.repeat(n), [n, 3]),
        &Default::default(),
    );
    let generated = model
        .generate_from(&points().with_quantity(&l, l_batched.clone()).build(), n)
        .unwrap();
    let (x_sampled, _) = generated.variables.get(&x.id()).expect("x is sampled");
    let x_sampled = x_sampled.clone().unwrap_float();

    // The score with respect to L at every draw: [n, 3].
    let values = points()
        .with_quantity(&x, x_sampled)
        .with_quantity(&l, l_batched)
        .build();
    let ord1 = model.log_density_ord1(&values).unwrap();
    let scores = into_vec(ord1.cotangents[&l.id()].clone());

    // mean_i(g_i g_iᵀ) over the batch, as a 3x3 host matrix.
    let mut score_cov = vec![0.0; 9];
    for i in 0..n {
        for p in 0..3 {
            for q in 0..3 {
                score_cov[p * 3 + q] += scores[i * 3 + p] * scores[i * 3 + q] / n as f64;
            }
        }
    }

    // The expected information at batch 1. `x` is an unobserved leaf, so it needs a value
    // even though the root does not depend on it.
    let free = model.tangent_layout(&[l.id()]).unwrap();
    let at = points()
        .with_quantity(&x, row(&[0.0, 0.0]))
        .with_quantity(&l, row(&l_val))
        .build();
    let info = Dense::from_tensor(information(&model, &at, &free).unwrap()).unwrap();

    for (g, w) in info.data().iter().zip(&score_cov) {
        approx::assert_abs_diff_eq!(g, w, epsilon = 0.05 * g.abs().max(1.0));
    }
}
```

`PointEvalOrder1` (`crates/skoro-graph/src/graph.rs`) has public fields `value` and `cotangents: HashMap<Id, CotangentVector<B>>`; `ForwardOutput::variables` is the `PointMap` holding sampled leaves, and `PointMap::get` returns an `(&Input, &Arc<dyn DynQuantityType>)` pair.

- [ ] **Step 6: `information_is_the_gram_of_the_root`**

```rust
#[test]
fn information_is_the_gram_of_the_root() {
    let model = ModelClient::<B>::default();
    let mu = model.add_constant(EuclideanD(2), &VectorD::new(&[0.5, -0.5]));
    let l = model.add_constant(Tril(2), &VectorD::new(&[1.4, 0.3, 0.8]));
    let x = model.add_quantity(Normal::new(mu, l)).unwrap();

    let values = points().with_quantity(&x, row(&[0.2, 0.1])).build();
    let free = model.tangent_layout(&[x.id()]).unwrap();

    let info = information(&model, &values, &free).unwrap();
    let root = information_root(&model, &values, &free).unwrap();
    let gram = root.clone().swap_dims(1, 2).matmul(root);

    for (g, w) in into_vec(info).iter().zip(into_vec(gram)) {
        approx::assert_abs_diff_eq!(g, w, epsilon = 1e-12);
    }
}
```

- [ ] **Step 7: `se3_root_is_sized_by_dimension`**

```rust
#[test]
fn se3_root_is_sized_by_dimension() {
    let model = ModelClient::<B>::default();
    let mu = model.add_parameter(SE3);
    let l = model.add_parameter(Tril(6));
    let x = model.add_quantity(Normal::new(mu.clone(), l.clone())).unwrap();

    let density = model.log_density_of(&x).unwrap();
    let root_id = model.bound(density, skoro_model::roles::FISHER_ROOT)[0];
    let shape = model.server_call(|s| {
        let qty = s.graph().node(root_id).unwrap().quantity_type();
        let m = qty
            .as_any()
            .downcast_ref::<EuclideanMatrixD>()
            .expect("the root is a matrix quantity");
        (m.0, m.1)
    });
    assert_eq!(shape, (42, 33), "6 + 36 rows over 6 + 6 + 21 argument columns");

    let args = model.bound(density, skoro_model::roles::ARGUMENT);
    let layout = model.tangent_layout(&args).unwrap();
    assert_eq!(layout.total(), 6 + 6 + 21);
}
```

`Node::quantity_type` (`crates/skoro-graph/src/node.rs:42`) returns `&dyn DynQuantityType<B>`, whose `as_any` yields the wrapped `EuclideanMatrixD`.

- [ ] **Step 8: `a_constant_covariance_folds_the_root`**

```rust
#[test]
fn a_constant_covariance_folds_the_root() {
    let model = ModelClient::<B>::default();
    let mu = model.add_constant(EuclideanD(2), &VectorD::new(&[0.0, 0.0]));
    let l_const = model.add_constant(Tril(2), &VectorD::new(&[1.0, 0.0, 1.0]));
    let x = model.add_quantity(Normal::new(mu.clone(), l_const)).unwrap();
    let density = ModelClientOps::log_density_of(&model, &x).unwrap();
    let root_id = ModelClientOps::bound(&model, density, skoro_model::roles::FISHER_ROOT)[0];
    assert!(model.server_call(|s| matches!(
        s.graph().node(root_id),
        Some(skoro_graph::node::Node::Mapped(skoro_graph::node::MappedNode { value: Some(_), .. }))
    )));

    let mu2 = model.add_constant(EuclideanD(2), &VectorD::new(&[0.0, 0.0]));
    let l_param = model.add_parameter(Tril(2));
    let x2 = model.add_quantity(Normal::new(mu2, l_param)).unwrap();
    let density2 = ModelClientOps::log_density_of(&model, &x2).unwrap();
    let root_id2 = ModelClientOps::bound(&model, density2, skoro_model::roles::FISHER_ROOT)[0];
    assert!(model.server_call(|s| matches!(
        s.graph().node(root_id2),
        Some(skoro_graph::node::Node::Mapped(skoro_graph::node::MappedNode { value: None, .. }))
    )));
}
```

- [ ] **Step 9: `a_mismatched_covariance_is_refused`**

```rust
#[test]
fn a_mismatched_covariance_is_refused() {
    let model = ModelClient::<B>::default();
    let mu = model.add_constant(EuclideanD(3), &VectorD::new(&[0.0, 0.0, 0.0]));
    let l = model.add_constant(Tril(2), &VectorD::new(&[1.0, 0.0, 1.0]));
    assert!(matches!(
        model.add_quantity(Normal::new(mu, l)),
        Err(SkoroError::ShapeMismatch {
            expected: 3,
            found: 2
        })
    ));
}
```

- [ ] **Step 10: Run and iterate**

Run: `cargo test -p skoro-normal --test information`
Expected: PASS. Do not weaken a tolerance or an assertion to make a test pass; if a value disagrees, the bug is upstream (most likely a sign or transpose term in `gaussian_root_matrix`) — go back to Task 15's hand-derivation test and extend it before touching this file further.

- [ ] **Step 11: Commit**

```bash
git add crates/skoro-normal/tests/information.rs
git commit -m "$(cat <<'EOF'
Add the Fisher information integration tests (5.7)

End-to-end coverage of Normal::factor's FISHER_ROOT binding against
host references: H^T Sigma^-1 H for an observed nonlinear-mean
Gaussian, the raw precision for a Gaussian prior, the trace formula
and the score-covariance identity for the covariance block, the
Gram identity information = information_root^T information_root,
SE3 sizing, constant-folding, and the dimension-mismatch refusal.
EOF
)"
```

### Task 24: `joint_information_is_the_precision` (remaining row of §5.9)

**Files:**
- Modify: `crates/skoro/tests/stacked_tangent.rs` (new `mod information`)

**Interfaces:**
- Consumes: `information` (Task 14), the existing `coupled_precision()` fixture (`crates/skoro/tests/stacked_tangent.rs:59-72`), `JointNormal::factor`'s `FISHER_ROOT` binding (Task 16).

- [ ] **Step 1: Write the test**

Append a new module, following the exact fixture pattern already used by `mod records`'s `the_builder_records_a_joint_factor_as_one_row_block` (read at `crates/skoro/tests/stacked_tangent.rs:407-451` while writing this plan):

```rust
mod information {
    use super::*;
    use skoro_model::information::information;

    #[test]
    fn joint_information_is_the_precision() {
        let model = ModelClient::<B>::default();
        let pose_q: Q<B, SE3> = model.add_parameter(SE3);
        let point_q: Q<B, EuclideanD> = model.add_parameter(EuclideanD(3));

        let at = point_map()
            .with_quantity(&pose_q, pose_row(0.4, [1.5, -2.25, 0.75]))
            .with_quantity(&point_q, row(&[1.0, 2.0, 3.0]))
            .build();
        let (packed, normalizer) = whitening_from_precision(&coupled_precision()).unwrap();
        let whitening = model.add_constant(Tril(9), &VectorD::new(&packed));

        let factor = JointNormal::new(
            vec![pose_q.id(), point_q.id()],
            at.clone(),
            whitening,
            normalizer,
        );
        add_joint_factor(&model, &factor).unwrap();

        let free = model.tangent_layout(&[pose_q.id(), point_q.id()]).unwrap();
        let info = information(&model, at.as_ref(), &free).unwrap();
        let info = Dense::from_tensor(info).unwrap();

        let lambda = coupled_precision();
        for (g, w) in info.data().iter().zip(lambda.data()) {
            approx::assert_abs_diff_eq!(g, w, epsilon = 1e-8);
        }
    }
}
```

- [ ] **Step 2: Run the test**

Run: `cargo test -p skoro --test stacked_tangent joint_information_is_the_precision`
Expected: PASS. If it fails on shape or sign, the bug is in `JointNormal::factor`'s scatter matrix `P` (Task 16, Step 7) — check `P[i·D+j, i(i+1)/2+j] = 1` was built with the right row/column orientation (rows indexed by the *dense* `D×D` position, columns by the *packed* `Tril` position) before doubting `information`'s own assembly, which Task 14's construction already composes out of pieces (`jacobian_vjp`, the evaluated root, a `matmul`) each independently plausible.

- [ ] **Step 3: Commit**

```bash
git add crates/skoro/tests/stacked_tangent.rs
git commit -m "$(cat <<'EOF'
Add joint_information_is_the_precision (5.9)

For the coupled_precision() SE3+EuclideanD(3) fixture, information()
over both members equals the 9x9 precision the joint factor was
built from -- end-to-end proof that JointNormal::factor's dense
whitening root (C = W, C^T C = W^T W = Lambda) assembles correctly
through information_blocks/jacobian_vjp.
EOF
)"
```

### Task 25: `readme.md` (R30)

**Files:**
- Modify: `readme.md`

**Interfaces:**
- None — documentation only.

- [ ] **Step 1: Update the bullets**

Read `readme.md`'s current `skoro-graph`, `skoro-model`, `skoro-linalg`, and `skoro-normal` bullets. Per R30: describe bindings and roles under `skoro-graph`/`skoro-model`, `Factor` and the Fisher root under `skoro-model`/`skoro-normal`, and `FixedLinearMap`/`linear_map` under `skoro-linalg`. Remove every mention of `FactorRecord`, `ResidualRegistry`, `add_gaussian_quantity`.

- [ ] **Step 2: Verify**

Run:
```bash
grep -rn "FactorRecord\|ResidualRegistry\|record_factor\|residual_layout()\|add_gaussian_quantity\|NodeIsRecorded" crates readme.md
```
Expected: no output (this is the Phase 4b checkpoint's grep, re-run here since `readme.md` is the one file that check also covers and this task is the last thing that could reintroduce a hit).

- [ ] **Step 3: Commit**

```bash
git add readme.md
git commit -m "$(cat <<'EOF'
Update readme.md for bindings and Fisher roots (R30)
EOF
)"
```

### Final checkpoint

- [ ] Run: `cargo build --workspace` — expect PASS.
- [ ] Run: `cargo test --workspace` — expect PASS.
- [ ] Run: `cargo clippy --workspace --all-targets -- -D warnings` — expect PASS. Fix any lint this plan's code introduced (likely candidates: unused imports left behind by a deleted function, a `needless_range_loop` in `gaussian_root_matrix`/`scatter_matrix`'s nested loops, or a redundant `.clone()`) rather than suppressing it.
- [ ] Run: `cargo doc --workspace --no-deps` — expect PASS, no broken intra-doc links (check especially the doc comments this plan added that link to `[Graph::bind]`-style paths in `binding.rs`, `term.rs`, `information.rs`, `least_squares.rs`, `roles.rs`).

---

## Self-review notes (from writing this plan)

**Spec coverage.** Every requirement R1–R30 has a task: R1→Task 1, R2–R4→Task 5, R5–R11→Tasks 2–4, R12→Task 6, R13→Task 7, R14→Tasks 8 & 16, R15→Task 9, R16→Tasks 9 & 16, R17→Task 10, R18→Task 17, R19→Task 11, R20→Tasks 9/17, R21→Task 12, R22→Task 13, R23→Task 14, R24→Task 15, R25–R26→Task 16, R27→Task 18, R28→Task 19, R29→Task 20, R30→Task 25. Every §5 test table has a task: §5.1→Tasks 2–4, §5.2→Task 5, §5.3→Tasks 9–12 & 21, §5.4→Task 16 Step 8b (`terms.rs` needs `Distribution::factor`, so it lands with the trait change), §5.5→Task 13, §5.6→Task 16, §5.7→Task 23, §5.8→Task 22, §5.9→Tasks 22 & 24.

**Known gaps to resolve during execution, not before.** Three places in this plan are deliberately underspecified because the underlying file was not read while writing it (reading every file in a 30-requirement, 9-test-file spec before writing a word would have cost more than the plan is worth): the exact `PointEvalOrder1`/cotangent-accessor shape in Task 23 Step 5, the exact fixture names to reuse in Task 21 Step 2, and the precise substitution needed in Task 22's five files. Each of those steps says so explicitly and names the nearest existing file to pattern-match against. Treat a compiler error there as expected, not as a sign the plan is wrong.

**Type/signature consistency check.** `Factor<B>`, `Handle<B>`, `Role`, `FactorInfo`, `Term`/`TermKind`, `ARGUMENT`/`RESIDUAL`/etc. are each defined exactly once (Tasks 6, 7, 8, 2/6, 4, 11) and every later task's code uses those exact names — cross-checked while drafting Tasks 9–24 against the definitions in Tasks 2–8.
