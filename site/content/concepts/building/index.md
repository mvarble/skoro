---
title: Building a model
summary: A model is built inside a scope a writer opens, by ordinary functions of handles, and published as an immutable snapshot that may be read from any thread or forked into a branch.
order: 3
depends_on: [concepts/operations]
---

A model is built forward, the way it is thought of: a quantity, then what is computed from it,
then the next quantity. Building happens inside a **scope** that a **writer** opens, by ordinary
functions of [handles](../spaces/#typed-handles) that name no writer, device or precision, and
the writer publishes what was built as an immutable **snapshot**. This page describes the writer,
the scope and its functions, what happens on a mistake, and snapshots and branches.

## The writer and its scope

```rust
{../../../../crates/skoro-core/src/writer.rs:33..47}
```

`build` opens a building scope on the current thread and runs the closure in it. Every function
that builds (making a quantity, applying an [operation](../operations/), adding a term) adds its
node to the innermost open scope, which is why a function of handles needs no model passed to it:
the same function builds into whichever model's scope it is called in. When the closure returns,
the writer publishes the new model.

The scope's own functions make the roots of a model, its quantities and constants, and place
quantities at addresses:

```rust
{../../../../crates/skoro-core/src/scope.rs:319..351}
```

A **quantity** is an uncertain value of the model, made at an **address**: a path of names and
indices such as `tick/1/x`, where a segment of digits is an index. `namespace` prefixes the
addresses of everything made inside it, so the model of one tick can be written once, as a
function, and called under `tick/0`, `tick/1` and so on. `lookup` finds a quantity by its address
under the open namespaces. A **constant** is a value fixed when the model is built; a constant
and anything computed from constants alone is folded into a value as soon as it is added.

## The compilation graph

What a scope builds is the model's **compilation graph**. Its nodes are the model's values: the
quantities and constants, which are its roots, and the outputs of operations, among them the
log-densities of [terms](../terms/). Its edges run from each input of an operation to its output.
The graph only grows: each scope adds nodes on top of the previous snapshot's, and nothing already
published changes.

## Mistakes, and where they were made

Building never panics on a mistake in the model. Each building function checks its inputs (that
they exist in this model, that their spaces suit the operation, that an address is not taken) and,
on a fault, records it and returns a **failed** handle. Everything built on a failed handle fails
in turn, and the scope publishes nothing. `build` then returns the first fault, with the line of
user code that caused it:

```rust
{../../../../crates/skoro-core/tests/building.rs:52..75}
```

So the failure points at `add(x, v)`, where a real number was added to a vector, rather than at
the second mistake or at whatever later tried to use the model.

## Snapshots, readers and branches

A snapshot is an immutable model, cheap to clone, which the queries of
[the next pages](../queries/) are compiled from. A writer hands out its latest snapshot, and a
`Reader` that loads the newest one from any thread, so estimators can run on snapshots while the
writer goes on building.

A snapshot can also be **forked** into a writer of its own: a branch, which shares everything with
the snapshot it came from and changes independently of it. This is how a model is held under
several hypotheses at once.

```rust
{../../../../crates/skoro-core/tests/branches.rs:41..55}
```

A handle stays valid on every branch that shares its node, so the quantities made before the fork
are named the same way on each.

> [!DECISION] One writer per branch, many snapshots
> A model changes only through its writer, by building, forking or retiring, and is read only
> through snapshots. See
> [position 3 of the charter](../../archive/charter/writeup.md#3-one-writer-per-branch-many-snapshots).
