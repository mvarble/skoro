---
summary: A retirement's replacement carries the roles its factor binds, checked and bound in the one snapshot the retirement publishes, and the retirement gives back the factor's handle.
depends_on: [archive/core/implementation]
---

`Retirement::replace` adds the replacement's log-density as a likelihood factor and gives back
nothing, so what the caller built for it cannot be bound to a role. This chore makes the term the
core constructs transparent to the caller in two ways, without changing what a retirement computes:

- **The replacement carries its bindings.** `Replacement` gains a field `bindings` of role names
  and nodes, and a method `bind(role, node)` that appends one, as `TermHandle::bind` does on a term
  in a scope. `replace` checks each binding as `TermHandle::bind` does, a node reached from no
  quantity outside the factor's (`BindingOutsideTerm`), with the replacement's other checks, before
  anything is removed; and binds them to the factor it adds, in the same snapshot, so that a
  failure still leaves the model as it was.
- **The retirement gives back the factor.** `replace` returns the factor's `TermHandle`, or `None`
  for a retirement of nothing, which adds no factor, so that a caller can look the factor up in the
  snapshot.

Binding on a term after the retirement, in a scope of its own, was the alternative to the first: it
would publish a snapshot whose replacement has no roles before the one that binds them, and a
reader of the first would find none.

## The core (`skoro-core`, `retire`)

- [x] `Replacement::bindings` and `Replacement::bind`; the bindings' check and binding in
      `Retirement::replace`, and its return of the factor's handle; its documentation; every
      replacement written in the core's and the umbrella's tests given its bindings, empty.
      Tests: `replacement_binds_its_roles`, `replacement_binding_outside_its_factor_is_refused`,
      `retirement_of_nothing_adds_no_factor`.

## The normal (`skoro-normal`, `stacked`)

- [ ] The documentation of `Stacked::log_density`, which says a replacement written with it binds
      no roles until the core's `Replacement` carries bindings: it now says that the replacement it
      gives binds none, and that a caller binds them with `Replacement::bind`. Tests: none; the
      crate's tests pass unchanged.

## Done when

Every box is ticked, and these pass from the root:

```sh
bin/wf check
cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cd site && pnpm check && pnpm lint && pnpm test && pnpm build
```

No task changes `skoro-tensor`, a primitive's kernel or a fused operation, so the local run with
`--features wgpu` is not required before pushing.
