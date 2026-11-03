---
summary: The stacked tangent gives a retirement a ready-made replacement, its log-density with the residual, mean and scale bound as the factor binds them.
depends_on: [archive/normal/implementation, archive/replacement-bindings/plan]
---

`Stacked::factor` adds the replacement form as a likelihood factor and binds its whitened residual,
its mean and its scale; `Stacked::log_density` gives the log-density alone, for a retirement's
`Replacement`, which now carries the roles its factor binds. This chore gives `Stacked` a third
method, `replacement(mean, scale, constant)`, that returns that `Replacement` whole: the
log-density, no proposals, and the three bindings `factor` makes, from one private helper that both
use, so that the two cannot bind differently. A caller adds proposals, or further bindings, to the
`Replacement` it is given. `log_density` stays, for a caller that wants the log-density alone, and
its documentation points to `replacement`.

Changing `log_density` to return a `Replacement` was the alternative: it would break its callers,
and its name would no longer say what it gives.

## The normal (`skoro-normal`, `stacked`)

- [x] `Stacked::replacement`, and the bindings it and `Stacked::factor` share; the documentation of
      both and of `Stacked::log_density`. Tests: `stacked_replacement_binds_the_normal_roles`, in
      the umbrella's retirement tests: a retirement replaced by `Stacked::replacement` adds a factor
      whose residual, mean and scale are bound under the normal's roles, the residual equal to
      $W(\delta - m)$ at points of the blanket, and whose log-density is that of
      `Stacked::log_density`.

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
