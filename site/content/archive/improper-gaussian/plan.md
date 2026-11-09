---
summary: The core carries values of no entries through every pass, and skoro-normal builds the improper Gaussian factor over a stacked tangent, as a retirement's replacement or a likelihood factor with its roles bound.
depends_on: [archive/improper-gaussian/implementation]
---

The build has two parts, in this order. First the core carries values of no entries, by a
`broadcast` and a `reshape` in the tensor layer that every call of burn's `expand`, and every
reshape of a tensor that may have no entries, goes through
([A factor of no rows](implementation.md#a-factor-of-no-rows)). Then `skoro-normal` gains
`Improper`, built by `Stacked::improper`, and the roles' descriptions say what the math decided
([The factor](implementation.md#the-factor), [The roles' descriptions](implementation.md#the-roles-descriptions)).
The umbrella's tests then check the factor in retirements and on the sphere.

## The tensor layer (`skoro-tensor`)

- [x] `broadcast` and `reshape` in `value.rs`, both making a tensor of no entries with
      `Tensor::empty`, exported from the crate's root, and the two calls of `expand` in `host.rs`
      replaced by `broadcast`. Tests: `broadcast_to_no_entries`, `reshape_to_no_entries`,
      `values_of_no_entries_load`.

## The core (`skoro-core`)

- [x] The fourteen calls of `expand` in `program.rs` and in the primitives' `rule.rs`,
      `euclidean.rs` and `positive.rs` replaced by `broadcast`, and the seven reshapes of
      `euclidean.rs`, `rule.rs` and `positive.rs` replaced by `reshape`, as the implementation's two
      tables list them, and any further kernel that fails on values of no entries fixed where it fails, or the
      build stopped and the failure reported if a fix needs more than a guard around one burn call.
      Tests: `values_of_no_entries_through_every_pass`.

## The normal (`skoro-normal`)

- [x] `Binds` made `pub(crate)` in `stacked.rs`; `improper.rs` with `Improper`, `Stacked::improper`,
      `Improper::linear`, its accessors, `log_density`, `replacement` and `factor`, and the shared
      binding of the residual and the linear part; `Improper` exported from the crate's root, and
      the crate's documentation naming it. Tests: `value_is_the_definition`, `roles_are_bound`,
      `rows_differ_across_a_batch`, `no_blocks_is_refused`.
- [x] The tests of the mathematics' statements and examples on the flat-in-disguise types. Tests:
      `full_rank_is_the_stacked_normal`, `forms_give_one_function`, `positive_example_is_exact`,
      `gauss_newton_is_the_hessian_on_flat_types`.
- [x] The factor of no rows, with and without a linear part, through every pass. Tests:
      `no_rows_is_its_linear_part`.
- [ ] The descriptions of the module, `RESIDUAL` and `LOG_VOLUME` in `roles.rs`, as the
      implementation gives them. Tests: none; `cargo doc` resolves the link to `Improper`.

## The umbrella's tests (`skoro`)

- [ ] The improper factor in retirements: the planar bearings, the bindings on the factor a
      retirement adds, and a blanket quantity left undetermined. Tests:
      `retirement.rs::bearings_accumulate`, `retirement.rs::improper_replacement_binds_its_roles`,
      `retirement.rs::undetermined_blanket_needs_a_proposal`.
- [ ] The improper factor on the acceptance-test sphere. Tests:
      `sphere.rs::improper_factor_on_the_sphere`.

## Done when

Every box is ticked, and these pass from the root:

```sh
bin/wf check
cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cd site && pnpm check && pnpm lint && pnpm test && pnpm build
```

The first two tasks change `skoro-tensor` and the primitives' kernels, so the branch is pushed only
after

```sh
cargo test --workspace --features wgpu
```

passes on a machine with an adapter, and the commit that records it says so. A failure there on
values of no entries is reported, as one on the host is, rather than worked around.

## Revisions

**2026-11-09.** The implementation was reopened when the build found that burn's `reshape` reads a
dimension of zero as the source's size. The first task, built and ticked with the first body of
`broadcast`, is unticked and now adds `reshape` too, with `broadcast`'s corrected body and the test
`reshape_to_no_entries`; the second replaces the seven reshapes of the core as well as its calls of
`expand`.
