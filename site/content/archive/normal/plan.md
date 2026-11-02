---
summary: The crate skoro-normal built in order, from its roles, quantile and scale through the term, the sampler and the information root, then the umbrella's curved tests and the stacked tangent, each task a commit with its tests.
depends_on: [archive/normal/implementation]
---

Each task is one commit, and the workspace compiles and passes its tests after each. The design of
every item is in [the implementation companion](implementation.md); the tests named are those of
its [Tests](implementation.md#tests), each in exactly one task, run at both precisions of the Flex
device where the companion compares a value.

## The crate (`skoro-normal`)

- [x] The crate `crates/skoro-normal`, a member of the workspace depending on `skoro-core` and
      `skoro-tensor` alone, with the workspace's lints; `lib.rs` and its re-exports; `roles.rs` with
      every role key and `roles::scale`; `refuse.rs` with `Refused`; `sample.rs` with `quantile`,
      Wichura's AS 241. Tests: `quantile_matches_reference`.
- [x] `scale.rs`: `Form` and its name, `Scale` from quantities in the four forms, `triangular(d)`,
      and the constant helpers `sd`, `sds`, `covariance` and `precision`, with the host's Cholesky
      and reversed Cholesky factorizations and the refusal of a matrix that is not positive
      definite with `ValueShape`. Tests: `constant_scale_helpers`.
- [x] `composites.rs`: the cache keyed by form, dimension and part, and the composites
      `skoro.normal.whiten`, `skoro.normal.normalizer` and `skoro.normal.matrix` for every form,
      the covariance factor's forward substitution among them. Tests: `forward_substitution_solves`.
- [x] `normal.rs`: `Normal::on`, `Normal::in_space`, `Parts`, `Normal::log_density` with the
      log-volume left out on the Euclidean spaces, and `Normal::factor_at` with its residual, mean
      and scale bound. Tests: `term_is_the_closed_form`, `density_integrates_to_one`.
- [x] `NormalSample`, compiling the space's retraction for the host when made and drawing
      $\mu \oplus L \Phi^{-1}(u)$ in each form; `Normal::score`, with the check of the space's
      domain and the refusals through `Refused`, the exact sampler, and the roles but the root.
      Tests: `draws_are_functions_of_key_and_address`, `draws_have_the_normal_moments`,
      `exact_sampler_leaves_the_weight_to_the_likelihoods`, `same_key_same_draw_at_both_precisions`,
      `refuses_spaces_it_does_not_take`.
- [x] The composite `skoro.normal.root` for every form, its rows and columns in the companion's
      order, and its binding to `INFORMATION_ROOT` by `score`. Tests:
      `composites_pass_check_operation`, `root_squares_to_the_information`, `roles_are_bound`,
      `four_forms_agree`.

## The umbrella's tests (`skoro`)

- [x] `skoro-normal` as a development dependency of `skoro`; the guarded exponential sphere with
      its volume moved, unchanged, from `tests/sphere.rs` to `tests/common/sphere.rs`, which
      `tests/sphere.rs` then uses; in `tests/common/sphere.rs`, the sphere declared under another
      name without a volume, and the host values of the sphere's marker. Tests: the existing
      tests of `tests/sphere.rs`, unchanged and passing.
- [x] `tests/normal.rs`, the normal on the acceptance-test sphere: its density, its term, its
      sampler and its information. Tests: `normal.rs::sphere_density_integrates_to_one`,
      `normal.rs::sphere_term_reads_the_declared_volume`,
      `normal.rs::sphere_sampler_matches_the_density`, `normal.rs::sphere_information`.
- [ ] The Gauss–Newton iteration of the companion's test, with and without the log-volume's
      gradient, against the mode found by a dense search. Tests:
      `normal.rs::gauss_newton_reaches_the_model_mode`.
- [ ] The refusals on curved types, and the factor accepted where the term is refused. Tests:
      `normal.rs::refuses_a_chart_that_is_not_global`,
      `normal.rs::refuses_a_type_without_a_volume`.

## The stacked tangent (`skoro-normal`, tested in `skoro`)

- [ ] `stacked.rs`: `Stacked` with `block`, `tangent`, `log_density` and `factor`, the anchors made
      as constants from their markers, and the documentation on `log_density` that a replacement
      written with it binds no roles until the core's `Replacement` carries bindings. Tests:
      `normal.rs::stacked_tangent_on_the_sphere`, `retirement.rs::replacements_written_with_stacked`.

## Done when

Every box is ticked, and these pass from the root:

```sh
bin/wf check
cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cd site && pnpm check && pnpm lint && pnpm test && pnpm build
```

No task changes `skoro-tensor`, a primitive's kernel or a fused operation, and `skoro-normal` has
no kernel of its own, so the local run with `--features wgpu` is not required before pushing.
