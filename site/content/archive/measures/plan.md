---
summary: Reference measures and chart volumes built in order, from the declaration and the spaces' log-volumes through the core's refusals to the volume check and the sphere's tests, each task a commit with its tests.
depends_on: [archive/measures/implementation]
---

Each task is one commit, and the workspace compiles and passes its tests after each. The design of
every item is in [the implementation companion](implementation.md); the tests named are those of
its [Tests](implementation.md#tests), each in exactly one task, run at both precisions through the
test macro where the companion says `(target)`.

## The declaration (`skoro-core`, `embedded` and `error`)

- [ ] `Error::NoVolume { space, quantity }`; `Volume` with `global` and `within`, and `Domain`;
      the `volume` field of `Embedded`, which derives `Clone`; `Embedded::with_volume`, with its
      refusals, and `Embedded::volume`; `declare_flat`'s global zero volume,
      `skoro.flat.log_volume`; the documentation that a name and its parameters name a declaration
      whole, its volume included; the re-exports from `skoro-core` and `skoro`. Tests:
      `with_volume_refuses_a_wrong_signature`,
      `with_volume_refuses_a_second_volume_and_a_flat_declaration`.

## The spaces (`skoro-core`, `retraction` and `ops`)

- [ ] `Map::LogVolume` and `Map::Domain` in the cache; `Space::log_volume`, `Space::domain` and
      `Space::has_measure` for every space, as the companion's table gives them, the product's
      domain the conjunction of its factors'; the module documentation's new columns; the handles'
      `Q::log_volume` and `QDyn::log_volume`. Tests: `euclidean_log_volume_is_zero`,
      `positive_log_volume_is_the_log_of_the_moved_point`,
      `product_log_volume_is_the_sum_of_its_factors`,
      `discrete_space_has_a_measure_and_no_log_volume`, `flat_declaration_has_a_global_zero_volume`,
      `declaration_without_volume_has_no_measure`,
      `product_domain_is_the_conjunction_of_its_factors`.

## The refusals (`skoro-core`, `term` and `retire`)

- [ ] `NoVolume` from `Draft::add_term` for a scored quantity without a measure, from
      `Draft::add_proposal` for a proposed one, and from `Writer::retire` for an integrated one,
      each where the companion places it. Tests: `score_refuses_a_quantity_without_a_measure`,
      `factor_over_a_quantity_without_a_measure_is_allowed`,
      `propose_refuses_a_quantity_without_a_measure`,
      `generation_refuses_a_free_quantity_without_a_measure`,
      `retire_refuses_to_integrate_a_quantity_without_a_measure`.

## The volume check (`skoro-core`, `testing`, and the sphere's tests in `skoro`)

- [ ] `testing::check_volume`, with the log-determinant beside `positive_definite`; the
      acceptance test's sphere declares its log-volume
      [](eq:archive/measures/math/sphere-volume), global, in `tests/common/sphere.rs`. Tests:
      `sphere_passes_check_volume`, `sphere_log_volume_in_storage`.
- [ ] The guarded exponential sphere of `tests/sphere.rs` declares its log-volume, guarded below
      its cutoff, and its domain within $\lVert \tangent \rVert \le \pi$. Tests:
      `exponential_sphere_passes_check_volume`, `exponential_domain_ends_at_the_antipode`,
      `sphere_charts_have_the_mass_of_the_sphere`, `check_volume_reports_a_wrong_declaration`.

## Densities and products (`skoro`)

- [ ] The density pushed through the sphere's chart, and the log-volume of the acceptance test's
      product; `Volume` and `Domain` named in the `public_interface` test. Tests:
      `pushed_density_integrates_to_one_on_the_sphere`,
      `product_log_volume_agrees_with_its_factors`.

## Done when

Every box is ticked, and these pass from the root:

```sh
bin/wf check
cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cd site && pnpm check && pnpm lint && pnpm test && pnpm build
```

No task changes `skoro-tensor`, a primitive's kernel or a fused operation, so the local run with
`--features wgpu` is not required before pushing; the tests run through the test macro run on
wgpu wherever an adapter and the feature are present.
