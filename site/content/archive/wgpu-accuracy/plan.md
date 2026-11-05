---
summary: Admit the wgpu adapter in the core's test support, then raise the rotations' cutoff to θ² = 1 and the test sphere's to r² = 0.1, with their series lengthened and checked against independent references.
depends_on: [archive/wgpu-accuracy/implementation]
---

## skoro-core: admission

- [x] Add `Admission` to `skoro_core::testing`, with its bounds `NEAR`, `FAR` and `FAR_RANGE`, its
      private `inputs()`, `measure` and `admit`, and the message format `{:.2e}` of the
      implementation. Tests: `admission_names_the_error_and_the_bound`,
      `the_host_is_admitted_in_single_precision`.
- [x] Admit the adapter in `testing::wgpu()`, caching the outcome in a `OnceLock`, and add the
      wgpu-gated test that builds the adapter's target directly, prints its `Admission` and asserts
      it is admitted. Run it on the adapter and record the `Admission` in the commit. Tests:
      `the_adapter_is_admitted`.

## skoro-spatial: the coefficients

- [x] Move `BERNOULLI` ($B_2 \ldots B_{24}$) from the tests into `series.rs`, and build `C_SERIES`
      from it with `table!`, still at its present seven terms. Add the test helper `zeta`. Give
      `summed` twenty terms, with `C`'s terms from `zeta`. Tests:
      `the_bernoulli_numbers_are_the_zeta_values`, `series_meet_closed_forms` (its `f32` and `f64`
      instances, against the new reference).
- [ ] Raise `CUTOFF` to `1.0` in `series.rs` and in `tests/common/mod.rs`, and lengthen the series
      to 8 terms for `a` and `b`, 9 for `A`, `B`, `D` and `E`, and 12 for `C`. Update the module's
      doc comment and the comment on `summed`. Move the points of
      `coefficients_are_their_closed_forms_away_from_the_identity` to $1.5$, $2$ and $9$, and add
      `2.5` to the `lengths` of `series_meet_closed_forms`. Run the crate's tests with `--features
      wgpu` on the adapter, and say so, with the adapter, in the commit. Tests:
      `series_are_truncated_below_rounding`,
      `coefficients_are_their_closed_forms_away_from_the_identity`,
      `series_meet_closed_forms::wgpu`, `composites_pass_check_operation::wgpu`.

## skoro: the test sphere

- [ ] Split `CUT` in `tests/common/sphere.rs` into `EXP_CUT = 0.1` and `LOG_CUT = 1e-8`. Add
      `COS_SERIES`, `SINC_SERIES` (eight terms each), `LOG_VOLUME_SERIES` (ten terms) and a local
      `horner`. Have the retraction's `Near::Selected` and `Near::Guarded` and the guarded
      log-volume select their series below `EXP_CUT`, and the logarithm use `LOG_CUT`. Leave
      `Near::Closed` as it is. Add a `zeta` helper to the sphere's tests. Run the crate's tests with
      `--features wgpu` on the adapter, and say so, with the adapter, in the commit. Tests:
      `exponential_series_are_truncated_below_rounding`, `exponential_series_meet_closed_forms`,
      `exponential_sphere_passes_check_volume::wgpu`.

## Done when

- `bin/wf check` passes.
- `cargo fmt --all --check` passes, and `cargo clippy --workspace --all-targets -- -D warnings`
  passes, with and without `--features wgpu`.
- `cargo test --workspace` passes.
- `cargo test --workspace --features wgpu` passes on the adapter, with every wgpu test admitted. The
  last commit states the adapter and the `Admission` measured.
- `cd site && pnpm build` passes.
