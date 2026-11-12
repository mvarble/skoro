---
summary: The core's types stored up to a group and their checks, the linearizer's refusal of a point that is not finite, skoro-spatial's Cayley chart and the maps of the groups, and the g2o reader, its tests and its example, built in that order, each task a commit with its tests.
depends_on: [archive/quaternion-signs/implementation]
---

Each task is one commit, and the workspace compiles and passes its tests after each. The design of
every item is in [the implementation companion](implementation.md); the tests named are those of
its [Tests](implementation.md#tests), each in exactly one task, run at both precisions through
`precision_tests!` wherever the companion compares a pass. After each task that touches
skoro-spatial, every file a site page quotes by line range stays at least as long as its ranges
reach, so that `pnpm build` passes until the publish stage rewrites the quotes.

## The core (`skoro-core`)

- [x] `Linear`, the signed permutations of a type's storage, with `new`, `signs`, `storage`,
      `apply` and `then`; `Embedded::stored_up_to`, `identification` and `group`, with the
      declaration's new field, and its refusals; both re-exported from `skoro_core`. Tests:
      `stored_up_to_refuses_what_it_cannot_accept`, `group_closes_the_generators`.
- [x] The group's conditions in `check_embedded` (freedom, the points' images on the manifold,
      equivariance of R and P, invariance of Q) and in `check_volume` (invariance of Λ), each failure
      naming the generator and the condition; `testing::close_up_to`; and the test declaration of the
      lines through the origin of ℝ², with its variant whose inverse reads the representative. Tests:
      `the_checks_hold_on_the_lines`, `the_checks_name_a_broken_condition`,
      `close_up_to_compares_orbits`.

## The refusal (`skoro-linearize`)

- [x] `Error::NotFinite { term, row }` and its display; the scan of each factor's residual and
      Jacobian in `Linearizer::at` and `at_in_chart`, returning the first factor and row in the
      factors' order. Tests: `a_non_finite_residual_is_refused`.

## The groups (`skoro-spatial`)

- [x] The Cayley chart: `cayley` and `cayley_inverse` in `algebra.rs` in place of `delta` and
      `delta_inverse`; the declarations' retraction, inverse and log-volume, stored up to the
      quaternion's sign; the documentation of the module, the markers and the inverse's cut at the
      half-turns. The examples `pose_graph` and `sliding_window` still pass their checks. Tests:
      `chart_is_cayley` (in place of `chart_is_stereographic`), `volume_has_the_stated_mass`,
      `types_pass_the_core_checks`, `normal_density_on_rotations`, `normal_draws_on_rotations`,
      `the_term_is_the_same_at_either_sign`.
- [x] The maps of the groups: `log`, `axis_angle` and `slerp` of the rotations, the poses' `log`,
      `interpolate` and `interpolate_parts`, and the extended poses' `log` and `interpolate`, all of
      the shorter turn; every `*_shorter` method and composite removed, and the `shorter_turn` flags
      of `log_stored`, `shorter` and `slerp_stored` with them; `aligned_with` documented as the
      identity on SO(3); `host::log` of the shorter turn and `host::canonical`; the series check's
      list and `composites_pass_check_operation` renamed to match. Tests: `exp_and_log`,
      `interpolations`, `motion_exp_and_log`, `maps_of_the_group_ignore_the_sign`,
      `host_maps_agree_with_the_graph`.

## Outside data (`skoro`)

- [x] `examples/common/g2o.rs`, self-contained: `Graph`, `Edge`, `read`, the conversion of each
      edge's information to skoro's precision, and `one_turn_loop`, its vertices in the signs that
      composing the odometry gives; and `tests/signs.rs`, including it by path, with the model of a
      pose graph read from it. Tests: `g2o_information_is_converted`.
- [ ] The estimates on outside signs, in `tests/signs.rs`. Tests: `a_one_turn_loop_closes`,
      `the_estimate_ignores_every_sign`, `a_half_turn_is_non_finite`.
- [ ] The example `examples/g2o.rs`: a file's path as its argument, or `one_turn_loop(100, 0.01,
      SEED)` without one, estimated by Levenberg–Marquardt with its convergence checked; and
      `cargo run --release -p skoro --example g2o` added to the checks of `AGENTS.md` and
      `.github/workflows/ci.yml`. Tests: the example runs.

## Done when

- `bin/wf check` passes.
- `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` are clean.
- `cargo test --workspace` passes, every test above among it, at both precisions where it runs
  through `precision_tests!`.
- `cargo run --release -p skoro --example pose_graph`, `--example sliding_window` and
  `--example g2o` run and pass their checks.
- `node --test 'tools/*/*.test.ts' && site/node_modules/.bin/tsc -p tools` pass, and in `site/`,
  `pnpm check && pnpm lint && pnpm test && pnpm build` and `SITE_CONTENT=test/content pnpm build`
  pass, the pages' quotes of the changed files still within their files.
- No change touches `skoro-tensor`, a primitive's kernel or a fused operation, so the wgpu run
  is not required; every new map is a composite.
