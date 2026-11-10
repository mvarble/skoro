---
summary: The order in which the examples are built, from the host maps of the rigid motions and the core's two additions, through the Levenberg–Marquardt method, to the two examples and their place in CI.
depends_on: [archive/examples/implementation]
---

## `skoro-spatial`

- [x] Add `host::pose_compose` and `host::pose_transform`, written with the module's `compose` and `matrix`, with the graph's sign convention for the quaternion. Tests: `host_pose_maps_agree_with_the_graph`.

## `skoro-core`

- [x] Add `Layout::local`, with the compiled inverse retraction cached per target in a `locals` field that `Layout::of` initializes empty and `Clone` leaves empty. Tests: `local_inverts_retract`, `local_refuses_as_retract_does`.
- [x] Add `Observations::observe_value`, which reads a `Points` to the host and observes it one value per row, deferring a discrete value or a storage of the wrong size to `ValueShape` when the set is used. Tests: `generated_values_are_observed_row_by_row`, `a_discrete_generated_value_is_refused`.

## `skoro-linearize`

- [x] Add `Linearizer::objective`, the negated density program's forward pass at a point, per row. Tests: `the_objective_is_the_negative_log_density`.
- [x] Add `minimize.rs` with `LevenbergMarquardt`, `Status`, `Iterate`, `RowOutcome`, `Minimized`, the private `marquardt`, `nielsen` and `select_rows`, and `minimize` itself: per-row damping by row-wise elimination, the trial batched and the decision per row, a row stopping on a step's promise before its acceptance, and the Fisher assembly refused. Re-export them at the crate's root. Tests: `the_marquardt_diagonal_is_the_column_norms`, `the_schedule_is_nielsens`, `the_fisher_assembly_is_refused`, `non_convergence_is_an_outcome`.
- [x] Test that the method reaches the mode, its stationarity checked by `decrement`, the undamped promise per row (added with `minimize.rs`), in the terms the stopping rule bounds: the gradient against what a decrement of $10\, \epsilon$ allows, the Gauss–Newton mode within $40\, \epsilon$ in the metric of $J$, the linear mode's excess of $f$ in closed form. Tests: `levenberg_marquardt_finds_the_mode_on_rotations`, `levenberg_marquardt_finds_the_linear_gaussian_mean`.
- [x] Test the method's behaviour across rows and rejections, on the crate's own models. Tests: `rows_are_minimized_independently`, `a_rejected_step_keeps_the_point`, `rows_at_rounding_converge`.

## `skoro` (the examples)

- [x] Declare the examples' dev-dependencies (`skoro-normal`, `skoro-spatial`, `skoro-linearize`, `skoro-elimination`, `faer`) and an `[[example]]` for each of `pose_graph` and `sliding_window` with `test = true`, with `examples/common/mod.rs` opening with `#![allow(dead_code)]`. Both examples compile as empty programs returning `Ok(())`.
- [x] Write `common/scenario.rs`: `Scenario`, `Scales`, `circuit`, `open_path` with the range of 8.5 m (refusing a landmark whose sightings are not an interval) and `first_sightings`. Tests: `the_scenarios_are_sighted`.
- [x] Write `common/model.rs`: `Tick` and `tick`, every term of the model as one `Normal::on(...).score(...)`, the odometry addressed under the tick it retires with.
- [x] Write `common/laplace.rs`: `normalized_errors`, `Calibration`, `calibration`, and the constants `CHI2_6_90`, `CHI2_3_90`, `SLOPE_6` and `SLOPE_3`. Tests: `the_chi_squared_constants`.
- [x] Write `common/report.rs`: the printing, and the checks as errors that name the figures.
- [x] Write `pose_graph.rs`: the circuit generated for $N = 100$ rows, the start on the host, `minimize`, the decrements, the bound on $\sigma_\phi^2$, the calibrations of the poses and the landmarks, and the report. Tests: `pose_graph`. Measure its run time in release and report it in the commit.
- [x] Write `common/live.rs` (`Live`, the window's handles, observed rows and landmark sightings) and `sliding_window.rs`: the window over the open path with $W = 5$, retirement and replacement at the window's mode, the batch on the same path, and the filtering calibration and the gap reported and not checked. Tests: `sliding_window`.

## CI and the checks

- [x] Add `cargo run --release -p skoro --example pose_graph` and `... --example sliding_window` to the Rust job of `.github/workflows/ci.yml` after `cargo test`, and the same two lines to the `## Checks` list of `AGENTS.md`.

## Done when

- `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` passes.
- `cargo test --workspace` passes, and so do both examples under `cargo run --release`.
- `cargo test --workspace --features wgpu` passes on a machine with an adapter, since the new forward passes and retractions run on it, and the commit that adds them says so.
- `bin/wf check` passes, and the site builds: `cd site && pnpm check && pnpm lint && pnpm test && pnpm build`.
