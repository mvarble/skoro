---
summary: The core built in order, from the tensor layer through spaces, operations, building, compiling and passes, terms and queries, embedded types, generation and retirement, each task a commit with its tests.
depends_on: [archive/core/implementation]
---

Each task is one commit, and the workspace compiles and passes its tests after each. The design of
every item is in [the implementation companion](implementation.svx); the tests named are those of
its [Tests](implementation.svx#tests), each in exactly one task, run at both precisions through
the test macro unless the companion says otherwise.

## The workspace

- [x] Create the workspace at the root (edition 2024, `rust-version = "1.95"`) with the three
      crates `skoro-tensor`, `skoro-core` and `skoro`, their dependencies, and the lints that deny
      `unwrap_used`, `expect_used` and `panic` outside tests. `skoro` re-exports the other two, and
      grows its re-exports as each task adds public items. Add the cargo commands to the checks in
      `AGENTS.md`.

## The tensor layer (`skoro-tensor`)

- [x] `Precision`, `Target` (`new`, `host`), the device check on `burn-dispatch`'s
      `DispatchDevice` that accepts the Flex device and cubecl's wgpu devices alone, and
      `TensorError`; the dependency on `burn-dispatch`, and burn's `ndarray` feature for the tensor
      layer's tests; the `wgpu` feature and the test helper that finds an adapter; the local wgpu
      run before pushing a change to the tensor layer, a kernel or a fused operation, added to the
      checks in `AGENTS.md`. Tests: `target_refuses_an_autodiff_device`,
      `target_refuses_an_unsupported_precision`, `target_refuses_an_unsupported_backend`,
      `wgpu_tests_fail_without_an_adapter`.
- [x] `Points`, `Frame`, `Integers` and `Value`, created only through a target; `into_burn` and
      `from_burn`; `to_host`; `HostValue` and `HostData`, with shared and per-row loading; the
      re-export of burn's tensor module. Tests: `two_precisions_share_a_device`,
      `arithmetic_keeps_its_precision`, `integers_to_points_takes_the_target_precision`,
      `comparison_takes_the_target_integers`, `from_burn_refuses_a_foreign_tensor`,
      `host_round_trip_is_exact_in_double`, `shared_value_expands_to_the_batch`,
      `per_row_value_pins_the_batch`.

## Identities and keys (`skoro-core`)

- [x] `NodeId` from the process-wide counter; `Segment` and `Address`, their parsing, display and
      order; `Key` with `from_seed`, `fold`, `at`, `split` and `uniforms` under their domain tags.
      Tests: `name_and_integer_segments_differ`, `segments_fold_one_at_a_time`,
      `fold_matches_the_reference_vectors`, `split_keys_are_distinct`,
      `uniforms_lie_in_the_open_interval`, `uniforms_have_uniform_moments`,
      `address_parses_digits_as_indices`, `quoted_names_display_distinctly`.
- [x] `Error` and `Failure`, with the variants of the tasks so far; later tasks add the variants
      they raise.

## Spaces and operations

- [x] `Space` without its embedded variant, with `storage`, `dimension`, `cardinality` and
      `blocks`; `Block`; the markers, `Typed` and `Declares`; `Q<T>` and `QDyn` with the two
      reserved failed identifiers, and `widen`. Tests: `product_refuses_a_discrete_factor`.
- [x] `Operation`, `Lowered`, `Lowering` (the target and its cache), `Param`, and the kernel
      traits `Forward`, `FirstOrder`, `SecondOrder`, `Pullback`, `SecondPullback`, with `Kernel` as
      the enumeration of orders and its accessors by upcasting.
- [x] `testing::Tolerance`, the test macro over both precisions, and `testing::check_operation`
      on a lowered kernel, moving inputs of the core's spaces by their retractions as the table of
      retractions gives them, on the host values. The Euclidean primitives (`add`, `sub`, `neg`, `mul`, `scale`,
      `matmul`, `sum`, `exp`, `log`, `slice`, `concatenate` with its reshape, `transpose`) with
      kernels of order two. Tests: `add`, `sub`, `neg`, `mul`, `scale`, `matmul_matrix`,
      `matmul_vector`, `sum`, `exp`, `log`, `slice`, `concatenate`, `reshape`, `transpose`.
- [x] The trigonometric primitives `sin`, `cos` and `atan2`. Tests: `sin`, `cos`, `atan2`,
      `atan2_is_checked_off_its_branch_cut`.
- [x] `select`, and the positive primitives `positive.as_real`, `.exp`, `.log`, `.mul`, `.recip`.
      Tests: `select`, `select_routes_the_cotangent_to_the_chosen_value`, `positive_as_real`,
      `positive_exp`, `positive_log`, `positive_mul`, `positive_recip`.
- [ ] The discrete operations (integer arithmetic and comparisons), the embedding `to_real` and
      the discretization `is_nonnegative`, of order zero. Tests: `integer_arithmetic_is_exact`,
      `comparisons_yield_booleans`.

## Building

- [ ] The `Model` with its persistent maps and nodes, `Snapshot`, `Writer` with `build`,
      `snapshot` and `reader`, the thread-local stack of frames and its guard, the four steps of
      every building function with `#[track_caller]`, and `scope::quantity`, `quantity_in`,
      `constant`, `namespace`, `lookup` and `apply`, with `IntoHostValue`, and `narrow`, which
      checks the node's space in the open scope. Constants and nodes fixed in the model fold when
      added, lowered for `Target::host()`. Tests: `widening_is_free_and_narrowing_checks_the_space`,
      `operation_outside_a_scope_fails`, `failure_poisons_what_is_built_on_it`,
      `scope_reports_the_first_failure_and_where`, `shape_mismatch_fails_when_the_node_is_added`,
      `failed_scope_publishes_nothing`, `panic_in_a_scope_discards_the_draft`,
      `readers_see_before_or_after`, `scopes_nest_and_build_into_the_innermost`,
      `namespaces_prefix_addresses`, `lookup_is_relative_to_the_open_namespaces`,
      `address_cannot_be_taken_twice`, `constants_fold_when_added`.
- [ ] `Snapshot::fork`, branches with cells of their own, and handles valid on every branch that
      shares them. Tests: `handle_is_valid_on_every_branch_that_shares_it`,
      `handle_from_another_model_fails`, `branches_do_not_see_each_other`,
      `fork_copies_nothing_it_does_not_change`, `dropping_the_last_snapshot_frees_retired_nodes`.
- [ ] The operator traits `QAdd`, `QSub`, `QMul`, `QDiv`, `QNeg`, `QScale` and `QShift`, the
      generic `std::ops` impls on `Q`, the core markers' impls (concrete markers only), the
      operators of `QDyn` over the core's spaces, and the methods and `scope` functions of the
      operations table but `retract`, `local`, `include` and `project`, which come with the
      retractions and embedded types. Tests: `operators_build_the_nodes_of_their_markers`,
      `dyn_operands_of_different_spaces_fail`.
- [ ] `Composite::define` with its definition frame, `Composite::call`, and the `Registry` with
      the core's operations, `register_operation` and `operation` by name. Tests:
      `every_core_operation_round_trips_through_its_name`, `unknown_operation_is_refused_by_name`.

## Compiling and passes

- [ ] `Observations`, `Query::of` and `given`, and `Snapshot::compile`: reach, lower with
      composites expanded under the chain of `Lowering` (with `Lowering::compile` and
      `Composite::compile`), classify, fold, prune and finish, into a `Program` of steps over
      slots. `Program`'s `forward`, `tangent`, `cotangent`, `hessian`, `gradient` and
      `hessian_products`, with `Values`, `Frames` and `Second`, and `CompositeCycle`. The model's
      folding of a node whose operation lowers to a composite, compiled for `Target::host()`.
      Tests:
      `tangent_pass_matches_finite_differences`, `cotangent_pass_is_its_transpose`,
      `hessian_pass_matches_finite_differences_of_the_gradient`,
      `hessian_is_symmetric_on_flat_spaces`, `unseeded_root_still_receives_its_cotangents`,
      `fan_out_accumulates_cotangents`, `rows_do_not_interact`,
      `values_are_freed_after_their_last_reader`, `expanding_a_composite_changes_no_pass`,
      `composites_nest`, `repeated_composite_off_the_chain_is_not_a_cycle`,
      `cycle_through_a_fused_operation_is_refused`,
      `cycle_through_a_fused_fallback_defined_afresh_is_refused`.
- [ ] Admissibility: `Pass`, `Program::admits`, the sweeps for "reached from" and "reaches", the
      order each step requires, and `OrderTooLow` naming the step, its composites and its node.
      Tests: `first_order_step_off_the_pass_admits_a_hessian`,
      `first_order_step_without_velocity_admits_a_hessian`,
      `first_order_step_with_velocity_refuses_a_hessian`,
      `zero_order_step_on_the_pass_refuses_a_gradient`, `refusal_comes_before_any_kernel_runs`,
      `order_is_read_off_the_kernel`, `embedding_and_discretization_have_order_zero`.
- [ ] Fused operations: lowering to their own kernel on the targets it covers and to their
      composite elsewhere, and `testing::check_agreement`. Tests:
      `fused_operation_agrees_with_its_composite`, `fused_operation_falls_back_to_its_composite`.
- [ ] The retractions of the core's spaces as composites cached per space, `Space::retraction`
      and `inverse_retraction`, and `retract` and `local` on handles. Tests:
      `retraction_round_trips`, `positive_mirror_identity_is_exact`,
      `positive_chart_hessian_equals_the_pullback_hessian`.

## Terms and queries

- [ ] `Term`, `scope::score` and `scope::factor` with `TermHandle`, `involves` read off the
      compilation graph, `scoring` and `joined`, `scope::propose`, roles and bindings with
      `Snapshot::bound`, and the reading surface, `blanket` among it. Tests:
      `term_neighbours_are_read_off_the_compilation_graph`, `quantity_is_scored_at_most_once`,
      `term_cannot_be_conditioned_on_a_later_quantity`, `unscored_quantity_is_a_parameter`,
      `creation_order_is_ancestral`, `blanket_of_the_first_tick`,
      `binding_lookup_checks_multiplicity`, `binding_must_stay_within_its_term`.
- [ ] `Query::log_density` and `log_density_of`, observed quantities folded per program, selection
      by a fixed index, pruning, the program cache keyed by targets, observation set and target,
      bounded and empty in every new model, and `Program::free_discrete` with `FreeDiscrete`.
      Tests: `query_density_is_the_sum_of_terms`, `log_density_of_sums_the_named_terms`,
      `log_density_of_ignores_their_order`, `blanket_density_is_the_conditional_up_to_a_constant`,
      `log_density_of_refuses_a_non_term`, `roots_are_values_not_draws`,
      `folded_and_unfolded_programs_agree`, `observed_values_fold_once_per_program`,
      `free_values_are_not_folded`, `selection_by_an_observed_index_drops_the_other_branches`,
      `selection_by_a_per_row_index_stays_a_step`, `same_observations_reuse_the_program`,
      `program_cache_is_bounded`, `new_snapshot_starts_with_an_empty_cache`,
      `programs_are_cached_per_query_and_target`, `one_snapshot_serves_two_precisions`,
      `free_discrete_quantities_are_reported`, `gradient_at_a_free_discrete_value_is_conditional`.
- [ ] `Layout` with `product`, `dimension`, `blocks`, `retract` and `flatten`. Tests:
      `layout_is_the_product_of_the_free_spaces`, `flat_step_is_the_product_retraction`.

## Embedded types

- [ ] `Embedded::declare` with a retraction and its inverse, `Space::Embedded` with its blocks,
      the boundary kernels `skoro.include` and `skoro.retract` over the compiled programs of $R$
      and $Q$ under the chain, their orders, the embedded retraction and inverse retraction,
      `include` on handles, the registry's embedded declarations, `testing::check_operation` moving
      inputs of an embedded type by its retraction, and `testing::check_embedded` without a
      projection. The
      test sphere in `crates/skoro/tests/common/sphere.rs`, and the integration test that keeps
      `skoro`'s tests on its public interface. Tests: `storage_and_dimension_of_each_space`,
      `cycle_through_a_boundary_kernel_is_refused`,
      `sphere_basis_is_the_derivative_of_its_retraction`, `sphere_retraction_reaches_the_far_side`,
      `removable_singularity_is_reported`.
- [ ] The optional projection: `projection` in `Embedded::declare`, `has_projection`, the kernel
      `skoro.project` over the programs of $P$ and $Q$, `scope::project` with `NoProjection`, and
      `check_embedded` covering it. The sphere declares its normalizing projection. Tests:
      `sphere_passes_check_embedded`, `sphere_projection_agrees_with_its_retraction`,
      `projection_returns_drift_to_the_sphere`, `midpoint_is_a_map_into_the_sphere`,
      `project_without_a_declared_projection_fails`, `boundary_orders_follow_the_composites`.
- [ ] `Embedded::declare_flat`, recognised by the boundary kernels, which then run no inner
      passes; the `Complex` marker in `crates/skoro/tests/common/complex.rs`, with its product
      through `QMul`. Tests: `feature_marker_defines_its_product`, `feature_marker_is_not_a_vector`,
      `flat_declaration_skips_inner_passes`.
- [ ] The acceptance test of the mathematics, on the product of two Euclidean factors, a real
      line, a positive factor and the sphere. Tests: `product_retraction_agrees_with_its_factors`,
      `chart_derivatives_agree_with_blocks_and_finite_differences`,
      `chart_hessian_differs_from_the_pullback_hessian_on_the_sphere_alone`.

## Generation and retirement

- [ ] `Sample`, `Sampler` and `Proposal` with the rules on what each may read
      (`SamplerOutsideTerm`), `Snapshot::generate` with draw steps in increasing identifier, keys
      by address, `CannotGenerate`, and the weight summed as the companion says; the private normal
      in `crates/skoro/tests/common/normal.rs`. Tests:
      `generation_refuses_a_quantity_it_cannot_draw`, `exact_samplers_cancel_in_the_weight`,
      `inexact_sampler_contributes_its_density`, `parameter_is_drawn_from_its_proposal`,
      `sampler_must_stay_within_its_term`, `weights_estimate_the_evidence`,
      `draw_depends_on_key_and_address_alone`, `rows_draw_independently`,
      `draws_agree_across_precisions`.
- [ ] `Writer::retire`, `Retirement`, `Plan` with `unproposed`, `Replacement`, and `replace` with
      its checks, its removal and its one published snapshot. Tests:
      `retirement_reads_the_blanket_and_removes_what_it_names`,
      `replacement_outside_the_blanket_is_refused_and_changes_nothing`,
      `kalman_replacement_matches_quadrature`,
      `blanket_quantity_becomes_a_parameter_drawn_from_its_proposal`,
      `retired_handle_fails_on_the_new_model_and_works_on_the_old`,
      `proposal_reading_a_retired_quantity_is_removed_and_reported`,
      `nothing_left_refers_to_a_removed_node`.

## Finishing

- [ ] Write `REVISIT.md` at the root of the repository, listing what the companion names: what
      the archive had and the core does not bring back, compiling between cuts among it, and what
      the charter names that no entry builds yet.

## Done when

- `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings` and
  `cargo test --workspace` pass, as CI runs them.
- `cargo test --workspace --features wgpu` has passed on a machine with an adapter, as the checks
  in `AGENTS.md` require, and the last commit says so.
- Every test named in the companion's [Tests](implementation.svx#tests) exists under its name and
  passes at both precisions.
- The site builds (`pnpm build`, and the fixture), and `bin/wf check` passes.

## Revisions

**2026-10-15.** Reopened at the sixth task. `narrow` checks a node's space inside the open scope,
which the building task adds, so neither it nor `widening_is_free_and_narrowing_checks_the_space`
can be built with the spaces. It moves to the building task, with its test. Reading the plan
again for the same fault found three more, each a task reaching for what a later one adds, and
each moved to where it can be built:

- The operator task built every method of the operations table, but `retract` and `local` need
  the retractions of the core's spaces, and `include` and `project` the embedded types. They come
  with those tasks; `project` already did.
- The model folds a node whose inputs are all fixed when it is added, and a composite call is run
  by compiling it, which the compiling task adds after `Composite::call`. Folding such a node
  moves to the compiling task.
- `check_operation` moves its inputs by their spaces' retractions, which the companion builds as
  composites, run only once there is a compiler. For the core's spaces each retraction is a
  formula of the table of retractions, and the check computes it on the host values, which lets
  the primitives be checked as they are written; inputs of an embedded type are moved by its
  retraction once the embedded types exist. A retraction computed apart from its composite also
  makes the check independent of the composites it will later be used to check.
