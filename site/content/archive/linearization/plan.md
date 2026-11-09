---
summary: The additions to the core and the normal, then `skoro-elimination` from its structure to its answers, then `skoro-linearize` from its refusals through the residual and Fisher assemblies to the replacement, each task with its tests.
depends_on: [archive/linearization/implementation]
---

## The core

- [ ] Add `Query::observations` and `Query::terms`. Tests: `query_gives_its_terms_and_observations`.
- [ ] Add `Observations::union`, keeping this set's value where both observe a quantity. Tests:
      none named for it; the replacement's tasks below exercise it.
- [ ] Make `roots` public as `Snapshot::reached_from`. Tests: `reached_from_is_what_a_term_involves`.
- [ ] Add `Composite::reads` and `Space::volume_reads_point`. Tests: `composites_read_their_inputs`,
      `volume_reads_point_by_space`.
- [ ] Add `scope::constant_in`. Tests: `constant_in_is_constant`.

## The normal

- [ ] Add the role `roles::TANGENT`, bound by `Normal::score` and not by `factor_at`, and extend
      `roles_are_bound` with it. Tests: `score_binds_the_tangent`.
- [ ] Add `Stacked::block_in`. Tests: `block_in_is_block`.

## `skoro-elimination`

- [ ] Add the crate to the workspace, with `faer` 0.24.4 among the workspace's dependencies:
      `Matrix`, `Structure`, `Rows`, `Problem` with `value`, `Damping` and `Problem::promised`,
      and `Error`. Tests: none of its own; the crate compiles and passes clippy.
- [ ] The schedule: COLAMD on the block pattern, `Schedule::{all, keeping, eliminating,
      in_order}`, the elimination tree and the junction tree of its cliques. Tests:
      `the_star_eliminates_its_leaves_first`, `the_map_has_three_cliques`.
- [ ] The numeric elimination, `Problem::eliminate`, with the pushes of the linear terms, the
      positive diagonal and the constants, and `Elimination::{minimizer, minimum}`. Tests:
      `triangular_root_of_the_example`, `the_ill_conditioned_example_keeps_its_digits`,
      `eliminating_one_block_is_the_lemma`, `two_scalars`,
      `a_linear_term_in_the_row_space_stays_there`.
- [ ] The rank test at `u^{3/4}` of the problem's roundoff. Tests: `an_undetermined_block_is_refused`.
- [ ] `Elimination::{log_integral, solve, log_determinant, marginal}` and `Marginal`. Tests:
      `orders_agree`, `the_integral_is_the_determinant`, `solve_is_the_inverse_root`,
      `marginal_whitening_is_lower_triangular`.
- [ ] Damping rows appended at each block's front. Tests:
      `damping_appends_rows_and_keeps_the_schedule`.
- [ ] `Elimination::reduced` and `Reduced`, with no decision of rank. Tests:
      `replacing_from_some_factors_is_the_split`, `reduced_rows_have_their_shape`.

## `skoro-linearize`

- [ ] Add the crate, with `Assembly`, `Error`, and `Linearizer::new` up to the factors and their
      refusals: the density program, the free discrete quantities, the residuals, the scales
      reached from free quantities, the structure. Tests: `a_free_discrete_quantity_is_refused`,
      `a_factor_without_a_residual_is_refused`,
      `a_free_scale_is_refused_by_the_residual_linearization`.
- [ ] The rows program, its seeds from `rows.free()` by colouring the factor structure, and the
      residual assembly at a point, `Linearizer::at`, without folding: rows, right-hand sides,
      the linear term from the log-volumes that are not folded, and `c_0`. Tests:
      `linear_gaussian_terms_are_exact`, `the_linear_term_is_zero_without_log_volumes`,
      `positive_log_volumes_are_a_linear_term`.
- [ ] `Linearized::eliminate`, `Solved::minimizer`, `Linearizer::schedule`, and the mapping of an
      undetermined block to its quantity. Tests: `the_step_is_gauss_newton`,
      `the_step_keeps_its_digits`, `batch_rows_share_the_schedule`,
      `damping_promises_the_decrease`, `an_undetermined_quantity_is_refused`.
- [ ] Folding: the fold condition, the targets bound under `TANGENT`, `MEAN` and
      `INFORMATION_ROOT`, and `ĉ_t` from the space's log-volume on the host. Tests:
      `the_linearization_agrees_with_the_query`, `folded_log_volumes_are_in_the_right_hand_sides`,
      `steps_converge_on_rotations`.
- [ ] `Linearizer::{at_in_chart, schedule_keeping}`, `Solved::{marginal, log_integral, draw}` and
      `QuantityMarginal`. Tests: `the_laplace_approximation_is_centred_in_the_chart`,
      `draws_have_the_marginal_covariance`, `marginals_name_their_curved_quantities`,
      `an_integral_without_a_volume_is_refused`.
- [ ] The Fisher assembly: the scale rows `J^θ_t` and `u_t`, `J^μ_t = W_t D s_t` for a free scale,
      `c_F`, and its refusals. Tests: `fisher_rows_are_a_root_of_the_expected_information`,
      `fisher_gradient_is_the_query_gradient`,
      `fisher_with_fixed_scales_is_the_residual_linearization`,
      `fisher_scoring_converges_to_the_estimate`, `fisher_marginals_are_the_inverse_information`,
      `fisher_shares_the_schedule`, `fisher_gives_no_integral`,
      `a_free_scale_without_a_root_is_refused_by_fisher`.
- [ ] `Replacing::new`, with its checks of the blanket and the union of the retirement's values
      with the caller's. Tests: `a_mismatched_blanket_is_refused`.
- [ ] `Replacing::at`, `LinearizedReplacement` and `build`, on the linear-Gaussian chains. Tests:
      `retiring_the_scalar_chain_is_the_kalman_filter`,
      `retiring_a_chain_one_state_at_a_time_is_the_kalman_filter`,
      `an_undetermined_retired_quantity_is_refused`.
- [ ] The replacement on nonlinear, positive, rotation and undetermined blankets. Tests:
      `a_nonlinear_chain_is_the_extended_kalman_filter`, `improper_replacements_accumulate`,
      `a_bearing_replacement_is_the_linearized_integral`, `the_positive_replacement_is_exact`,
      `rotations_and_vectors_leave_no_linear_term`, `a_batch_of_replacements_has_one_shape`.

## Done when

```sh
bin/wf check
cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cd site && pnpm check && pnpm lint && pnpm test && pnpm build
```

all pass. Every test above runs at both precisions where the implementation's `## Tests` says so.
No kernel of `skoro-tensor` or of a primitive changes, so no run with a wgpu adapter is needed; a
task that finds it must change one says so, and its commit is pushed only after
`cargo test --workspace --features wgpu` passes on a machine with an adapter.
