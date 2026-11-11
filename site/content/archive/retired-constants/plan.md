---
summary: Index the nodes a model holds, free what a retirement leaves needed by nothing, add pins, and check that the sliding window stops growing.
depends_on: [archive/retired-constants/implementation]
---

## The held index (`skoro-core`)

- [x] Add `held`, `pinned` and `spared` to `Model`, with `hold`, `release` and `keeps`, and clone
      them in `Clone for Model`. Add the helpers `named(term)` and `named_by(proposal)`, and hold
      and release through them in `TermHandle::bind`, `TermHandle::sampler` (releasing a replaced
      sampler), `Draft::add_proposal` (releasing a replaced proposal), the bindings of
      `Retirement::replace`, and `remove` (the terms of $\mathcal T_E$ and the proposals it
      drops). Tests: `held_index_matches_the_terms`.

## Freeing (`skoro-core`)

- [x] Record the nodes a draft inserts in `Draft::created`, through a `Draft::insert` that
      `mapped`, `constant` and `quantity` call. Add `Snapshot::nodes`.
- [ ] Make `remove` return its candidates, add `free`, and call it at the end of
      `Retirement::replace` with the candidates and `draft.created`, before publishing. Extend the
      documentation of `Retirement::replace`, `Writer::retire` and the `retire` module to say what
      is freed. Move `a` to the removed nodes in the existing test. Tests:
      `retirement_reads_the_blanket_and_removes_what_it_names`,
      `freed_nodes_are_what_nothing_needs`, `nothing_left_refers_to_a_removed_node`,
      `kalman_replacement_matches_quadrature`, `part_of_a_tick_matches_the_joint_normal`,
      `shared_constant_stays_while_a_term_reads_it`, `sampler_inputs_are_held`,
      `replacement_reusing_a_retired_constant_keeps_it`,
      `unused_values_of_the_replacement_scope_are_freed`,
      `steady_window_holds_a_constant_number_of_nodes`.

## Pins (`skoro-core`)

- [ ] Add `scope::pin` and `scope::unpin`, re-exported from `skoro::scope`, and
      `Snapshot::is_pinned`. Have `free` spare pinned nodes into `spared`, `Retirement::replace`
      release the spared nodes no longer pinned into the candidates, and `remove` drop the pins
      and spared entries of removed nodes. Say in `UnknownNode`'s documentation and message that a
      retirement may have freed the node and that pinning keeps it. Tests:
      `pinned_constant_survives_a_retirement_that_leaves_it_unused`,
      `unpinned_node_is_freed_at_the_next_retirement`,
      `pin_on_a_node_reached_from_a_retired_quantity_does_not_keep_it`,
      `pins_belong_to_their_branch`, `pin_refusals`.

## The example (`skoro`)

- [ ] Report the window's node count in `sliding_window`, and check that the largest count after
      tick $2W$ is at most 1.5 times the largest from tick $W + 1$ to $2W$.

## Done when

```sh
bin/wf check
cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo run --release -p skoro --example pose_graph
cargo run --release -p skoro --example sliding_window
cd site && pnpm build
```

all pass, `sliding_window` reports a node count that no longer grows with the ticks, and the
public-interface test (`crates/skoro/tests/public_interface.rs`) is updated if it lists the
functions of `skoro::scope` or the methods of `Snapshot`.
