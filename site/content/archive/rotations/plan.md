---
summary: The crate skoro-spatial built in order, from its storage arithmetic and the three declarations through the maps of the rotations, the Euler angles and the reading of matrices, to the maps of the rigid motions, the checks over every composite and the normal on these types, each task a commit with its tests.
depends_on: [archive/rotations/implementation]
---

Each task is one commit, and the workspace compiles and passes its tests after each. The design of
every item is in [the implementation companion](implementation.md); the tests named are those of
its [Tests](implementation.md#tests), each in exactly one task, run at both precisions of the Flex
device through `precision_tests!` wherever the companion compares a pass, at the rows its Tests
section prescribes.

## The crate and its types (`skoro-spatial`)

- [x] The crate `crates/skoro-spatial`, a member of the workspace and of `workspace.dependencies`,
      depending on `skoro-core` and `skoro-tensor` alone, with `skoro-normal` as a development
      dependency and the workspace's lints; `lib.rs` and its re-exports; `algebra.rs` with the
      storage arithmetic of the companion, `both` among it. Tests: none of the companion's; the
      crate builds and lints clean.
- [x] `types.rs`: the declarations `skoro.spatial.rotation`, `skoro.spatial.pose` and
      `skoro.spatial.extended_pose` with their retractions normalizing the base point, inverses,
      projections and global volumes, each in a `LazyLock` with its dead branch `unreachable!`; the
      markers `Rotation`, `Pose` and `ExtendedPose`; their host values, normalized and refused when
      zero or non-finite; and `register`. Tests: `types_are_declared`, `types_pass_the_core_checks`,
      `chart_is_stereographic`, `volume_has_the_stated_mass`.

## The maps of the rotations

- [x] `rotation.rs`, the group: `QMul` for `Rotation` on itself, `Vector<3>`, `VectorDyn` and
      `Matrix<3, N>`; `RotationMaps` with `inverse`, `between`, `unrotate`, `matrix` and `adjoint`;
      `rotation::identity`; and in `so3.rs`, `hat` and `vee`. Tests: `group_laws`,
      `actions_are_the_matrix`, `conjugation_is_exact`, `hat_and_vee`.
- [ ] `series.rs`: the coefficients `a`, `b`, `A`, `B`, `C`, `D`, `E` and `log_ratio`, with the
      cutoffs and series lengths of the companion's table and both guards; `rotation::exp`,
      `RotationMaps::log` and `log_shorter`; and `so3::jl`, `jr`, `jl_inv` and `jr_inv`. Tests:
      `exp_and_log`.
- [ ] The constructions: `rotation::elementary`, `about` and `from_two_vectors`,
      `RotationMaps::axis_angle` with its guarded axis, and `aligned_with`; in `host.rs`, `align`.
      Tests: `constructions`, `alignment`.
- [ ] `euler.rs`: `Axis`, `Sequence`, `Frame`, `Euler` with `YAW_PITCH_ROLL` and `axes`; the
      composites `skoro.spatial.from_euler` and `skoro.spatial.euler`, cached by intrinsic
      sequence, with the Tait–Bryan rearrangement, `LOCK` and its guards, and the extrinsic
      conventions by reversal; `rotation::from_euler` and `RotationMaps::euler`; `host::from_euler`
      and `host::euler`. Tests: `euler_round_trips`.
- [ ] `rotation::from_matrix` by the tournament over the four arguments, projecting the one chosen;
      the rest of `host.rs`: `compose`, `matrix`, `from_matrix`, `exp` and `log`. Tests:
      `from_matrix_branches`.

## The maps of the rigid motions

- [ ] `pose.rs` and `extended.rs`, the groups: `QMul` for `Pose` on itself and on `Vector<3>`, and
      for `ExtendedPose` on itself; `PoseMaps` and `ExtendedPoseMaps` with `inverse`, `between`,
      `rotate`, `untransform`, the parts, `matrix` and `adjoint`; `as_product`; `pose::identity`,
      `from_parts` and `from_product`; `extended_pose::identity` and `from_parts`. Tests:
      `motion_group_laws`, `parts_and_products`.
- [ ] The motions' exponentials and logarithms, `pose::exp`, `extended_pose::exp`, `log` and
      `log_shorter` on both; the composite `skoro.spatial.motion_q`; and the Jacobians and their
      inverses, `pose::jl`, `jr`, `jl_inv`, `jr_inv` and the extended pose's. Tests:
      `motion_exp_and_log`, `adjoints_conjugate`, `jacobians_are_the_passes`,
      `second_order_agrees_on_rotations`, `series_meet_closed_forms`.
- [ ] The interpolations and the angle: `RotationMaps::slerp`, `slerp_shorter` and `angle_to`;
      `PoseMaps::interpolate`, `interpolate_shorter`, `interpolate_parts` and
      `interpolate_parts_shorter`; `ExtendedPoseMaps::interpolate` and `interpolate_shorter`.
      Tests: `interpolations`.

## The checks over every composite

- [ ] `testing::check_operation` over every composite of the crate at the fixed and random rows,
      the discrete maps at rows away from their jumps; and the jumps themselves, the exact locks of
      the Euler angles built in storage among them. Tests: `composites_pass_check_operation`,
      `discretizations_jump_cleanly`.

## The normal on these types

- [ ] With `skoro-normal`, the normal scored on `Rotation` and `Pose`: its density through the
      chart, its term against the Gaussian of the chart coordinate, and its draws through $\pi$.
      Tests: `normal_density_on_rotations`, `normal_density_on_poses`,
      `normal_draws_on_rotations`.

## Done when

Every box is ticked, and

```sh
bin/wf check
cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cd site && pnpm check && pnpm lint && pnpm test && pnpm build
```

pass. The crate writes no kernel, so the wgpu tests are not required of it; the core's acceptance
test is untouched and passes as before.
