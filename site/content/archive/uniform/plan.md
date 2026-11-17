---
summary: The core's declared mass and its check, the masses of the rotations and the test spheres, the new crate skoro-uniform with its term and its sampler, the uniform's place in a linearization, and the normal's documentation and test of a large scale, built in that order, each task a commit with its tests.
depends_on: [archive/uniform/implementation]
---

Each task is one commit, and the workspace compiles and passes its tests after each. The design of
every item is in [the implementation companion](implementation.md); the tests named are those of
its [Tests](implementation.md#tests), each in exactly one task. The workspace asks for Rust 1.95,
so the build begins by confirming that the toolchain in use is at least that. After each task
that touches a file a site page quotes by line range, the file stays at least as long as the
ranges reach, so that `pnpm build` passes until the publish stage rewrites the quotes.

## The core (`skoro-core`)

- [ ] `Mass` and `UniformDraw`, `Embedded::with_mass` and `Embedded::mass` with the declaration's
      new field and its refusals, re-exported from `skoro_core`; `Error::NoMass` and its message;
      `Space::log_mass` and `Space::draw_uniform`, a product by its factors. Tests:
      `log_mass_of_the_spaces`, `with_mass_is_refused_without_a_volume_or_twice`,
      `a_product_has_the_mass_of_its_factors`.
- [ ] `testing::check_mass`; the test spheres `SPHERE` and `EXPONENTIAL` declare `log 4π` and
      Archimedes' draw; the spheres' quadrature compared with the declared masses; and the
      additions recorded in the public interface. Tests: `sphere_masses_check`,
      `sphere_charts_have_the_mass_of_the_sphere` (amended), `public_interface`.

## The rotations (`skoro-spatial`)

- [ ] `uniform_rotation` and `LOG_MASS`, and `Rotation`'s declaration ending in `with_mass`;
      `Pose` and `ExtendedPose` unchanged. Tests: `volume_has_the_stated_mass` (amended),
      `rotation_mass_checks`, `uniform_rotation_is_shoemakes_map`,
      `uniform_rotations_have_the_haar_angle`, `poses_have_no_mass`.

## The distribution (`skoro-uniform`)

- [ ] The crate, a member of the workspace and among its dependencies, with its description and
      crate documentation; `Uniform` with `on`, `in_space`, `space`, `log_mass` and `score`, the
      term a call of the composite `skoro.uniform.log_density`, its refusals, and the empty
      residual bound; `Distribution for Uniform`. The sampler is a `UniformSample` whose draw
      handles the continuous spaces. Tests: `term_is_minus_the_log_mass` (the categorical cases
      included, which need no draw), `derivatives_are_zero`,
      `a_space_of_infinite_mass_is_refused`, `a_quantity_of_another_space_is_refused`,
      `binds_an_empty_residual_and_nothing_else`, `draws_are_functions_of_key_and_address`.
- [ ] `category` and the categorical branch of `UniformSample::draw`. Tests:
      `category_never_returns_n`, `categorical_draws_are_floor_n_u`,
      `categorical_counts_are_uniform`, `generation_weighs_the_likelihood_alone`.
- [ ] `skoro-uniform` a dev-dependency of `skoro`, and the uniform on a chart that is not
      global. Tests: `uniform_on_the_exponential_sphere`.

## In a linearization (`skoro-linearize`)

- [ ] `skoro-uniform` a dev-dependency, and no change to the linearizer. Tests:
      `a_uniform_prior_adds_no_rows`, `a_uniform_prior_leaves_the_estimate_to_the_data`,
      `a_quantity_under_a_uniform_alone_is_undetermined`.

## The normal (`skoro-normal`, `skoro-spatial`)

- [ ] The paragraph on a large scale on a curved type in the crate documentation of
      `skoro-normal` and the doc comment of `Normal`, naming `skoro_uniform::Uniform` as how
      "unknown" is said; no code of the normal changes. Tests:
      `broad_normals_turn_towards_the_half_turns`.

## Done when

- `bin/wf check` passes.
- `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` are clean.
- `cargo test --workspace` passes, every test above among it.
- `cargo run --release -p skoro --example pose_graph`, `--example sliding_window` and
  `--example g2o` run and pass their checks.
- `node --test 'tools/*/*.test.ts' && site/node_modules/.bin/tsc -p tools` pass, and in `site/`,
  `pnpm check && pnpm lint && pnpm test && pnpm build` and `SITE_CONTENT=test/content pnpm build`
  pass, the pages' quotes of the changed files still within their files.
- No change touches `skoro-tensor`, a primitive's kernel or a fused operation, so the wgpu run
  is not required: the one new operation is a composite of a constant, and every draw runs on
  the host.
