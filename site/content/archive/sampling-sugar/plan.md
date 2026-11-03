---
summary: A trait for what scores a quantity, implemented by the normal, and `sample`, which declares a quantity and scores it by one in a single call.
depends_on: [archive/core/implementation, archive/normal/implementation]
---

A quantity drawn from a distribution is written in two statements, `let x = quantity("x", T);` and
`normal.score(x);`. This chore adds a function that writes both, and the trait it needs to score by
any distribution rather than by `Normal` alone. It builds nothing the two statements do not.

- **The trait.** `skoro-core` gains `Distribution`, with one method:

  ```rust
  pub trait Distribution {
      /// Adds the term that scores `x`, with its sampler and roles, in the open scope.
      #[track_caller]
      fn score(&self, x: QDyn) -> TermHandle;
  }

  impl<D: Distribution + ?Sized> Distribution for &D { /* forwards */ }
  ```

  It lives beside the core's `score` and `factor`, in `term`, and is re-exported at the crate's
  root. The term it adds carries whatever the distribution gives it: `Normal::score` already adds
  the sampler and binds the roles, so the trait asks for nothing more. The archive's
  `Distribution` also named the quantity's type, the sampler and the parents; here the space comes
  from the marker, and the sampler and parents from the term the distribution builds.
- **The sugar.** `scope` gains `sample` and `sample_in`, beside `quantity` and `quantity_in`:

  ```rust
  #[track_caller]
  pub fn sample<T: Declares>(
      address: impl Into<Address>,
      marker: T,
      distribution: impl Distribution,
  ) -> Q<T>;

  #[track_caller]
  pub fn sample_in(address: impl Into<Address>, space: Space, distribution: impl Distribution)
      -> QDyn;
  ```

  Each declares the quantity, scores it by the distribution, and returns the quantity. A
  distribution is taken by value or, through the blanket implementation, by reference. A refusal
  by either step is recorded at the line that called `sample`, as it would be at the line of the
  statement that failed.
- **The normal.** `Normal` implements `Distribution` by its own `score`, which stays: a caller that
  scores by a `Normal` it names does not import the trait.

`sample` gives back the quantity and not the term. A caller that wants the term, to bind a role of
its own or to look it up, writes the two statements, which `sample` leaves as they are. A macro was
the alternative to a function: a function suffices, and with `#[track_caller]` it records refusals
where a macro would. So was taking the space from the distribution, `sample("x", normal)`: the
marker is what types the handle, and a `Normal` does not know its marker.

## The core (`skoro-core`, `term` and `scope`)

- [ ] `Distribution`, its implementation for references, and its re-export; `scope::sample` and
      `scope::sample_in`; their documentation. Tests, in the core's term tests, with a
      distribution written in the test: `sample_declares_and_scores`, a sample under a namespace
      is a quantity at the namespaced address, of the marker's space, scored by the term the
      distribution adds, taken once by value and once by reference;
      `sample_in_declares_a_quantity_of_its_space`; `sample_records_a_refusal_at_its_line`, a
      distribution whose term is refused fails the build at the line of the `sample` call.

## The normal (`skoro-normal`, `normal`)

- [ ] `Distribution` for `Normal`; the documentation of `Normal::score`, which points to `sample`.
      Tests: `sample_scores_as_score_does`, a normal's sample binds the roles and adds the sampler
      that `quantity` and `Normal::score` do, and its log-density agrees at points;
      `sample_of_a_normal_refuses_at_its_line`, a normal sampled on a discrete space fails with
      `NotContinuous` at the line of the `sample` call.

## Done when

Every box is ticked, and these pass from the root:

```sh
bin/wf check
cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cd site && pnpm check && pnpm lint && pnpm test && pnpm build
```

No task changes `skoro-tensor`, a primitive's kernel or a fused operation, so the local run with
`--features wgpu` is not required before pushing.
