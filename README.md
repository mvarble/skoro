# skoro

Probabilistic modeling in Rust.

This repository started over on 2026-10-09. Everything written before then (the crates, the
documentation site, and their history) is kept on the
[`archive`](https://github.com/mvarble/skoro/tree/archive) branch. It is being brought back one
piece at a time, each piece with a written record of the mathematics and the design behind it.

## How it is developed

Every change is an entry in the archive (`site/content/archive/`), moved through stages in a
branch of its own: for a feature, the mathematics, an implementation companion, a plan, the build,
and last the documentation pages that say what now is. Each stage is approved by a person before
the next begins, and an approved document is frozen unless its stage is reopened.

- [`AGENTS.md`](AGENTS.md): how to work in this repository, for people and agents alike.
- [`.agents/reference/workflow.md`](.agents/reference/workflow.md): the procedure.
- `bin/wf`: the command line that keeps the record and enforces it (`bin/wf help`).
- [`site/`](site/README.md): the documentation (`cd site && pnpm install && pnpm dev`).

## License

[MIT](LICENSE).
