# Writing skoro's documents

How the documents in `site/content/` are written: where each kind goes, its frontmatter, links,
prose, and how mathematics, statements, citations and code are written. `site/README.md` covers
the same ground for a person editing the site; this file is what an agent follows.

## Where things go

```
site/content/
    landing.md                          the home page's prose
    <track>/<slug>/index.md             a page: start, guide, concepts, reference
    math/<slug>/index.md                a math note
    archive/<slug>/index.md             an entry, opened by `bin/wf start`
    archive/<slug>/math.md              its documents, one per stage that has one
    archive/<slug>/implementation.md
    archive/<slug>/plan.md
    archive/<slug>/writeup.md
    archive/<slug>/statements/<x>.md    statements its documents show
    references.bib                      the bibliography
```

The three kinds of writing differ in what they promise the reader.

- **Pages say what is.** The guide (top-down) starts from what a user writes; the concepts
  (bottom-up) build the library from its primitives, each page resting only on the ones before
  it. Pages are rewritten whenever the library changes, quote the code that exists, and only ever
  describe the current library. They change at an entry's `publish` stage.
- **Math notes say what is true**, independently of skoro: one idea to a note (a retraction, the
  Fisher information, a junction tree), at enough length to be worth reading alone. A note the
  site already has is linked, never rewritten in another document.
- **Archive documents discuss what may be.** They argue, weigh alternatives and leave room for
  doubt, as they stood when approved. Once approved they are frozen: they may drift from the code,
  and that is what they are for.

Before writing about an idea, search for it: `grep -rni 'retraction' site/content/`. If the site
explains it, link there. If a writeup needs an idea the site does not have, explain it in a
section of the writeup; promote it to a math note only when it stands alone and takes several
sections, or when a second document needs it.

## Frontmatter

```yaml
---
title: Retractions
summary: Maps from a tangent space back onto the manifold. # one plain sentence
depends_on: [tangent-space, archive/normal/math]
katex_macros:
    '\retr': '\operatorname{R}'
---
```

- `title` may contain inline math. An entry's documents default to their stage's name
  ("Mathematics", "Plan"), which is usually right.
- `summary` is shown in lists, the sidebar, previews and `llms.txt`. Without it, the first
  paragraph stands in, so either write one or make the first paragraph a good summary.
- `depends_on` lists what a reader must understand first: a bare slug is a math note, anything
  else a key (`concepts/graph`, `archive/normal/math`). These are the solid arrows of the map on
  the Mathematics page, so list real prerequisites, not everything mentioned.
- A page also needs `order`, its place in its track.
- An entry's `index.md` lists the `pages` it changed (written at `publish`) and the entries it
  `revises` (given to `wf start`).

## Links

- Write links the way the files sit on disk: `[retractions](../../math/retraction/)`,
  `[the plan](plan.md)`, `[the guide](../../guide/)`. The build rewrites them to the site's URLs
  and reports a link to a document that does not exist.
- A section's anchor is its heading in lowercase, with `$` removed and every run of characters
  other than `a`–`z` and digits turned into one hyphen: `## Fubini's theorem` is
  `#fubini-s-theorem`. Anchors are not checked, so work them out from the heading as written.
- Never put a link inside a heading.

## Prose

Write in complete sentences and paragraphs, in the register of a textbook, with tables and lists
only where they genuinely help. Never refer to the conversation that produced a document: it is
read later, by someone who was not there. Prefer a concrete example to an abstract definition
alone. Name things the way the code names them, in backticks; generics such as `Q<B, M>` must be
in backticks anyway, since `<` and `{` are Svelte syntax outside code and math.

## Mathematics

Inline math is `$x + y$`. A displayed equation opens with `$$` alone on a line, has its body
indented by one tab, and closes with `$$` alone on a line:

```
$$
	R_x(0) = x @tag(fixed-point)
$$
```

Never write `$$f(x) = 0$$` on one line, and never use `\[ ... \]`. The site-wide macros are in
`site/katex.config.ts` (`\bbR`, `\calI`, `\bfx`, `\defeq`, `\R`, `\E`, `\retr`, ...); notation for
one document goes in its `katex_macros`.

skoro's own notation is written with its macros, never with the letters they stand for, so that it
can change in one place: `\manifold` and `\otherManifold`, `\point` and `\otherPoint`, `\tangent`
and `\otherTangent` (a tangent vector or a frame of them alike), `\dimension`, `\map`,
`\discretePoint`, `\order` and `\req`; `\density`, `\logterm`, `\proposal`, `\weight` and `\kernel`; and the
typography `\D`, `\dd` and `\hvp`. `katex.config.ts` says what each stands for.

## Graphs

skoro has several graphs, and a document never says "the graph" alone: it names the one it means.

- The **compilation graph** has the model's values as nodes (quantities, constants, and mapped
  values, among them the terms' log-densities) and an edge from each input of an operation to its
  output. Its roots are the quantities and constants. Programs are compiled from it.
- The **probabilistic graph** has the model's quantities and terms as nodes, an edge from each
  quantity a term is conditioned on to the term, and an edge from the term to the quantity it
  scores. Ancestral order and Markov blankets are read on it.
- A program's **computation graph** has its slots as nodes and an edge from each input of a step to
  its output; its **backpropagation graph** has the same nodes with every edge reversed, carrying
  the pull-backs of the reverse pass.

A node with no incoming edge, and so no ancestor, is a *root*: the inputs of a computation are the
roots of its graph, never its leaves. A *leaf* has no outgoing edge, so a node is a root of a
computation graph if and only if it is a leaf of its backpropagation graph.

Before a document reasons about a graph's structure (its roots, edges, paths or neighbours), it
says what the nodes and edges represent and the properties it relies on ("the roots are the inputs
of the map"), and shows a picture (see [Figures](#figures)), or it links to a part of the site that
does all three. The core's mathematics introduces the compilation and probabilistic graphs
(`archive/core/math`, under "The compilation graph" and "The probabilistic graph"), and the note
on automatic differentiation the computation and backpropagation graphs, until the concepts pages
do.

Number only the equations the text refers back to, with `@tag(slug)` at the end of the last line,
and refer to them as `[](eq:slug)`, which renders as "(1)". From another document:
`[](eq:math/<slug>/<eq>)` or `[](eq:archive/<entry>/<document>/<eq>)`. An implementation companion
should point at the equations it implements this way.

## Statements and proofs

A definition, theorem or lemma that the text states formally and returns to is a statement: a
short document of its own in a `statements/` folder beside the document that shows it.

```yaml
---
type: statement
kind: lemma # theorem, proposition, corollary, definition, remark, example
title: Whitening # optional
---
```

Its body is the statement alone. The document must be `.svx` to show it:

```svelte
<script>
	import Statement from '$lib/components/Statement.svelte';
	import Proof from '$lib/components/Proof.svelte';
	import * as whitening from './statements/whitening.md';
</script>

<Statement {...whitening} />

<Proof>

The covariance of $L^{-1} \varepsilon$ is $L^{-1} L L^\top L^{-\top} = I$.

</Proof>
```

Statements and equations share one count within a document. Refer to one as
`[%full](statement:whitening)` ("Lemma 2"). Never number or italicise a statement by hand.

## Citations

Entries go in `site/content/references.bib`, keyed `<surname><year>`. `[](cite:absil2008)` renders
as `[Absi08]`, and `[Section 4.1](cite:absil2008)` as `[Absi08, Section 4.1]`; each page ends with
exactly what it cites. Cite only sources you are certain of, with their real details: leave a
citation out rather than guess one.

## Callouts

GitHub-style blockquotes, with an optional title after the marker:

```
> [!DECISION] Why conditioning belongs to the query
> ...
```

`NOTE`, `TIP`, `WARNING`, `DECISION`, `TODO`. A `[!TODO]` is an open question: `wf approve`
refuses a document that still has one, so a draft carries its open questions as `[!TODO]`s until
they are answered. A page's `[!DECISION]` says briefly why something is the way it is and links
to the archive entry that decided it.

## Code

- **Pages quote real code.** A fence whose only line is `{path}` or `{path:start..end}` is
  replaced by those lines of the file (1-indexed, inclusive), with a caption linking to GitHub.
  Paths are relative to the document. Quoting doc comments renders them as Markdown.
- **Archive documents never quote with `{path}`.** The quote is re-read at every build, so it
  would change as the code does, and a line range would eventually break the build. Sketch the
  code inline, and when an existing interface matters, paste the lines and name the file and the
  commit they were read at.
- Sketches are Rust, compile-plausible, and show signatures and types rather than bodies, unless
  the body is the point.

## Figures

A graph of any of the kinds in [Graphs](#graphs) is drawn with `<ModelGraph spec={name} />` from a
`*.graph.ts` file beside the document (see `site/README.md`). Any other figure is an image or a Svelte component in the
document's folder, referenced relatively.
