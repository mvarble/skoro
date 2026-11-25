# Writing skoro's documents

skoro's own conventions for the documents in `site/content/`, over the ones every project
documented this way shares: `.agents/reference/mesearch/authoring.md` (what each kind of writing
promises, frontmatter, links, prose, callouts, code and figures) and `markup.md` beside it
(mathematics, numbered equations, statements and citations). Where they differ, this file wins.
`site/README.md` covers the same ground for a person editing the site.

## Where things go


```
site/content/
    landing.md                          the home page's prose
    <track>/<slug>/index.md             a page: start, guide, concepts, reference
    reference/crates/<crate>/index.md   a crate's description, shown on its page with its
                                        reading order (Markdown, read by no other route)
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
  Fisher information, a junction tree). Together they hold all the general mathematics the site
  relies on, in an order: read from the first to the last they are one story, each note resting
  only on the notes before it. A note the site already has is linked, never rewritten in another
  document. [The math notes](#the-math-notes) says how they are written and kept.
- **Archive documents discuss what may be.** They argue, weigh alternatives and leave room for
  doubt, as they stood when approved. Once approved they are frozen: they may drift from the code,
  and that is what they are for.

Before writing about an idea, search for it: `grep -rni 'retraction' site/content/`. If the site
explains it, link there. If a document needs a piece of general mathematics the site does not
have, that mathematics is written as a math note, or as a section of one, at its place in the
order, and the document links to it and keeps what is skoro's own.

## Frontmatter

As the shared conventions say, and:

- A math note's `part` is one of `geometry`, `probability` and `estimation` (`MATH_PARTS` in
  `site/src/lib/site.ts`).
- An entry's `index.md` also lists the `crates` it changed, by their directories under `crates/`,
  written at `publish`. `wf approve` refuses a changed crate that is not listed, a listed one the
  branch did not change, and a changed crate with no description in
  `site/content/reference/crates/<crate>/index.md`.

## The math notes


**What belongs in a note.** A piece of mathematics is *general* when it can be stated and proved
without naming anything skoro defines: no quantity, term, query, role, binding or key, and neither
of the library's graphs. It is *skoro's own* when its subject is something skoro defines, or when
it is a choice among true things (which chart the rotations use, that a volume is declared). The
first kind is written in a note. The second stays in the archive, which argued for it, and in the
concept pages, which say what it now is. Where a statement about skoro is an instance of a general
one, the general one gets the note and the instance stays behind. A note never speaks of skoro,
links to a page or to the archive, or lists either in its `depends_on`.

**The geometry is written in tangent coordinates.** The first note,
[Smooth manifolds, smooth maps, and retractions](../../site/content/math/retraction/index.svx),
defines tangent spaces, differentials and retractions without coordinates, then fixes a basis
$\basis_\point$ of each tangent space and ends with an index of the abuses of notation that follow
from it. Every later note writes a tangent vector as a column in $\T_\point\manifold = \R^\dimension$,
a derivative as the matrix $\D\map(\point)$, and a retraction as $\oplus$ and $\ominus$, and never
writes $\retr$, $\dd\map_\point$ or $\basis_\point$. A note that needs a new abuse adds a row to
that index. A result that compares tangent coordinates at nearby points (a second derivative, a
density in a moving chart) assumes that $\oplus$ is smooth near the point, and says so.

**A note is a chapter.** It may assume the notes before it, use their notation and results without
restating them, and refer back to them by equation and statement. It never rests on a note after
it. It is still one idea, and its `depends_on` is still the least a reader needs, not everything
before it.

**A feature's general mathematics is a note.** At an entry's `math` stage, whatever in its
mathematics is general goes into a note, new or existing, at its place in the order; the entry's
`math` document links to it and keeps the problem, the choices and the reading in skoro's terms.
A chore never opens the notes. When a note is inserted, read the notes after it and rewrite the
ones that explained what it now explains.

**A rewrite keeps the archive's references true.** The archive's documents refer to the notes'
equations (`eq:math/<slug>/<tag>`), statements (`statement:math/<slug>/<name>`) and headings
(`math/<slug>/#<anchor>`): `grep -rn 'math/<slug>' site/content/archive` before rewriting a note.
In order of preference: what is cited keeps its name and its note wherever the story has a place
for it there; where it reads better in another note it moves, and the archive's reference is
changed to follow it; where it belongs in no note any longer, it is written into the archive
document that cited it. The last two are amendments of a done entry, made at `publish` and listed
under `amends`.

## Mathematics

The site-wide macros are in `site/katex.config.ts` (`\bbR`, `\calI`, `\bfx`, `\defeq`, `\R`, `\E`,
`\retr`, ...); notation for one document goes in its `katex_macros`.

Lay equations out so that they read the way the argument runs:

- **Chains are displayed.** A chain of equations, `a = b = c`, never goes inline, least of all one
  with stacked vectors. An equation between stacked vectors is displayed even when it is a single
  equality. A lone short equation, such as $w\transpose x = \lvert x \rvert^2 - \alpha x_1$, may
  stay in the sentence.
- **A short chain takes one line.** If the reader can take it in at once, it goes on one line. Several
  such chains in a row go in one `gathered` block, a chain to a row. Give a chain one relation per
  row, in `aligned`, only when it is a derivation to follow step by step or too wide for one line.
- **Align on the last relation of the first row.** When the first row holds several relations,
  the `&` goes before the last of them, so that the next row continues from it:

  ```
  $$
  	\begin{aligned}
  		a = b + c + d &= f + g \\
  		&= h + i .
  	\end{aligned}
  $$
  ```

  Never put it before the first, as in `a &= b + c + d = f + g`.
- **Prose separates equations that do not continue each other.** Write the first in a display,
  then "and" or "where", then the second, rather than aligning them together or setting them side
  by side with `\qquad`. There are two exceptions. Short definitions that fit side by side, such as
  a small example's $A$ and $b$, may share a display with `\qquad` between them. A parallel pair
  that shares one `@tag` goes in a `gathered` block. A tag after `\end{aligned}` or
  `\end{gathered}` numbers the whole block.
- **Commas stack blocks.** The tuple $(v, u)$ is the column vector with $v$ above $u$; never write
  it with semicolons. Say so where a document first uses one. Write a row by juxtaposition, as in
  $[A \;\, b]$. Matrices and displayed vectors use `bmatrix`.

skoro's own notation is written with its macros, never with the letters they stand for, so that it
can change in one place: `\manifold` and `\otherManifold`, `\point` and `\otherPoint`,
`\randomPoint` and `\otherRandomPoint` (the random variables, in boldface), `\tangent` and
`\otherTangent` (a tangent vector or a frame of them alike), `\dimension`, `\map`,
`\discretePoint`, `\order` and `\req`; `\positive` for the positive reals $(0, \infty)$, written
$\mathbb R_{>0}$ and never $\mathbb R_+$, which some read as $[0, \infty)$; `\density`, `\logterm`,
`\proposal`, `\weight`, `\kernel` and `\key`; and the typography `\D`, `\dd` and `\hvp`.
`katex.config.ts` says what each stands for.

Number only the equations the text refers back to, with `@tag(slug)` at the end of the last line,
and refer to them as `[](eq:slug)`, which renders as "(1)". The reference brings its own
parentheses, so never wrap it in more: "by [](eq:slug)", not "([](eq:slug))". From another document:
`[](eq:math/<slug>/<eq>)` or `[](eq:archive/<entry>/<document>/<eq>)`. An implementation companion
should point at the equations it implements this way.

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
of the map"), and shows a picture (see "Figures" in `mesearch/authoring.md`), or it links to a part of
the site that does all three. The core's mathematics introduces the compilation and probabilistic graphs
(`archive/core/math`, under "The compilation graph" and "The probabilistic graph"), and the note
on automatic differentiation the computation and backpropagation graphs, until the concepts pages
do.

## Code

Sketches are Rust, compile-plausible, and show signatures and types rather than bodies, unless the
body is the point.
