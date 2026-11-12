---
summary: The Mathematics section holds all the general mathematics the site relies on, as notes in an explicit order that the build checks, so that read from start to finish it is one story in which each note rests only on those before it.
---

## Context

### The section as it stands

The Mathematics section was set up as a library: one idea to a note, each note written to be read
alone, and the notes linked by what each builds on. It holds seven notes, about 19,000 words with
their statements. Its index shows a map of the notes and of the archive's documents that build on
them, and under the map a list. The list is in an order the site derives: `readingOrder`, in
`site/src/lib/cms/runtime.ts`, places each note after the notes it depends on and breaks ties by
title. Today that gives

1. Automatic differentiation over frames,
2. Elimination in square-root form,
3. Models built forward, for the measure theorist,
4. Retractions and tangent charts,
5. Derivatives in tangent charts,
6. Product manifolds,
7. Densities in tangent charts.

Every note there comes after what it depends on, and the list is still not an order anyone would
read in: orthogonal factorizations sit between automatic differentiation and measure theory because
of the letter E. A note has no previous or next note, as a page of a track has.

The Concepts track is the contrast. Its eight pages each carry an `order`, each rests only on the
pages before it, and a reader who starts at the first and stops at the last has been told how the
library is built.

### Where the mathematics is

Most of the site's mathematics is not in the notes. The archive's ten `math` documents hold about
66,000 words, more than three times the notes, because the rule for writing them has been to
explain an idea in the entry that needs it and to promote it to a note "only when it stands alone
and takes several sections, or when a second document needs it". Under that rule these ideas, all
of which the library's pages rely on, have no note:

- the rotations, the unit quaternions and their double cover, and the rigid motions $SE(3)$ and
  $SE_2(3)$ ([rotations](../rotations/math.svx));
- the quotient of a manifold by a finite group, and what the antipode does in the stereographic
  chart ([quaternion signs](../quaternion-signs/math.svx));
- the reference measures of the flat and positive spaces ([measures](../measures/math.svx));
- the normal distribution in a chart, with its scale, its sampler and its expected information
  ([normal](../normal/math.svx));
- the improper Gaussian factor ([improper Gaussian](../improper-gaussian/math.svx));
- linearization, damping and the Levenberg–Marquardt method, the Laplace approximation, and Fisher
  scoring by roots of the expected information ([linearization](../linearization/math.svx),
  [examples](../examples/math.svx));
- the calibration of credible sets ([examples](../examples/math.svx#calibration));
- the weight of a draw made from proposals ([core](../core/math.svx#terms-and-the-importance-weight));
- how the error of a sine and a cosine reaches a closed form near a removable singularity
  ([accuracy on the GPU](../wgpu-accuracy/math.md)).

So the pages send their readers into the archive for mathematics: the Concepts and Guide pages
refer to the archive's `math` documents twenty-three times. That is the wrong place to learn
something, for three reasons.

An archive document argues. It is written to settle what skoro should do, so its mathematics comes
wrapped in a problem, alternatives and a decision, and is stated in skoro's terms: a reader who
wants to know what the double cover is finds it in an argument about whether a rotation should be
stored as a matrix or as a quaternion.

An archive document is frozen, and may drift. [The rotations' mathematics](../rotations/math.svx)
makes the rotations the unit quaternions under the stereographic chart, and
[a later entry](../quaternion-signs/math.svx) replaced that with the rotation group itself under the
Cayley chart. The first document now describes a design the library no longer has, and is still
the only place on the site where the double cover is explained.

And nothing tells a reader where the mathematics begins or whether they have seen all of it. The
map shows every arrow at once, which is a picture of the network and not a way through it.

## Decision

> [!DECISION] Mathematics is read from start to finish
> The Mathematics section is to the mathematics what Concepts is to the library. It holds every
> idea the site relies on that is true independently of skoro, one idea to a note, and the notes
> stand in an explicit order in which each rests only on the notes before it. Read from the first
> to the last, they are the whole of the mathematics, told once.

Four things make that position concrete.

### What belongs in the section

All the general mathematics, and only that. A piece of mathematics is *general* when it can be
stated and proved without naming anything skoro defines: no quantity, term, query, role, binding
or key, and neither of the library's graphs. It is *skoro's own* when its subject is something
skoro defines, or when it is a choice among true things. The first kind gets a note. The second
stays where it is: in the archive, which argued for it, and in the concept pages, which say what
it now is.

By that test the mathematics of retirement, of freeing the constants a retirement leaves behind, of
random keys, of the order a pass requires and of declared volumes is skoro's own, and so is every
choice: which chart the rotations use, that a volume is declared and not computed, that the normal
asks for a global chart. Where a statement about skoro is an instance of a general one, the general
one gets the note and the instance stays behind. Retirement is the example: that integrating a
block of variables out of a sum of squared rows leaves one factor on its neighbours is in
[the note on elimination](../../math/square-root-elimination/#eliminating-one-block), and what a
retirement does to a model is not.

The test leaves a few cases to rule on.

> [!TODO] The line, case by case
> These are the cases the test does not settle by itself, each with the ruling this draft
> proposes.
>
> - **The compilation graph** is skoro's own. It is a computation graph in the sense of
>   [the note on automatic differentiation](../../math/automatic-differentiation/), with skoro's
>   kinds of node, and nothing general about it is missing from that note.
> - **The probabilistic graph** is skoro's own as an object, a graph of quantities and terms. What
>   is general is the *dependence graph* of a model built forward, with its ancestral orders and
>   Markov blankets, which
>   [the note on such models](../../math/generative-models/#how-skoro-is-built-on-this) already
>   introduces under the heading "How skoro is built on this". That section is the one place where
>   a note speaks of skoro and links into the archive. Proposed: its general content stays in the
>   note as a section on the dependence graph, and what it says of skoro moves to the page
>   [Terms and the probabilistic graph](../../concepts/terms/).
> - **The sliding window** is skoro's own: it is an example, a model and a schedule of retirements.
>   The general fact inside it, that eliminating a block of a linear-Gaussian model is the update
>   and prediction of a Kalman filter, becomes an example in the note on the improper Gaussian.
> - **Storage up to a group** splits. That a retraction which commutes with a finite group acting
>   freely descends to the quotient, with its chart volume and its densities, is general and gets a
>   note. That a type is *stored* as a point of the covering space, and how it declares so, is
>   skoro's own.
> - **The measurements of one GPU** are not mathematics and stay in the archive. The analysis they
>   prompted, of how absolute and relative error reach a closed form and what a cutoff must
>   satisfy, is general.
>
> Is each of these right?

### The order is explicit, and the build checks it

Each note says where it stands, as a page does. A page belongs to a track and has an `order`
within it; a note belongs to a *part* and has an `order` within it, and the parts are listed in
`site/src/lib/site.ts` in the order they are read, as the tracks are:

```yaml
---
title: Densities in tangent charts
part: probability
order: 2
depends_on: [retraction, product-manifold, rotations, rigid-motions, generative-models]
---
```

The build sorts the notes by part and then by `order`, and refuses

- a note without a part or an order, or two notes at the same place;
- a note whose `depends_on` names a note at its own place or after it;
- a note whose `depends_on` names anything but a note, since a note that rested on a page or on an
  archive document would not be independent of skoro.

`depends_on` keeps its meaning: it lists what a reader must understand first, and remains the solid
arrows of the map. The order is a second, stronger statement, that everything before a note may be
assumed, and the check ties the two together in the one direction a build can check.

> [!TODO] Flat, or in parts
> The alternative is one flat sequence, as a track is. This draft proposes parts, for two reasons.
> The mathematics is not a chain: geometry and probability do not meet until densities are pushed
> through charts, and elimination is linear algebra that rests on neither until linearization
> joins all three. A flat list of some nineteen notes hides that a reader who knows manifolds may
> begin at the seventh, and parts say so. And an insertion renumbers the notes after it, which in
> a part is at most a handful of files. The cost is one more field and one more list to keep. Are
> parts wanted, and are three the right ones?

### A note is a chapter

A note may assume the notes before it, use their notation and results without restating them, and
refer back to them by equation and statement. It may not rest on a note after it. It remains one
idea, and it remains independent of skoro.

This changes what a note promises. It has been "worth reading alone"; it becomes worth reading in
its place. A reader who arrives at a note from a concept page is not asked to read everything
before it: the note's `depends_on` is still the least they need, and it is shown on the note as it
is now.

When a note is inserted, the notes after it are rewritten to lean on it, as the concept pages
after a new page are. A note is therefore never finished in the way an archive document is: it is
kept true, and kept in step with its neighbours.

### The notes, and their order

The story has three parts: `geometry`, `probability` and `estimation`. The tables give each note's
place, what it holds, and the notes it rests on directly, where a bare number is a note of the same
part. A new note also names the archive documents its mathematics is drawn from.

**Spaces and their charts** builds the manifolds and how to differentiate on them.

| | note | holds | rests on |
| --- | --- | --- | --- |
| 1 | `retraction` | retractions, tangent charts, $\oplus$ and $\ominus$ | |
| 2 | `automatic-differentiation` | programs, their graphs, the forward and reverse passes, Hessian products | |
| 3 | `derivatives-in-charts` | the derivative, gradient and Hessian product read in charts | 1, 2 |
| 4 | `product-manifold` | products, stacked tangents, block derivatives | 1, 3 |
| 5 | `quotient-manifold` (new) | a manifold divided by a finite group acting freely; the retraction that descends to it. From `quaternion-signs`. | 1, 3 |
| 6 | `rotations` (new) | $SO(3)$, the unit quaternions, the double cover and why it has no section, the exponential and the logarithm, the stereographic chart and its antipode, the Cayley chart, Euler angles. From `rotations` and `quaternion-signs`. | 1, 3, 5 |
| 7 | `rigid-motions` (new) | $SE(3)$ and $SE_2(3)$ as semidirect products, the translated chart, their maps. From `rotations`. | 4, 6 |
| 8 | `removable-singularities` (new) | closed forms with a removable singularity, their series, how the error of a sine and a cosine reaches them, and the conditions on a cutoff. From `rotations` and `wgpu-accuracy`. | 6, 7 |

**Probability on those spaces** builds models, densities and the normal.

| | note | holds | rests on |
| --- | --- | --- | --- |
| 1 | `generative-models` | models built forward, densities against reference measures, conditioning, likelihood factors, the dependence graph | |
| 2 | `chart-volume` | the volume of a chart, densities pushed through charts, the measure a retraction normalizes, the usual charts; gains the flat and positive spaces and the quotient, from `measures` and `rotations`. | geometry 1, 4–7; 1 |
| 3 | `normal` (new) | the normal in a chart, its scale as a whitening, its sampler, the log-normal as an instance, its expected information. From `normal`. | 1, 2 |
| 4 | `importance-weights` (new) | drawing a model ancestrally from proposals, and the weight that corrects the draw. From `core`. | 1 |

**Least squares and Gaussian approximations** builds the estimators.

| | note | holds | rests on |
| --- | --- | --- | --- |
| 1 | `square-root-elimination` | least squares by orthogonal factorization, elimination of a block, the elimination tree, marginals | |
| 2 | `improper-gaussian` (new) | the improper Gaussian factor, its forms, how factors add, the factor of full rank as a normal, the Kalman filter as an example. From `improper-gaussian` and `linearization`. | probability 3; 1 |
| 3 | `linearization` (new) | a sum of normal terms linearized into whitened rows and linear terms, the Gauss–Newton matrix, the linear-Gaussian case as exact. From `linearization`. | geometry 3; probability 2, 3; 1 |
| 4 | `levenberg-marquardt` (new) | damping, the gain ratio, the schedule and the stopping rule. From `linearization` and `examples`. | 1, 3 |
| 5 | `laplace-approximation` (new) | the approximation at a mode, its marginals by elimination, what its covariance leaves out. From `linearization` and `examples`. | 1, 3 |
| 6 | `fisher-scoring` (new) | the expected information in place of the Hessian, and its roots as a second set of rows. From `normal` and `linearization`. | probability 3; 3 |
| 7 | `calibration` (new) | credible sets that contain the truth as often as they claim, under the model, and the error an approximation adds. From `examples`. | probability 1; 5 |

That is nineteen notes, twelve of them new.

> [!TODO] The list and the order
> The tables are a proposal. The choices in them that could go the other way:
>
> - **Reference measures have no note of their own.** What `measures` holds that is general (the
>   flat and positive spaces, the area formula) is a few paragraphs beside what `chart-volume`
>   already says of the measure a retraction normalizes and of products, so it is folded in.
> - **`quotient-manifold` is its own note,** short as `product-manifold` is, and not a section of
>   `rotations`, because it is one idea and the rigid motions use it too.
> - **`rotations` and `rigid-motions` are two notes,** and `levenberg-marquardt`, `linearization`,
>   `laplace-approximation` and `fisher-scoring` are four where they could be two.
> - **`importance-weights` is a note.** It was not among the ideas first counted as missing, and it
>   is general.
> - **`removable-singularities` closes the first part** and is not a part of its own, since one
>   note does not make a part and its subject is the maps of the two notes before it.
> - **`retraction` comes first,** before automatic differentiation, though neither rests on the
>   other, because the first concept page rests on it.
>
> Which of these should change, and is anything missing or misplaced?

### This entry writes the notes

The twelve new notes, and the revisions of the seven, are written at this entry's publish stage,
together with the pages' changes. A section that claimed to be the whole story while missing most
of it would be wrong on the day it was published, so the claim and the notes arrive together.

Two rules keep so large a publish reviewable. It adds no new mathematics: every statement in a new
note is one that an approved archive document or an existing note already proves, restated in
general terms, or a standard fact with its citation. And if writing a note shows a frozen document
to be mistaken, the publish stops and says so, since the code built from that document may be
mistaken with it. The notes are committed one to a commit, in reading order, so that they can be
read as they arrive.

## Consequences

**The site learns the order.** These changes are to the site's own code and to the tooling's
references, and are not gated:

- `part` and `order` in a note's frontmatter, the parts in `site/src/lib/site.ts`, and the check
  above in place of `readingOrder`. The same check can run on the tracks, whose pages already have
  an order; all of them pass it today.
- The Mathematics index lists the notes by part, in order and numbered, above the map. The map
  stays, as the picture of why the order is what it is.
- A note shows the previous and the next note, as a page of a track does. The sidebar and
  `llms.txt` follow, since both read `cms.math.list()`.
- `MATH.blurb` says what the section now is: the mathematics skoro rests on, told from the
  beginning, each note one idea resting only on the notes before it.

**Later entries keep the story whole.** `authoring.md` gains the rule that a feature's general
mathematics is written as a note, or as a section of one, at its place in the order, and its
threshold for promoting an idea goes. The note is written at the feature's `math` stage, which is
when the notes open, and the entry's `math` document links to it and keeps what is skoro's own:
the problem, the choices, and the reading in skoro's terms. Only a feature or a decision can do
this, since a chore never opens the notes. A new idea therefore costs more than its own note: the
notes after it are reread, and rewritten where they explained what it now explains.

**The pages point at the notes.** At publish, a page's link to an archive document for a piece of
general mathematics becomes a link to the note that holds it, and its `depends_on` names that
note. A page keeps its links to the archive for what the archive is for, why skoro is the way it
is. [Reading these docs](../../start/reading-these-docs/) and
[How skoro is developed](../../reference/how-skoro-is-developed/) are rewritten where they
describe the section. The archive's documents are frozen and keep their links to one another.

**Frozen documents hold the notes in place.** The archive's documents refer to the notes' numbered
equations and statements 98 times, 36 distinct ones, and to ten of their sections by heading. A
frozen document cannot be changed to follow a note, so the note must not move from under it.

> [!TODO] What a rewrite must keep
> Proposed: a note's slug, and every equation tag, statement and heading that a frozen document
> refers to, are kept where they are. A rewrite may add to a note, reword it and reorder it, and
> may not move a cited statement into another note or rename a cited heading. For this entry that
> decides one thing: the table of the usual charts stays in `chart-volume`, where the archive
> points six times, and the new `rotations` note comes before it and derives what the table
> lists. The alternative is to teach the site an alias, as a moved page has `formerly`, so that a
> statement could move and still resolve. This draft proposes not to build that until a note
> first needs it. Is that the right constraint, or should the alias come now?

**What it costs.**

- The mathematics is on the site twice: once in a frozen document, as it was argued, and once in a
  note, as it is true. The rule that a note is "linked, never rewritten in another document" binds
  documents written after the note; the archive that came before it is history.
- A note is not frozen. An approved `math` document that leans on a note leans on text that may be
  rewritten later, and what the note said at the time of approval is in git and nowhere else. That
  is already so of the 36 equations and statements above.
- The check sees only what a note declares. It refuses a dependency that points forward; it cannot
  find one that was never listed, and it does not read a note's links. Whether a note truly rests
  only on those before it is still a matter of reading it.
- The publish is large: by the size of the documents the new notes draw on, twenty-five to
  thirty-five thousand words of notes, approved at once.

## Alternatives

**Ordering only the notes there are.** The seven notes would have had an order within a day, and
the order would have told a story with most of it missing: it would pass from product manifolds to
a table of the charts of rotations that no note had introduced, and the pages would have gone on
sending readers to the archive for the rest. An order is a promise that the reader has been told
everything so far, and seven notes cannot keep it.

**Deriving the order from `depends_on`.** This is what the site does now, and the list in
[the section as it stands](#the-section-as-it-stands) is the result. The seven notes have 84 orders
that respect their dependencies, and a rule that picks one by title picks without regard to what
reads well after what. Dependencies say what must come before a note. They do not say what should
come next, and that is a judgment the author of a story makes.

**Keeping each note standalone.** A note that may assume nothing either repeats what earlier notes
establish, so that the normal would derive again how a density is pushed through a chart, or
replaces the repetition by links, which is the section as it stands. An order that the notes do
not use is only a listing.

**Leaving the new notes to later entries.** Each feature would have written the notes its own
mathematics needed. But the mathematics missing from the section belongs to features that are
done, and no entry is coming that would have reason to write it: a chore cannot open the notes,
and a decision opened only to write a note has nothing to decide. The section would have claimed
to be the whole story for as long as that took.

**Keeping the order in one file.** A single list of slugs, with the parts as its headings, would
make an insertion a one-line change and show the whole table of contents in one place. It would
also be a second mechanism beside the one the tracks use, and a note would no longer say where it
stands. Parts keep the renumbering small enough that the pages' own mechanism serves.
