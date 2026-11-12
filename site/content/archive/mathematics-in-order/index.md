---
title: "The Mathematics section, read start to finish"
pages:
    - start/reading-these-docs
    - guide/first-model
    - guide/pose-graph
    - concepts/spaces
    - concepts/terms
    - concepts/queries
    - concepts/the-normal
    - concepts/rotations-and-motions
    - concepts/linearization
    - reference/how-skoro-is-developed
crates: []
---

The Mathematics section was a collection of notes, one idea to a note, linked by what each builds
on, while most of the mathematics the pages relied on sat in the archive's frozen documents, argued
in skoro's terms. This entry made the section the mathematical analogue of Concepts (bottom-up):
it holds every piece of mathematics the site relies on that is true independently of skoro, in
nineteen notes that stand in an explicit order, in three parts, each note resting only on the
notes before it. [The writeup](writeup.md) argues the position.

Twelve notes are new, each restating in general terms what an approved document of the archive
already proves: quotients by a finite group, the rotations and unit quaternions, the rigid
motions, removable singularities in floating point, the normal distribution in a chart, importance
weights, the improper Gaussian factor, linearization and the Gauss–Newton step, the
Levenberg–Marquardt method, the Laplace approximation, Fisher scoring, and the calibration of
credible sets. Two of the seven that existed were revised: the note on models built forward no
longer speaks of skoro, and the note on densities in tangent charts reaches the quotient. The site
reads a note's `part` and `order`, refuses a note that rests on anything but an earlier note, lists
the notes in order and gives each its previous and next. The pages now send their readers to the
notes for the mathematics, and to the archive for why skoro is the way it is.

It also changed how the archive is kept. A later entry may bring a finished entry's references into
the notes up to date, on the record, and `bin/wf` learned to allow, require and record that. This
entry needed no such change: every equation, statement and heading that the archive cites is where
it was.
