---
summary: The geometry begins with smooth manifolds, tangent vectors and the tangent bundle as Lee defines them, fixes a basis of each tangent space by a convention that need not be smooth, and only then defines a retraction; every later note speaks in tangent coordinates alone.
katex_macros:
    '\basis': '\iota'
    '\dimx': 'd_x'
    '\dimy': 'd_y'
---

## Context

### The order of the first two notes

The Mathematics section opens with
[Retractions and tangent charts](../../math/retraction/) (geometry, 1) and comes back to
[Derivatives in tangent charts](../../math/derivatives-in-charts/) (geometry, 3), after the note on
automatic differentiation. The first defines a retraction as a smooth map
$R : \T\manifold \to \manifold$ with

$$
	R_\point(0) = \point \qquad \text{and} \qquad \D R_\point(0) = I_\dimension ,
$$

"with $\T_\point\manifold$ identified with $\R^\dimension$". The third defines $\D$, the derivative
of a map between manifolds, by reading the map in the charts that retractions provide.

So the derivative a retraction is defined by is not defined until two notes later, and when it is,
it is defined through retractions. The definition rests on a derivative in *some* chart, which the
notes never name. Neither do they define what $\T\manifold$ is, or what it means for a map on it to
be smooth. Both definitions are standard and short, and the notes need them.

### The identification of tangent spaces

The first note handles the tangent spaces in one paragraph: each $\T_\point\manifold$ "is identified
with $\R^\dimension$ by a basis chosen smoothly in $\point$". It then concedes that the sphere has no
such basis, by the hairy ball theorem, and asks a computation to stay where the basis is smooth.
That assumption carries more weight than it looks. First-order notions (the derivative at a point,
the chain rule, the tangent chart at a point) need a basis at each point and nothing more. Second-order
notions (the mirror identity, the chart Hessian, densities read in the chart at a moving point)
compare bases at nearby points, and they do need the bases to vary smoothly. The note does not say
which is which.

skoro itself works only in coordinates. Every space's tangent is a `Vector` of the space's
dimension, as the doc comment of `crates/skoro-core/src/retraction.rs` says (read at `d90aa47`):

```rust
//! The tangent of every space is a `Vector` of the space's dimension.
```

Its sphere, the embedded type of the acceptance test, carries exactly the data in question
(`crates/skoro/tests/common/sphere.rs`, at `d90aa47`):

```rust
//! The unit sphere `S²` of the acceptance test: storage 3, dimension 2, with the basis `B(x)`
//! smooth off the south pole and the inverse stereographic retraction scaled so that its
//! derivative at 0 is `B(x)`.
```

$B(\point)$ is a basis of the tangent space at each point but the south pole, smooth where it is
defined. The retraction is built so that, read through $B(\point)$, its derivative at $0$ is the
identity. The mathematics should say this in its own terms: a basis of each tangent space, how
regular that basis is, and a retraction defined independently of it.

## Decision

> [!DECISION] Smooth maps before retractions
> The first note of the geometry, retitled *Smooth manifolds, smooth maps, and retractions*, sets
> out smooth manifolds, smooth maps, tangent vectors, differentials and the tangent bundle as
> [Chapters 1–3](cite:lee2013) define them, stating each briefly and proving none of it. It then fixes
> a basis $\basis_\point$ of each tangent space by a convention that need not be smooth. Only then
> does it define a retraction, without reference to any basis. It reads the retraction in those
> bases as $\point \oplus \tangent$, obtains the tangent charts from the inverse function theorem,
> and shows that derivatives read through the retraction are the derivatives the bases define. It
> ends with a table of the notation that every later note uses. No later note speaks of
> derivations, of $\T\manifold$ as a manifold, or of a differential that is not a matrix.

The note keeps its slug, `retraction`, so that the links to it stay valid. Its sections follow,
in order.

### Smooth manifolds and smooth maps

The note cites [Chapters 1 and 2](cite:lee2013) for a cohesive theory, and recalls what it uses.
$\manifold$ is a smooth manifold of dimension $\dimension$: a topological manifold with a smooth
structure, that is, a maximal atlas of charts $(U, \varphi)$ whose transition maps are smooth. A
map $\map : \manifold \to \otherManifold$, between smooth manifolds of dimensions $\dimx$ and
$\dimy$, is of class $C^k$ when it is continuous and, for every chart $(U, \varphi)$ of
$\manifold$ and every chart $(V, \psi)$ of $\otherManifold$, the coordinate representation

$$
	\psi \circ \map \circ \varphi^{-1} : \varphi\bigl(U \cap \map^{-1}(V)\bigr) \to \R^\dimy
$$

is of class $C^k$. Continuity makes $U \cap \map^{-1}(V)$ open, so the representation is a map
between open subsets of Euclidean spaces. Because the transition maps are smooth, it suffices to
check this for the charts of any atlas. The set of these maps is $C^k(\manifold, \otherManifold)$,
and a smooth map is one of class $C^\infty$.

### Tangent vectors and the tangent bundle

The note cites [Chapter 3](cite:lee2013) for these, and recalls:

1. the tangent space $\T_\point\manifold$, a real vector space of dimension $\dimension$, and the
   differential $\dd\map_\point : \T_\point\manifold \to \T_{\map(\point)}\otherManifold$ of a
   smooth map, which is linear and satisfies the chain rule
   $\dd(G \circ \map)_\point = \dd G_{\map(\point)} \circ \dd\map_\point$;
2. that a chart $(U, \varphi)$ around $\point$ gives a basis of $\T_\point\manifold$, its coordinate
   basis, in which $\dd\map_\point$ has the Jacobian of the coordinate representation as its matrix;
3. that a vector space $E$ has each $\T_e E$ identified with $E$ itself, canonically, so that on
   $\R^n$ the differential is the Jacobian;
4. the tangent bundle $\T\manifold = \bigsqcup_\point \T_\point\manifold$, a smooth manifold of
   dimension $2\dimension$ whose charts are built from those of $\manifold$, a point paired with the
   components of a vector in that chart's coordinate basis; and its zero section $0_\point$.

This is everything the definition of a retraction needs, and the note needs nothing more from
Lee.

### A basis of each tangent space

The note then fixes, for each point $\point$, a linear isomorphism

$$
	\basis_\point : \R^\dimension \to \T_\point\manifold , @tag(basis)
$$

a basis of the tangent space at $\point$. Its columns $\basis_\point e_1, \dots, \basis_\point e_\dimension$
are the basis vectors. The coordinate basis of
a chart around $\point$ is one choice, and every choice is the coordinate basis of some chart. In these
bases, the derivative of a smooth map is a matrix:

$$
	\D\map(\point) \defeq \basis_{\map(\point)}^{-1} \circ \dd\map_\point \circ \basis_\point \in \R^{\dimy \times \dimx} . @tag(derivative)
$$

On $\R^n$ the convention is the identity, under the canonical identification, so $\D$ of a map
between Euclidean spaces is its Jacobian. The chain rule becomes the product of matrices,
$\D(G \circ \map)(\point) = \D G(\map(\point))\, \D\map(\point)$, because the bases at
$\map(\point)$ cancel. This is [%full](statement:math/derivatives-in-charts/chain-rule), which the third note proves today,
moved and proved in one line.

No regularity in $\point$ is asked of the convention, and it need not be defined everywhere. It may
be given on an open subset $\manifold_\circ \subseteq \manifold$ only, and everything below is then
said of points of $\manifold_\circ$, with every computation kept inside it. The sphere is the example.
No convention on $S^2$ is smooth everywhere, by the hairy ball theorem, and the basis $B(\point)$ above
is defined everywhere except the south pole and smooth where it is defined. On the rotations, a group's left
translations give a convention that is defined and smooth everywhere.

A convention without regularity is enough for everything at one point: the derivative, the chain
rule and the tangent chart at $\point$. It is not enough for what compares the bases at nearby
points: the mirror identity, the chart Hessian and the second-order half of the passes, and
densities read in the chart at a moving point. For example, $\point \ominus (\point \oplus \tangent)$
is measured in the basis at $\point \oplus \tangent$, and if that basis jumps, the mirror identity
fails at first order. Those results assume that the convention is *smooth near* $\point$: that
$(\otherPoint, \tangent) \mapsto \basis_\otherPoint \tangent$ is smooth from $W \times \R^\dimension$ to
$\T\manifold$ on an open $W \ni \point$, which says that its columns form a smooth local frame on
$W$ ([Chapter 8](cite:lee2013)). The retractions section shows that this is the same as $\oplus$
being smooth near $\point$, which is how later notes state it.

### Retractions

With the tangent bundle and the differential in hand, a retraction is defined without any basis,
as in optimization on manifolds (Absil, Mahony and Sepulchre, *Optimization Algorithms on Matrix
Manifolds*, Section 4.1):

> A _retraction_ on $\manifold$ is a smooth map $\retr : \T\manifold \to \manifold$ such that, for
> every $\point$, its restriction $\retr_\point$ to $\T_\point\manifold$ satisfies
> $\retr_\point(0_\point) = \point$ and $\dd(\retr_\point)_{0_\point} = \mathrm{id}_{\T_\point\manifold}$,
> where $\T_{0_\point}(\T_\point\manifold)$ is identified with $\T_\point\manifold$ as in item 3.

The condition on the differential is coordinate-free, so it holds whatever the convention.
Read in the bases [](eq:basis), the retraction at $\point$ is a map from $\R^\dimension$:

$$
	R_\point \defeq \retr_\point \circ \basis_\point : \R^\dimension \to \manifold ,
	\qquad \point \oplus \tangent \defeq R_\point(\tangent) , @tag(oplus)
$$

and its two conditions become, for every convention,

$$
	\point \oplus 0 = \point \qquad \text{and} \qquad \D R_\point(0) = \basis_\point^{-1} \circ \dd(\retr_\point)_{0_\point} \circ \basis_\point = I_\dimension .
$$

"The derivative at the origin is the identity" now has a meaning, given by [](eq:derivative),
before it is used.

The note also shows that the convention is smooth on an open set $W$ if and only if
$(\point, \tangent) \mapsto \point \oplus \tangent$ is smooth on $W \times \R^\dimension$. If the
convention is smooth, $\oplus$ is the composite of $\retr$ with a smooth map. Conversely, the
retraction's condition on its differential gives

$$
	\basis_\point \tangent = \frac{\dd}{\dd t}\Big|_{t=0} (\point \oplus t\tangent) ,
$$

the velocity of a curve that depends smoothly on $(\point, \tangent)$ when $\oplus$ does. So later
notes can state the regularity they need as "$\oplus$ is smooth near $\point$", and never need to
mention the basis.

The note keeps the two examples, rewritten to show the two pieces apart. On $\R^\dimension$,
$\retr_\point(w) = \point + w$, with the identity convention. On the positive reals, with
$\T_\point\positive = \R$ canonically, the retraction is $\retr_\point(w) = \point\, e^{w / \point}$,
and the convention $\basis_\point(\tangent) = \point\, \tangent$ measures relative change. Together
they give $\point \oplus \tangent = \point\, e^\tangent$.

### Tangent charts

$R_\point$ is smooth and its differential at $0$ is $\basis_\point$, an isomorphism. The inverse
function theorem for manifolds ([Chapter 4](cite:lee2013)) makes it a diffeomorphism from an open
neighbourhood of $0$ onto an open neighbourhood of $\point$. Its inverse,
$\otherPoint \mapsto \otherPoint \ominus \point \defeq R_\point^{-1}(\otherPoint)$, is a smooth chart
centred at $\point$, compatible with the smooth structure because it is a diffeomorphism onto an
open subset of $\R^\dimension$. Its coordinate basis at $\point$ is $\basis_\point$: the tangent chart
reproduces the convention at its centre, and nowhere else need it agree with any other chart. The
equations [](eq:oplus) and `roundtrip` keep their tags.

### Derivatives through the retraction

Because $\D R_\point(0) = I$ and, by the inverse function theorem, $\D(\cdot \ominus \otherPoint)(\otherPoint) = I$,
the chain rule gives the formula every later note computes with. For smooth $\map$ and
$\otherPoint = \map(\point)$,

$$
	\D\map(\point) = \D\bigl[(\cdot \ominus \otherPoint) \circ \map \circ (\point \oplus \cdot)\bigr](0),
	\qquad
	\D\map(\point)\, \tangent = \frac{\dd}{\dd t}\Big|_{t=0} \bigl(\map(\point \oplus t\tangent) \ominus \otherPoint\bigr) ,
$$

the Jacobian at $0$ of a map between open subsets of Euclidean spaces. The derivative that
[](eq:derivative) defines from the differential and the derivative read through the retraction are
the same matrix. This is the statement the third note now makes as a definition, together with the
recipe for deriving a Jacobian by hand: perturb, push through, solve for the tangent, and
differentiate.

### The mirror identity

The mirror identity stays in this note, with its hypothesis made explicit: the convention is smooth
near $\point$. Its proof differentiates
$\varphi(\tangent, c) = R_\point^{-1}(R_{\point \oplus \tangent}(c))$ in $\tangent$, which moves the
basis.

### From here on

The note ends with an index of every place where later notes abuse notation, each standing for an
object of this note read through the convention. A later note never writes $\retr$, $\dd\map_\point$
or $\basis_\point$. It treats a tangent vector as a column of $\dimension$ numbers attached to a
point, as skoro does, and keeps writing $\T_\point\manifold$ for the space of those columns, so that
it is visible which point a vector belongs to. Inside this note $\T_\point\manifold$ is the space of
derivations. After it, $\T_\point\manifold$ is its coordinate space, and the index says so first.

| written later as | stands for | first used in |
| --- | --- | --- |
| $\T_\point\manifold = \R^\dimension$, the tangent space at $\point$ | $\T_\point\manifold$, through $\basis_\point$ | every later note |
| $\tangent \in \T_\point\manifold$, a column of $\dimension$ numbers | the tangent vector $\basis_\point \tangent$ | every later note |
| a frame, a $\dimension \times n$ matrix of tangent vectors at $\point$ | $n$ tangent vectors, each through $\basis_\point$ | automatic differentiation; gradients and Hessians |
| a cotangent $\alpha \in \R^\dimension$, a column acting by $\alpha\transpose \tangent$ | the covector $\alpha\transpose \basis_\point^{-1}$ on $\T_\point\manifold$ | gradients and Hessians |
| $\D\map(\point) \in \R^{\dimy \times \dimx}$ | the differential $\dd\map_\point$, by [](eq:derivative) | this note, and every later one |
| $\D\map(\point)\transpose$, applied to a cotangent | the pull-back of a covector by $\dd\map_\point$ | gradients and Hessians |
| the gradient $g(\point) = \D f(\point)\transpose 1$ | the differential $\dd f_\point$, not a Riemannian gradient | gradients and Hessians |
| $\point \oplus \tangent$, $\otherPoint \ominus \point$, $R_\point$, $R_\point^{-1}$ | $\retr(\basis_\point \tangent)$ and its local inverse | this note, and every later one |
| pairs $(\point, \tangent)$ with $\tangent \in \R^\dimension$ | the tangent bundle $\T\manifold$ over the convention's domain | this note |
| "$\oplus$ is smooth near $\point$" | the convention is a smooth local frame near $\point$ | this note; gradients and Hessians; chart volume |
| $\T_\point\manifold = \T_{\point_1}\manifold_1 \oplus \cdots \oplus \T_{\point_k}\manifold_k$, stacked columns | the direct sum, with the convention that is each factor's on its block | product manifolds |
| the Lebesgue measure on $\T_\point\manifold$, and densities with respect to it | Lebesgue measure on $\R^\dimension$, carried to $\T_\point\manifold$ by $\basis_\point$, which depends on the convention | chart volume; the normal |

The last row is the one where the convention matters most. A density on a tangent space changes by
$\lvert\det\rvert$ of a change of basis, so a density "on $\T_\point\manifold$" is always one in the
coordinates the convention gives.

### Gradients and Hessians in tangent charts

The derivative, its formula through the retraction and the chain rule move into the first note, as
above. The third note keeps the rest under its slug, `derivatives-in-charts`, retitled *Gradients
and Hessians in tangent charts*: the gradient [](eq:math/derivatives-in-charts/gradient), the chart
Hessian [](eq:math/derivatives-in-charts/chart-hessian), and the statement that the passes of
automatic differentiation compute them. These cannot move into the first note: the passes rest on
[automatic differentiation](../../math/automatic-differentiation/) (geometry, 2), and the build
refuses a note that depends on a later one. Keeping the slug keeps the archive's references to the
gradient, the chart Hessian and the passes valid. Only the references to `derivative` and to the
chain rule follow those results into the first note. The chart Hessian and the second-order half of
the passes now state their hypothesis, that $\oplus$ is smooth near $\point$.

## Consequences

**The definitions come in order.** Every symbol in the definition of a retraction is defined before
it. The regularity the bases need is stated once and invoked where it is used, rather than assumed
for every point and then conceded for the sphere.

**Later notes stay in coordinates.** After the table, a reader works with $\oplus$, $\ominus$, $\D$
and vectors in $\R^\dimension$ alone, as the code does. Two notes speak of an action that "carries
the basis at $\point$ to the basis at" its image: [the quotient](../../math/quotient-manifold/)
and [the chart volume](../../math/chart-volume/). That is the matrix statement
$\D(g\,\cdot)(\point) = I_\dimension$ under [](eq:derivative), and at publish they restate it that
way. The regularity they need becomes "$\oplus$ is smooth". A rule
goes into `authoring.md`, under "The math notes", so that new notes keep to it: past the first
note, the geometry is written in tangent coordinates.

**The first note gets longer, and reads quickly.** It adds four short sections of recalled
definitions, each a paragraph citing Lee, and proves only what is new: the coordinate form of the
retraction's conditions, the tangent charts, and the derivative through the retraction.

**References to keep true.** At publish:

- `eq:math/retraction/oplus` and `roundtrip` keep their tags in the first note. The heading
  `## The definition`, cited from `archive/core/math` as `math/retraction/#the-definition`, is kept
  or the reference is amended, since the build does not check anchors.
- `eq:math/derivatives-in-charts/derivative`, cited from `archive/core/math`,
  `archive/core/implementation` and `archive/rotations/statements/first-order`, moves to
  `eq:math/retraction/derivative`, and those references are amended and listed under `amends`.
- `statement:math/derivatives-in-charts/chain-rule` moves to the first note, and the proof of the
  passes cites it there.
- `derivatives-in-charts` keeps its slug and its place, geometry 3, retitled *Gradients and
  Hessians in tangent charts*. Each note that lists it in `depends_on` is reread. One that
  used it for the derivative alone depends on `retraction` instead. `quotient-manifold`, which
  cites the chart Hessian, keeps it.
- The notes after the first that rely on the old wording are reread: `product-manifold` (its
  stacking of tangent spaces), `quotient-manifold` and `chart-volume` (the action on bases,
  restated as $\D(g\,\cdot)(\point) = I$, and the smoothness of $\oplus$ that their densities need),
  and `rotations` (its retraction, smooth everywhere).
- `\basis` becomes a site-wide macro in `site/katex.config.ts`, and Absil, Mahony and Sepulchre
  enter `references.bib`.

**No code changes.** The library already works this way, so the entry changes only the notes, and
the pages that link to them, `concepts/spaces` and `concepts/queries`, need at most new link text.

## Alternatives

**Keep the order, and assume a smooth basis.** This is today's note. It defines a retraction with
a derivative not yet defined, and its assumption is false on the sphere, which it then has to
concede.

**A chart at each point.** One could fix a chart $\varphi_\point$ centred at each point and define
$\D\map(\point)$ as the Jacobian at $0$ of $\psi_{\map(\point)} \circ \map \circ \varphi_\point^{-1}$.
That gives the same derivatives as [](eq:basis): only the chart's differential at its centre is ever
used, and that differential is a basis. A chart carries second-order information that nothing reads,
and suggests that the second-order notions are fixed by it, when they depend on how the bases vary
from point to point. The basis is exactly the data used.

**A retraction defined in coordinates.** One could skip $\T\manifold$ and define a retraction
as a smooth $\oplus : \manifold \times \R^\dimension \to \manifold$. That requires a basis smooth
everywhere, which only parallelizable manifolds have, and so rules out the sphere. It also hides
the fact that the retraction and the basis are separate choices, a fact the positive reals' example
shows.

**The exponential map of a metric.** A Riemannian structure gives a canonical retraction and
orthonormal bases. The notes have avoided a metric so far, and nothing here needs one: a retraction
is only required to agree with the exponential map to first order.

**Tangent spaces as subspaces of an ambient space.** For a manifold embedded in $\R^n$,
$\T_\point\manifold$ is a subspace of $\R^n$ and needs no derivations. Not every space of the notes
is embedded as it is stored (the rotations are stored up to the sign of a quaternion), and the ambient
coordinates have $n$ entries rather than $\dimension$, while every tangent in skoro is a vector of
the space's dimension.
