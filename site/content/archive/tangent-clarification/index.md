---
title: "Smooth maps before retractions"
pages: []
crates: []
revises: [mathematics-in-order]
---

The math notes define a retraction before they say what a derivative on a manifold is, yet a
retraction is defined by its derivative at the origin, so the definition already assumes a
derivative in some chart. This entry reorders the foundation so that nothing is used before it is
defined. A single note, "Smooth manifolds, smooth maps, and retractions", replaces "Retractions
and tangent charts" and "Derivatives in tangent charts". It cites Lee for the general theory and
defines smooth maps through their coordinate representations. It then fixes a chart at each point,
which identifies each tangent space with $\mathbb{R}^d$ and each differential with a matrix, and
only after that defines a retraction. Because a retraction's derivative at the origin is the
identity, the inverse function theorem makes its local inverses charts. The chain rule then shows
that derivatives taken in those charts agree with the ones the fixed coordinates give.
