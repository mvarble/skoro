---
title: "Smooth maps before retractions"
pages: []
crates: []
revises: [mathematics-in-order]
---

The math notes define a retraction before they say what a derivative on a manifold is, yet a
retraction is defined by its derivative at the origin, so the definition already assumes a
derivative in some chart. This entry reorders the foundation so that nothing is used before it is
defined. The first note, retitled "Smooth manifolds, smooth maps, and retractions", takes over the
derivative and the chain rule from "Derivatives in tangent charts", which keeps the gradient and the
chart Hessian as "Gradients and Hessians in tangent charts". It cites Lee for the general theory and
defines smooth maps through their coordinate representations, then recalls tangent vectors,
differentials and the tangent bundle. It fixes a basis of each tangent space by a convention that
need not be smooth, which turns each differential into a matrix, and only after that defines a
retraction, on the tangent bundle and without reference to any basis. Because a retraction's derivative at the origin is the
identity, the inverse function theorem makes its local inverses charts. The chain rule then shows
that derivatives taken in those charts agree with the ones the bases give.
