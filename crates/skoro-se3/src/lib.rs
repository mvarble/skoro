//! # skoro-se3
//!
//! The SE(3) rigid-body transform group as a Skoro manifold ([`quantity::SE3`],
//! a rotation + translation). [`ops`] provides its differentiable retraction and
//! inverse retraction, and the pose operations: `compose`, `invert`, and `transform` /
//! `untransform` of points. All are differentiated in the body-frame chart
//! `(r, p) ⊕ (a, b) = (r Exp(a), p + r b)`, on the arithmetic of `skoro_so3::ops::body`.

pub mod ops;
pub mod quantity;
