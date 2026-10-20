`skoro` is the crate a user depends on. It re-exports everything `skoro-core` makes public, and
`skoro-tensor` as `skoro::tensor`, with `Device`, `Precision` and `Target` at its root, so a model,
a query and a target are all written against one crate.

It also holds the tests that may use the public interface alone, as a later crate would: the
acceptance test of the mathematics, a sphere declared as an embedded type, generation and
retirement, and a normal distribution private to the tests. One of its tests checks that none of
the others names `skoro-core` or `skoro-tensor` directly.
