`skoro-tensor` is the one place skoro meets burn. It holds the tensor shapes the rest of skoro
computes with: `Points`, a batch of continuous values laid out as rows of a space's storage;
`Frame`, a batch of tangent or cotangent frames, `[rows, dimension, columns]`; `Integers`, a batch of
discrete values; and `Value`, either of the first and last. Each is made for a `Target`, a device and
a precision checked once when the target is made: burn's Flex device on the CPU, or a wgpu device,
in single or double precision, never one wrapped for burn's own automatic differentiation. Every
float tensor in a wrapper has its target's precision by construction, since nothing makes one
without naming the target.

It also holds `HostValue`, a value on the host: what a model stores for a constant and what a query
observes, either shared by every row of a batch or given row by row, and loaded onto a target when a
program runs. The crate re-exports burn's tensor module, for the kernels a later crate writes for an
operation of its own.

The point of the layer is containment. A burn upgrade touches this crate and the kernels, and
nothing else; and the precision a computation runs at is chosen by each query's
[target](../../../concepts/queries/#a-target), not fixed by the model.
