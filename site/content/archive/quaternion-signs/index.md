---
title: "Observed quaternions of arbitrary sign on outside data"
pages: []
crates: []
revises: [rotations, core, measures]
---

The type `Rotation` is the unit quaternions, a double cover of the rotations, and the normal on it
tells $q$ from $-q$: [rotations](../rotations/) asks the caller to give an observed quaternion the
sign of its prediction before it is observed (`host::align`). One given with the other sign sits
beside the one point the chart cannot reach, where its residual grows without bound; nothing
refuses it, and no optimizer step can cross to the other sign. Both examples draw their
measurements from the model, so their signs are right by construction, and the path has never been
run on outside data. This entry observes quaternions of arbitrary sign, among them a loop closure
given with the opposite sign to its prediction, on a standard g2o dataset if loading one is cheap
enough. It makes the types the groups themselves, $SO(3)$ and its motions stored as quaternions up
to sign under the Cayley chart, which cannot tell $q$ from $-q$, so that no observed quaternion is
aligned; and it teaches the core to declare a type stored up to a finite group.
