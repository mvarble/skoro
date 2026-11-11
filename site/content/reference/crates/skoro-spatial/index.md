`skoro-spatial` gives skoro its [rotations and rigid motions](../../../concepts/rotations-and-motions/),
a crate built on the core's public interface alone. It declares three embedded types: `Rotation`,
the unit quaternions, a double cover of the rotations; `Pose`, a rotation and a position; and
`ExtendedPose`, a rotation, a velocity and a position. Each has a global chart, stereographic in the
rotation and moving the translations in the point's own frame, and a declared volume against Haar
measure, so that a quantity of any of them is scored by the normal, observed and simulated like any
other.

Its maps are composites of the core's operations, with no kernel and no derivative of their own:
`*` composes and acts on vectors; the extension traits `RotationMaps`, `PoseMaps` and
`ExtendedPoseMaps` give the inverses, relative motions, matrices, adjoints, logarithms,
interpolations and parts of a handle; and the free functions of `rotation`, `pose`,
`extended_pose` and `so3` make them from rotation vectors and twists, axes and angles, pairs of
directions, Euler angles and matrices, with the Jacobians of the exponentials and their inverses.
The removable singularities at the identity are computed by series below a cutoff, one for every
precision and device, high enough for a GPU's sine. Euler angles, in all twelve conventions, are
compared on the circle by `angle_residual`. `host` does the same conversions in double precision
for the values a model is given, aligns an observed quaternion with its prediction, and composes
and applies rigid motions as the graph does, for a start computed from the measurements.
