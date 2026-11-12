`skoro-spatial` gives skoro its [rotations and rigid motions](../../../concepts/rotations-and-motions/),
a crate built on the core's public interface alone. It declares three embedded types, the groups
themselves: `Rotation`, the rotations $SO(3)$; `Pose`, a rotation and a position; and
`ExtendedPose`, a rotation, a velocity and a position. Each is stored as a unit quaternion and its
translations, up to the sign of the quaternion, which the core checks it carries; each has the
Cayley chart in the rotation, global on the group and the same for either sign, moves the
translations in the point's own frame, and declares its volume against Haar measure, so that a
quantity of any of them is scored by the normal, observed in whatever sign it is given, and
simulated like any other.

Its maps are composites of the core's operations, with no kernel and no derivative of their own,
and each is a map of the groups, the same whichever sign its inputs are stored in: `*` composes and
acts on vectors; the extension traits `RotationMaps`, `PoseMaps` and `ExtendedPoseMaps` give the
inverses, relative motions, matrices, adjoints, logarithms and interpolations of the shorter turn,
and the parts of a handle; and the free functions of `rotation`, `pose`, `extended_pose` and `so3`
make them from rotation vectors and twists, axes and angles, pairs of directions, Euler angles and
matrices, with the Jacobians of the exponentials and their inverses. The removable singularities at
the identity are computed by series below a cutoff, one for every precision and device, high enough
for a GPU's sine. Euler angles, in all twelve conventions, are compared on the circle by
`angle_residual`. `host` does the same conversions in double precision for the values a model is
given, composes and applies rigid motions as the graph does, for a start computed from the
measurements, and chooses the quaternion with $w \ge 0$ to report.
