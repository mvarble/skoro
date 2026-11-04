---
title: "Accuracy on the GPU"
pages: []
crates: []
revises: [rotations, measures]
---

Three tests that pass in single precision on the CPU fail on a wgpu device, where CI, having no
adapter, cannot see them. Two are of [the rotations](../rotations/): the second-order pass of the
pose's left Jacobian, `pose::jl`, misses double precision on the host by $1.4 \cdot 10^{-4}$ against
the $10^{-4}$ allowed, in `series_meet_closed_forms` and `composites_pass_check_operation`. The
third is of [the core's measures](../measures/): the second-order map of the exponential test
sphere's log-volume at $t = 0$ misses central differences by $0.07$ against $5 \cdot 10^{-3}$, in
`exponential_sphere_passes_check_volume`, and fails on `main` as it stood before the rotations were
merged. Each is a removable singularity computed by a series below a cutoff and a closed form above
it, and the cutoffs were measured on the CPU alone; the suspected cause is the GPU's transcendental
functions, which WGSL holds to a looser accuracy than the CPU's single precision. This entry
measures what the GPU actually gets wrong, works out what that does to each closed form near its
cutoff, and changes the cutoffs, the series, the formulas or the tolerances so that every test
passes on wgpu as on the CPU.
