`skoro-normal` is skoro's [normal distribution](../../../concepts/the-normal/), a crate built on
the core's public interface alone. A `Normal` is a space, a mean and a scale; `score` adds the term
that scores a quantity, the Gaussian log-density of its tangent at the mean less the chart's
log-volume, with a sampler that draws it exactly by the space's own retraction, and binds the roles
an estimator reads: the whitened residual, the mean, the scale under the role of its form, the
log-volume, and the root of the expected information. It takes every continuous space whose chart
is global and whose volume is known, and refuses the others with the core's own errors; on those,
`factor_at` holds a measurement as a likelihood factor over data.

The scale is given in four forms, a standard deviation, a vector of them, a covariance factor or a
whitening, from quantities or from constants that the crate factors on the host; the composites
that read it are defined once per form and dimension. `Stacked` builds the stacked tangent of a
retirement's blanket at constant anchors, and the Gaussian replacement on it, which binds its
residual, mean and scale under the normal's roles whether it is added as a factor or given to a
retirement as its `Replacement`.
