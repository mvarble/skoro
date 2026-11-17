//! The uniform distribution of skoro: the law `ν / ν(𝕏)` of a space whose reference measure `ν`
//! has finite mass, with no parameters.
//!
//! Such a space is `Categorical(n)`, whose counting measure has mass `n`; a curved type that
//! declares its mass and its uniform draw, as the rotations do; and a product of such types. On
//! every other space, the Euclidean ones and the poses among them, the measure has infinite mass
//! or the type declares none, and a uniform distribution is refused.
//!
//! Its term is the constant `-log ν(𝕏)`, so it changes no mode and no step; its sampler draws a
//! uniform point exactly, from the variates of each row's key; and it binds the residual of
//! [`skoro_normal::roles`] to a vector of no components, so that a linearization reads it as a
//! factor with no rows. It is how a model says that nothing is known of a quantity, which a
//! normal of a large scale on a curved type does not say.

mod refuse;
mod sample;
mod uniform;

pub use sample::UniformSample;
pub use uniform::Uniform;
