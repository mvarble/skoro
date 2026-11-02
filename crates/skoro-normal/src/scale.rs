//! The scale of a normal, in one of four forms, each read through its whitening `W`.

/// Which of the four forms of the scale a value is, read through `W` in every one.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Form {
    /// `σ ∈ ℝ>0`, `Σ = σ²I`: a `Positive(1)`.
    Isotropic,
    /// `σ ∈ ℝ>0^d`, `Σ = diag(σ)²`: a `Positive(d)`.
    Diagonal,
    /// `L` with `Σ = LLᵀ`: a `triangular(d)`.
    CovarianceFactor,
    /// `W` with `Λ = WᵀW`: a `triangular(d)`.
    Whitening,
}

impl Form {
    /// The form's name, as composites and samplers record it.
    pub fn name(&self) -> &'static str {
        match self {
            Form::Isotropic => "isotropic",
            Form::Diagonal => "diagonal",
            Form::CovarianceFactor => "covariance_factor",
            Form::Whitening => "whitening",
        }
    }
}
