//! What the model knows about its least-squares factors, as runtime bookkeeping.
//!
//! An inference algorithm has preconditions — Levenberg-Marquardt needs the negative
//! log-density to be a sum of squared residuals — and something has to carry those facts
//! from where they are known to where they are needed. That "somewhere" is never the built
//! graph: a [`Distribution`](crate::distribution::Distribution) is consumed by value when a
//! quantity is added and what remains records nothing identifying it, so **"was this factor
//! a `Normal`?" has no answer after the model is built** unless someone wrote it down.
//!
//! This is where it is written down. A builder that knows it is assembling a whitened
//! residual records a [`FactorRecord`]; the [`ResidualRegistry`] holds them in the order
//! they were recorded, which is the row order of `J`.
//!
//! ## What is checked, and when
//!
//! The record is validated **at record time, against the graph as it stands** — the density
//! must be a live likelihood term, the residual must be a continuous node in the arena, and
//! every id the record mentions must exist. See
//! [`ModelServer::record_factor`](crate::model::ModelServer::record_factor).
//!
//! What is *not* checked is the mathematics: nothing verifies that the recorded density
//! really is `−½‖r‖² + n`, or that `normalizer_parents` is the whole of what the normalizer
//! depends on. Those remain the recorder's obligation. Recording is an assertion about a
//! factor, made by whoever built it.
//!
//! ## Retiring a factor
//!
//! A record naming a deleted node is worse than no record —
//! [`residual_layout`](crate::model::ModelServer::residual_layout) would resolve it through
//! `TangentLayout::from_ids` and fail. So the registry lives in the
//! [`ModelServer`](crate::model::ModelServer), and every mutation that can invalidate a
//! record prunes it inside the same locked call, against two questions asked of the graph:
//! is the residual in the arena, and is the density still a live likelihood term. The second
//! is the one arena membership cannot answer — demotion orphans a density without removing
//! it, and an orphaned density scores nothing, so a record kept on `contains` alone would
//! hand `J` rows for a term the objective no longer holds.

use std::collections::HashSet;

use skoro_core::id::Id;

/// What a builder recorded about one factor, at the moment it built it.
///
/// [`location`](Self::location) and [`whitening`](Self::whitening) are optional because not
/// every least-squares factor has them separately reachable. They are not Gaussian-specific:
/// they are the two halves of `r = W (x ⊖ μ)` that a filter reads apart, since `H = ∂μ/∂x`
/// and `R` are different objects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FactorRecord {
    /// The log-density node.
    pub density: Id,
    /// The whitened-residual node — a row block of `J`.
    pub residual: Id,
    /// The quantities the normalizer `n` depends on.
    ///
    /// Minimizing `½Σ‖rᵢ‖²` is minimizing `−log p` exactly when nothing named here is being
    /// optimized, which is what makes this worth recording separately from the factor's
    /// parents: a node's parent list is positional and untyped, so there is no way to ask it
    /// "which parent is the covariance".
    pub normalizer_parents: Vec<Id>,
    /// `μ` of `r = W (x ⊖ μ)`, when the factor has one. `H = ∂μ/∂x` is a Jacobian row block
    /// over this.
    pub location: Option<Id>,
    /// `W`'s source: the lower-triangular `L` with `R = L Lᵀ`, when the factor has one.
    pub whitening: Option<Id>,
}

impl FactorRecord {
    /// A record with no location/whitening refinement — the general least-squares case.
    pub fn new(density: Id, residual: Id, normalizer_parents: Vec<Id>) -> Self {
        Self {
            density,
            residual,
            normalizer_parents,
            location: None,
            whitening: None,
        }
    }

    /// Add the `μ` and `L` a filter reads apart.
    pub fn with_refinement(mut self, location: Id, whitening: Id) -> Self {
        self.location = Some(location);
        self.whitening = Some(whitening);
        self
    }

    /// Every id this record names as a node of the factor itself.
    ///
    /// Deliberately *not* the normalizer parents or the refinement: those are ordinary
    /// quantities a caller holds handles to, while the density and the residual are the two
    /// the record is responsible for keeping alive.
    fn named(&self) -> [Id; 2] {
        [self.density, self.residual]
    }
}

/// The factors a model has been told about, in the order they were recorded.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResidualRegistry {
    factors: Vec<FactorRecord>,
    /// Every node the records name, for O(1) [`names`](Self::names).
    ///
    /// Derived from `factors` and maintained beside every write to it, in this type only. A
    /// second owner would be a second thing to keep in step.
    named: HashSet<Id>,
}

impl ResidualRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether `id` is a node some record names — a recorded density or residual.
    ///
    /// Two callers, and they want it for opposite reasons: the collector must not *reclaim*
    /// such a node, and [`ModelServer::remove`](crate::model::ModelServer::remove) must not
    /// *delete* one. Both are the same question, so both ask it here.
    pub fn names(&self, id: Id) -> bool {
        self.named.contains(&id)
    }

    /// Record a factor.
    ///
    /// Private to the crate: the only public route is
    /// [`ModelServer::record_factor`](crate::model::ModelServer::record_factor), which
    /// validates the record against the graph first. A registry that could be populated with
    /// ids from nowhere would be a registry whose `names` set pins nothing real.
    pub(crate) fn record(&mut self, record: FactorRecord) {
        self.named.extend(record.named());
        self.factors.push(record);
    }

    /// The recorded factors, in the order they were recorded — the row blocks of `J`.
    pub fn factors(&self) -> &[FactorRecord] {
        &self.factors
    }

    /// The residual nodes, in the order they were recorded — the rows of `J`.
    pub fn residual_ids(&self) -> impl Iterator<Item = Id> + '_ {
        self.factors.iter().map(|f| f.residual)
    }

    /// The record naming `residual`, if any.
    pub fn factor_of(&self, residual: Id) -> Option<&FactorRecord> {
        self.factors.iter().find(|f| f.residual == residual)
    }

    /// Drop every record `keep` rejects, reporting each dropped record to `dropped`.
    ///
    /// **Order survives.** The rows of `J` are the factors in the order they were recorded,
    /// so the survivors must keep their relative positions; a caller reading a residual block
    /// by index would otherwise be reading a different factor's rows.
    ///
    /// `dropped` exists for one caller and one purpose: a record whose *residual* died while
    /// its density still scores leaves half a factor behind — a term the objective sums but
    /// `J` has no rows for — and `prune_registry` uses the callback to collect those densities
    /// and unmark them. It is **not** a hook for re-offering the dropped nodes to a collector.
    /// Losing a record un-pins its nodes, and the sweep that follows sees that as an ordinary
    /// question about reachability rather than as an event anyone must report.
    pub(crate) fn retain(
        &mut self,
        mut keep: impl FnMut(&FactorRecord) -> bool,
        mut dropped: impl FnMut(&FactorRecord),
    ) {
        let before = self.factors.len();
        self.factors.retain(|record| {
            let keep = keep(record);
            if !keep {
                dropped(record);
            }
            keep
        });
        if self.factors.len() != before {
            self.named = self.factors.iter().flat_map(FactorRecord::named).collect();
        }
    }
}
