//! The label on a binding: a reference edge's kind.
//!
//! A binding `(from, role, to)` means "`to` plays `role` for `from`". The edges themselves, and
//! the index over them, are graph state and live in `skoro-graph`; only the label lives here,
//! because [`SkoroError`](crate::error::SkoroError) names one — a consumer that requires a role
//! and finds a term without it reports which role was missing.
//!
//! The role constants themselves are declared where their contracts are documented, in
//! `skoro_model::roles`.

/// A labelled edge kind. Two roles compare equal iff their strings are identical.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Role(pub &'static str);

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}
