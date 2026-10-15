use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

use crate::{NodeId, Space};

/// A marker: the static type of a handle.
pub trait Typed: Copy + Send + Sync + 'static {
    /// Whether a node of `space` may be handled as this marker.
    fn admits(space: &Space) -> bool;
}

/// A marker whose value fixes a space, so that a quantity can be made from it.
pub trait Declares: Typed {
    fn space(&self) -> Space;
}

/// A handle on a node: an identifier and a marker, cheap to copy and bound to no writer.
pub struct Q<T> {
    id: NodeId,
    marker: PhantomData<fn() -> T>,
}

/// A handle with no static type.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct QDyn {
    id: NodeId,
}

impl<T> Clone for Q<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Q<T> {}

impl<T> PartialEq for Q<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<T> Eq for Q<T> {}

impl<T> Hash for Q<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl<T> fmt::Debug for Q<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Q<{}>({:?})", std::any::type_name::<T>(), self.id)
    }
}

impl<T: Typed> Q<T> {
    pub(crate) fn from_id(id: NodeId) -> Q<T> {
        Q {
            id,
            marker: PhantomData,
        }
    }

    pub fn id(&self) -> NodeId {
        self.id
    }

    /// Whether the operation that made this handle failed, inside a scope or for want of one.
    pub fn is_failed(&self) -> bool {
        self.untyped().is_failed()
    }

    /// The same node, with no static type.
    pub fn untyped(self) -> QDyn {
        QDyn { id: self.id }
    }

    /// The same node handled as `U`, checked against its space in the open scope
    /// (`SpaceMismatch`).
    #[track_caller]
    pub fn narrow<U: Typed>(self) -> Q<U> {
        Q::from_id(crate::scope::narrow::<U>(self.id))
    }
}

impl<T: Typed> From<Q<T>> for QDyn {
    fn from(q: Q<T>) -> QDyn {
        q.untyped()
    }
}

impl QDyn {
    pub(crate) fn from_id(id: NodeId) -> QDyn {
        QDyn { id }
    }

    pub fn id(&self) -> NodeId {
        self.id
    }

    pub fn is_failed(&self) -> bool {
        self.id == NodeId::FAILED || self.id == NodeId::OUTSIDE_SCOPE
    }

    /// The same node handled as `U`, checked against its space in the open scope
    /// (`SpaceMismatch`).
    #[track_caller]
    pub fn narrow<U: Typed>(self) -> Q<U> {
        Q::from_id(crate::scope::narrow::<U>(self.id))
    }
}

macro_rules! marker {
    ($(#[$doc:meta])* $name:ident $(<$(const $n:ident: usize),+>)? $(($($field:ident),+))?,
     admits: $admits:pat $(if $guard:expr)?, space: $space:expr) => {
        $(#[$doc])*
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
        pub struct $name $(<$(const $n: usize),+>)? $(($(pub $field),+))?;

        impl $(<$(const $n: usize),+>)? Typed for $name $(<$($n),+>)? {
            fn admits(space: &Space) -> bool {
                matches!(space, $admits $(if $guard)?)
            }
        }

        impl $(<$(const $n: usize),+>)? Declares for $name $(<$($n),+>)? {
            #[allow(unused_variables)]
            fn space(&self) -> Space {
                let $name $(($($field),+))? = *self;
                $space
            }
        }
    };
}

marker!(
    /// A real number.
    Real, admits: Space::Real, space: Space::Real
);
marker!(
    /// A vector of `N` reals.
    Vector<const N: usize>, admits: Space::Vector(n) if *n == N, space: Space::Vector(N)
);
marker!(
    /// A matrix of `M` rows and `N` columns.
    Matrix<const M: usize, const N: usize>,
    admits: Space::Matrix(m, n) if (*m, *n) == (M, N), space: Space::Matrix(M, N)
);
marker!(
    /// `N` positive reals.
    Positive<const N: usize>, admits: Space::Positive(n) if *n == N, space: Space::Positive(N)
);
marker!(
    /// One of `N` categories.
    Categorical<const N: usize>,
    admits: Space::Categorical(n) if *n == N, space: Space::Categorical(N)
);
marker!(
    /// A truth value: `Categorical(2)`.
    Boolean, admits: Space::Categorical(2), space: Space::BOOLEAN
);
marker!(
    /// An integer.
    Integer, admits: Space::Integer, space: Space::Integer
);

/// A vector of any length, fixed by the value.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct VectorDyn(pub usize);
/// A matrix of any size, fixed by the value: rows, columns.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MatrixDyn(pub usize, pub usize);
/// Positive reals of any number, fixed by the value.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct PositiveDyn(pub usize);
/// Categories of any number, fixed by the value.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct CategoricalDyn(pub usize);
/// Any product: a product quantity is made from its `Space`, untyped.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct ProductDyn;

macro_rules! dyn_marker {
    ($name:ident ($($field:ident),+), admits: $admits:pat, space: $space:expr) => {
        impl Typed for $name {
            fn admits(space: &Space) -> bool {
                matches!(space, $admits)
            }
        }

        impl Declares for $name {
            fn space(&self) -> Space {
                let $name($($field),+) = *self;
                $space
            }
        }
    };
}

dyn_marker!(VectorDyn(n), admits: Space::Vector(_), space: Space::Vector(n));
dyn_marker!(MatrixDyn(m, n), admits: Space::Matrix(_, _), space: Space::Matrix(m, n));
dyn_marker!(PositiveDyn(n), admits: Space::Positive(_), space: Space::Positive(n));
dyn_marker!(CategoricalDyn(n), admits: Space::Categorical(_), space: Space::Categorical(n));

impl Typed for ProductDyn {
    fn admits(space: &Space) -> bool {
        matches!(space, Space::Product(_))
    }
}

impl<const N: usize> Q<Vector<N>> {
    /// The same node as a vector of any length: free, since it cannot fail.
    pub fn widen(self) -> Q<VectorDyn> {
        Q::from_id(self.id)
    }
}

impl<const M: usize, const N: usize> Q<Matrix<M, N>> {
    pub fn widen(self) -> Q<MatrixDyn> {
        Q::from_id(self.id)
    }
}

impl<const N: usize> Q<Positive<N>> {
    pub fn widen(self) -> Q<PositiveDyn> {
        Q::from_id(self.id)
    }
}

impl<const N: usize> Q<Categorical<N>> {
    pub fn widen(self) -> Q<CategoricalDyn> {
        Q::from_id(self.id)
    }
}

impl Q<Boolean> {
    pub fn widen(self) -> Q<CategoricalDyn> {
        Q::from_id(self.id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markers_admit_their_spaces() {
        assert!(Vector::<3>::admits(&Space::Vector(3)));
        assert!(!Vector::<3>::admits(&Space::Vector(2)));
        assert!(VectorDyn::admits(&Space::Vector(2)));
        assert!(!VectorDyn::admits(&Space::Real));
        assert!(Matrix::<2, 3>::admits(&Space::Matrix(2, 3)));
        assert!(!Matrix::<2, 3>::admits(&Space::Matrix(3, 2)));
        assert!(Boolean::admits(&Space::Categorical(2)));
        assert!(CategoricalDyn::admits(&Space::Categorical(7)));
        assert!(!Integer::admits(&Space::Categorical(2)));
        let product = Space::product([Space::Real, Space::Vector(2)]).unwrap();
        assert!(ProductDyn::admits(&product) && !ProductDyn::admits(&Space::Vector(3)));
    }

    #[test]
    fn markers_declare_their_spaces() {
        assert_eq!(Matrix::<2, 3>.space(), Space::Matrix(2, 3));
        assert_eq!(MatrixDyn(4, 1).space(), Space::Matrix(4, 1));
        assert_eq!(Boolean.space(), Space::Categorical(2));
        assert_eq!(PositiveDyn(5).space(), Space::Positive(5));
        for space in [Real.space(), Vector::<2>.space(), Categorical::<4>.space()] {
            assert!(space.storage() > 0);
        }
    }

    #[test]
    fn failed_handles_are_marked() {
        let failed = Q::<Real>::from_id(NodeId::FAILED);
        let outside = Q::<Real>::from_id(NodeId::OUTSIDE_SCOPE);
        assert!(failed.is_failed() && outside.is_failed());
        assert!(QDyn::from_id(NodeId::FAILED).is_failed());
        assert_ne!(failed, outside);
        let fresh = Q::<Vector<2>>::from_id(NodeId::fresh());
        assert!(!fresh.is_failed());
        assert_eq!(fresh.widen().id(), fresh.id());
        assert_eq!(QDyn::from(fresh), fresh.untyped());
    }
}
