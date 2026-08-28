//! `Packable` and `Cardinality`: moving a discrete user type in and out of an element slice.
//!
//! One module per discrete type, because each owns its own domain check. The round trip is the
//! claim -- `unpack` after `pack` returns what went in -- and the `poor_*` tests are the other
//! half of it: a value outside the type's domain panics rather than being silently truncated
//! into range, which is what would make a corrupt sample look like a valid one.

use skoro_backend::Element;
use skoro_core::{
    discrete::{Boolean, Categorical, Integer},
    pack::{Cardinality, Packable},
};

/// `Boolean`: one element per value, `0` or `1`. The round trip is the claim; `poor_bool_unpacking`
/// is the other half of it, since an element outside `{0, 1}` has no boolean reading and must
/// panic rather than be truncated into one.
mod boolean {
    use super::*;

    fn bool_packing_impl<I>()
    where
        I: Element + PartialEq,
        Boolean: Packable<I, bool>,
    {
        let mut data = [0, 1, 1, 0, 1, 0].map(I::from_elem);
        let ty = Boolean;
        assert!(!ty.unpack(&data, 0));
        assert!(ty.unpack(&data, 1));
        assert!(ty.unpack(&data, 2));
        assert!(!ty.unpack(&data, 3));
        assert!(ty.unpack(&data, 4));
        assert!(!ty.unpack(&data, 5));
        ty.pack(&true, 0, &mut data);
        ty.pack(&false, 1, &mut data);
        ty.pack(&true, 2, &mut data);
        ty.pack(&false, 3, &mut data);
        ty.pack(&false, 4, &mut data);
        ty.pack(&false, 5, &mut data);
        assert_eq!(data, [1, 0, 1, 0, 0, 0].map(I::from_elem));
    }

    #[test]
    fn bool_packing() {
        bool_packing_impl::<i8>();
        bool_packing_impl::<i16>();
        bool_packing_impl::<i32>();
        bool_packing_impl::<i64>();
    }

    #[test]
    #[should_panic(expected = "Cannot unpack boolean data 3 not in {0, 1}.")]
    fn poor_bool_unpacking() {
        Boolean.unpack(&[0, 1, 3, 0], 2);
    }
}

/// `Categorical(n)`: one element per value, in `0..n`. The bound is checked on the way in *and*
/// on the way out -- a packed value above the maximum would otherwise unpack as a category the
/// caller never declared.
mod categorical {
    use super::*;

    fn categorical_packing_impl<I>()
    where
        I: Element + PartialEq,
        Categorical: Packable<I, usize>,
    {
        let mut data = [0, 3, 0, 2, 1, 3].map(I::from_elem);
        let ty = Categorical(4);
        assert_eq!(ty.unpack(&data, 0), 0);
        assert_eq!(ty.unpack(&data, 1), 3);
        assert_eq!(ty.unpack(&data, 2), 0);
        assert_eq!(ty.unpack(&data, 3), 2);
        assert_eq!(ty.unpack(&data, 4), 1);
        assert_eq!(ty.unpack(&data, 5), 3);
        ty.pack(&3, 0, &mut data);
        ty.pack(&3, 1, &mut data);
        ty.pack(&0, 2, &mut data);
        ty.pack(&0, 3, &mut data);
        ty.pack(&2, 4, &mut data);
        ty.pack(&1, 5, &mut data);
        assert_eq!(data, [3, 3, 0, 0, 2, 1].map(I::from_elem));
    }

    #[test]
    fn categorical_packing() {
        categorical_packing_impl::<i8>();
        categorical_packing_impl::<i16>();
        categorical_packing_impl::<i32>();
        categorical_packing_impl::<i64>();
    }

    #[test]
    #[should_panic(expected = "Cannot unpack categorical data 4 exceeding maximum 3.")]
    fn poor_categorical_unpacking() {
        Categorical(4).unpack(&[0, 1, 4, 0], 2);
    }

    #[test]
    #[should_panic(expected = "Cannot pack categorical data 4 exceeding maximum 3.")]
    fn poor_categorical_packing() {
        let mut data = [0i8, 1i8, 2i8];
        Categorical(4).pack(&4, 2, &mut data);
    }
}

/// `Integer`: one element per value, the whole element range, no domain check to fail. Run over
/// every integer element type the backends use, because the packing is generic in it.
mod integer {
    use super::*;

    fn integer_packing_impl<I>()
    where
        I: Element + PartialEq,
        Integer: Packable<I, I>,
    {
        let mut data = [0, -3, 0, 2, 1, 3].map(I::from_elem);
        let ty = Integer;
        assert_eq!(ty.unpack(&data, 0), I::from_elem(0));
        assert_eq!(ty.unpack(&data, 1), I::from_elem(-3));
        assert_eq!(ty.unpack(&data, 2), I::from_elem(0));
        assert_eq!(ty.unpack(&data, 3), I::from_elem(2));
        assert_eq!(ty.unpack(&data, 4), I::from_elem(1));
        assert_eq!(ty.unpack(&data, 5), I::from_elem(3));
        ty.pack(&I::from_elem(-3), 0, &mut data);
        ty.pack(&I::from_elem(-3), 1, &mut data);
        ty.pack(&I::from_elem(0), 2, &mut data);
        ty.pack(&I::from_elem(0), 3, &mut data);
        ty.pack(&I::from_elem(2), 4, &mut data);
        ty.pack(&I::from_elem(1), 5, &mut data);
        assert_eq!(
            data,
            [
                I::from_elem(-3),
                I::from_elem(-3),
                I::from_elem(0),
                I::from_elem(0),
                I::from_elem(2),
                I::from_elem(1)
            ]
            .map(I::from_elem)
        );
    }

    #[test]
    fn integer_packing() {
        integer_packing_impl::<i8>();
        integer_packing_impl::<i16>();
        integer_packing_impl::<i32>();
        integer_packing_impl::<i64>();
    }
}

/// Test cardinalities of different discrete quantity types.
#[test]
fn cardinalities() {
    assert_eq!(Boolean.cardinality(), Some(2));
    assert_eq!(Categorical(5).cardinality(), Some(5));
    assert_eq!(Integer.cardinality(), None);
}
