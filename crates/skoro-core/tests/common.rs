use skoro_backend::{Backend, Element, Int, Tensor};
use skoro_core::{
    discrete::{Boolean, Categorical, Integer},
    kind_erased_tensor::KindErasedTensor,
    pack::{Cardinality, Packable},
};

pub fn constant_int<B: Backend>() -> KindErasedTensor<B, 1> {
    Tensor::<B, 1, Int>::from_ints([0], &Default::default()).into()
}

pub fn constant_float<B: Backend>() -> KindErasedTensor<B, 1> {
    Tensor::<B, 1>::from_floats([0.0], &Default::default()).into()
}

pub fn constant_values<B: Backend>() {
    let value = constant_int::<B>();
    assert!(value.is_discrete());
    let _ = value.unwrap_int();
    let value = constant_float::<B>();
    assert!(!value.is_discrete());
    let _ = value.unwrap_float();
}

pub fn poorly_unwrap_float<B: Backend>() {
    constant_int::<B>().unwrap_float();
}

pub fn poorly_unwrap_int<B: Backend>() {
    constant_float::<B>().unwrap_int();
}

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
#[should_panic(expected = "Cannot unpack boolean data 3 not in {0, 1}.")]
fn poor_bool_unpacking() {
    Boolean.unpack(&[0, 1, 3, 0], 2);
}

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
fn cardinalities() {
    assert_eq!(Boolean.cardinality(), Some(2));
    assert_eq!(Categorical(5).cardinality(), Some(5));
    assert_eq!(Integer.cardinality(), None);
}

#[test]
fn bool_packing() {
    bool_packing_impl::<i8>();
    bool_packing_impl::<i16>();
    bool_packing_impl::<i32>();
    bool_packing_impl::<i64>();
}

#[test]
fn categorical_packing() {
    categorical_packing_impl::<i8>();
    categorical_packing_impl::<i16>();
    categorical_packing_impl::<i32>();
    categorical_packing_impl::<i64>();
}

#[test]
fn integer_packing() {
    integer_packing_impl::<i8>();
    integer_packing_impl::<i16>();
    integer_packing_impl::<i32>();
    integer_packing_impl::<i64>();
}

#[macro_export]
macro_rules! tests {
    ($B:ident) => {
        #[test]
        fn constant_values() {
            common::constant_values::<B>();
        }

        #[test]
        #[should_panic(expected = "expected a continuous (float) tensor, found a discrete one")]
        fn unwrap_float_fails() {
            common::poorly_unwrap_float::<B>();
        }

        #[test]
        #[should_panic(expected = "expected a discrete (int) tensor, found a continuous one")]
        fn unwrap_int_fails() {
            common::poorly_unwrap_int::<B>();
        }
    };
}
