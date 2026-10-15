#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! One test per row of the tables of primitives, each a call of `check_operation` at random
//! inputs, at both precisions.

use skoro_core::primitive::{
    Add, Concatenate, Exp, Log, MatMul, Mul, Neg, Scale, Slice, Sub, Sum, Transpose,
};
use skoro_core::testing::{check_operation, random};
use skoro_core::{Operation, Space};
use skoro_tensor::{HostValue, Target};

const ROWS: usize = 3;

/// A value of `space` per row, uniform in (-1, 1), drawn from `seed`.
fn input(space: Space, seed: u64) -> (Space, HostValue) {
    let value = random(&space, ROWS, seed, -1.0, 1.0);
    (space, value)
}

fn check(operation: &dyn Operation, inputs: &[(Space, HostValue)], target: &Target) {
    if let Err(error) = check_operation(operation, inputs, target) {
        panic!("{error}");
    }
}

skoro_core::precision_tests! {
    fn add(target) {
        let inputs = [input(Space::Matrix(2, 3), 1), input(Space::Matrix(2, 3), 2)];
        check(&Add, &inputs, &target);
    }

    fn sub(target) {
        let inputs = [input(Space::Vector(4), 3), input(Space::Vector(4), 4)];
        check(&Sub, &inputs, &target);
    }

    fn neg(target) {
        check(&Neg, &[input(Space::Vector(3), 5)], &target);
    }

    fn mul(target) {
        let inputs = [input(Space::Vector(4), 6), input(Space::Vector(4), 7)];
        check(&Mul, &inputs, &target);
    }

    fn scale(target) {
        let inputs = [input(Space::Real, 8), input(Space::Matrix(3, 2), 9)];
        check(&Scale, &inputs, &target);
    }

    fn matmul_matrix(target) {
        let inputs = [input(Space::Matrix(2, 3), 10), input(Space::Matrix(3, 4), 11)];
        check(&MatMul, &inputs, &target);
    }

    fn matmul_vector(target) {
        let inputs = [input(Space::Matrix(3, 2), 12), input(Space::Vector(2), 13)];
        check(&MatMul, &inputs, &target);
    }

    fn sum(target) {
        check(&Sum, &[input(Space::Matrix(2, 2), 14)], &target);
    }

    fn exp(target) {
        check(&Exp, &[input(Space::Vector(3), 15)], &target);
    }

    fn log(target) {
        let space = Space::Vector(3);
        let value = random(&space, ROWS, 16, 0.5, 2.0);
        check(&Log, &[(space, value)], &target);
    }

    fn slice(target) {
        let euclidean = Slice::new(Space::Vector(2), 1..3, 1..3);
        check(&euclidean, &[input(Space::Vector(4), 17)], &target);

        let product = Space::product([Space::Real, Space::Positive(2), Space::Vector(2)]).unwrap();
        let value = random(&product, ROWS, 18, 0.5, 2.0);
        let factor = Slice::new(Space::Positive(2), 1..3, 1..3);
        check(&factor, &[(product.clone(), value.clone())], &target);
        let run = Space::product([Space::Positive(2), Space::Vector(2)]).unwrap();
        check(&Slice::new(run, 1..5, 1..5), &[(product, value)], &target);
    }

    fn concatenate(target) {
        let inputs = [
            input(Space::Vector(2), 19),
            input(Space::Real, 20),
            input(Space::Matrix(1, 2), 21),
        ];
        check(&Concatenate::new(Space::Vector(5)), &inputs, &target);

        let product = Space::product([Space::Vector(2), Space::Positive(1)]).unwrap();
        let positive = random(&Space::Positive(1), ROWS, 22, 0.5, 2.0);
        let inputs = [input(Space::Vector(2), 23), (Space::Positive(1), positive)];
        check(&Concatenate::new(product), &inputs, &target);
    }

    fn reshape(target) {
        let inputs = [input(Space::Vector(6), 24)];
        check(&Concatenate::new(Space::Matrix(2, 3)), &inputs, &target);
    }

    fn transpose(target) {
        check(&Transpose, &[input(Space::Matrix(2, 3), 25)], &target);
    }
}

#[test]
fn spaces_are_checked_when_the_node_is_added() {
    use skoro_core::Error;
    let shape = Add.output(&[Space::Vector(2), Space::Vector(3)]);
    assert!(
        matches!(shape, Err(Error::ShapeMismatch { .. })),
        "{shape:?}"
    );
    let space = Add.output(&[Space::Vector(2), Space::Real]);
    assert!(
        matches!(space, Err(Error::SpaceMismatch { .. })),
        "{space:?}"
    );
    let inner = MatMul.output(&[Space::Matrix(2, 3), Space::Vector(2)]);
    assert!(
        matches!(inner, Err(Error::ShapeMismatch { .. })),
        "{inner:?}"
    );
    let product = Space::product([Space::Real, Space::Vector(2)]).unwrap();
    let misaligned = Slice::new(Space::Vector(2), 0..2, 0..2).output(&[product]);
    assert!(
        matches!(misaligned, Err(Error::SpaceMismatch { .. })),
        "{misaligned:?}"
    );
    let short = Concatenate::new(Space::Vector(4)).output(&[Space::Vector(3)]);
    assert!(
        matches!(short, Err(Error::ShapeMismatch { .. })),
        "{short:?}"
    );
}
