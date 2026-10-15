#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! One test per row of the tables of primitives, each a call of `check_operation` at random
//! inputs, at both precisions.

use skoro_core::primitive::{
    Add, Atan2, Concatenate, Cos, Exp, Log, MatMul, Mul, Neg, PositiveAsReal, PositiveExp,
    PositiveLog, PositiveMul, PositiveRecip, Scale, Select, Sin, Slice, Sub, Sum, Transpose,
};
use skoro_core::testing::{check_operation, lower, random};
use skoro_core::{Operation, Space};
use skoro_tensor::{Frame, HostData, HostValue, Target, Value};

const ROWS: usize = 3;

/// A value of `space` per row, uniform in (-1, 1), drawn from `seed`.
fn input(space: Space, seed: u64) -> (Space, HostValue) {
    let value = random(&space, ROWS, seed, -1.0, 1.0);
    (space, value)
}

/// A positive value of `space` per row, in (0.5, 2), drawn from `seed`.
fn positive(space: Space, seed: u64) -> (Space, HostValue) {
    let value = random(&space, ROWS, seed, 0.5, 2.0);
    (space, value)
}

/// Points `(a, b)` off the branch cut of `atan2`: `|a| ≥ 0.3`, so that no central difference
/// crosses the half-line `a = 0, b < 0`, and `b` of either sign.
fn off_the_cut(seed: u64) -> [(Space, HostValue); 2] {
    let space = Space::Vector(4);
    let (_, a) = input(space.clone(), seed);
    let HostData::Continuous(a) = a.data().clone() else {
        unreachable!("a continuous value")
    };
    let a = a.iter().map(|a| a + 0.3 * a.signum()).collect();
    let a = HostValue::new(Some(ROWS), 4, HostData::Continuous(a)).unwrap();
    [(space.clone(), a), input(space, seed + 1)]
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

    fn sin(target) {
        check(&Sin, &[input(Space::Vector(3), 26)], &target);
    }

    fn cos(target) {
        check(&Cos, &[input(Space::Matrix(2, 2), 27)], &target);
    }

    fn atan2(target) {
        check(&Atan2, &off_the_cut(28), &target);
    }

    fn atan2_is_checked_off_its_branch_cut(target) {
        check(&Atan2, &off_the_cut(30), &target);
        // Across the cut, at b = -1 and a = ±1e-3, the angle jumps by 2π less the twice
        // 1e-3 radians it turns on either side.
        let space = Space::Real;
        let value = |x: f64| HostValue::new(None, 1, HostData::Continuous(vec![x])).unwrap();
        let kernel = lower(&Atan2, &[space.clone(), space], &target).unwrap();
        let angle = |a: f64| {
            let inputs = [value(a), value(-1.0)].map(|v| v.load(1, &target).unwrap().unwrap());
            let Value::Continuous(y) = kernel.forward().forward(&inputs) else {
                unreachable!("a continuous value")
            };
            y.to_host()[0]
        };
        let jump = angle(1e-3) - angle(-1e-3);
        let expected = 2.0 * std::f64::consts::PI - 2.0 * (1e-3f64).atan2(1.0);
        assert!((jump - expected).abs() < 1e-5, "{jump} against {expected}");
    }

    fn select(target) {
        let index = (Space::Categorical(3), random(&Space::Categorical(3), ROWS, 40, 0.0, 0.0));
        let values = [40, 41, 42].map(|seed| input(Space::Vector(2), seed + 1));
        let inputs: Vec<_> = std::iter::once(index.clone()).chain(values).collect();
        check(&Select, &inputs, &target);

        let product = Space::product([Space::Real, Space::Positive(1)]).unwrap();
        let values = [44, 45, 46].map(|seed| positive(product.clone(), seed));
        let inputs: Vec<_> = std::iter::once(index).chain(values).collect();
        check(&Select, &inputs, &target);
    }

    fn select_routes_the_cotangent_to_the_chosen_value(target) {
        let spaces = [Space::Categorical(2), Space::Vector(2), Space::Vector(2)];
        let kernel = lower(&Select, &spaces, &target).unwrap();
        let load = |value: HostValue| value.load(4, &target).unwrap().unwrap();
        let index = HostValue::new(Some(4), 1, HostData::Discrete(vec![0, 1, 1, 0])).unwrap();
        let x = |v: f64| HostValue::new(None, 2, HostData::Continuous(vec![v, v])).unwrap();
        let inputs = [load(index), load(x(1.0)), load(x(2.0))];
        let alpha: Vec<f64> = (1..=8).map(f64::from).collect();
        let alpha = Frame::from_host(&alpha, 4, 2, 1, &target);
        let first = kernel.first().unwrap();
        let (_, pullback) = first.record(&inputs, &[false, true, true]);
        let cotangents = pullback.cotangent(&alpha);
        assert!(cotangents[0].is_none(), "the index receives nothing");
        let rows = |frame: &Option<Frame>| frame.as_ref().unwrap().to_host();
        assert_eq!(rows(&cotangents[1]), [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 7.0, 8.0]);
        assert_eq!(rows(&cotangents[2]), [0.0, 0.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0]);
    }

    fn positive_as_real(target) {
        check(&PositiveAsReal, &[positive(Space::Positive(3), 50)], &target);
    }

    fn positive_exp(target) {
        check(&PositiveExp, &[input(Space::Vector(3), 51)], &target);
        check(&PositiveExp, &[input(Space::Real, 52)], &target);
    }

    fn positive_log(target) {
        check(&PositiveLog, &[positive(Space::Positive(3), 53)], &target);
    }

    fn positive_mul(target) {
        let inputs = [positive(Space::Positive(2), 54), positive(Space::Positive(2), 55)];
        check(&PositiveMul, &inputs, &target);
    }

    fn positive_recip(target) {
        check(&PositiveRecip, &[positive(Space::Positive(2), 56)], &target);
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
