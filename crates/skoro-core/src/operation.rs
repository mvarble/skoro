use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use skoro_tensor::{Frame, Target, Value};

use crate::{Error, Space};

/// A map between spaces, described once and on no device, and lowered to a kernel when a query
/// is compiled.
pub trait Operation: Debug + Send + Sync + 'static {
    /// The registered identifier, such as `"skoro.add"`.
    fn name(&self) -> &'static str;
    /// The plain parameters which, with the name, let a registry make the operation again.
    fn params(&self) -> Vec<Param>;
    /// Checks the inputs' spaces and gives the output's. Called when the node is added.
    fn output(&self, inputs: &[Space]) -> Result<Space, Error>;
    /// What computes this operation on `lowering`'s target.
    fn lower(&self, inputs: &[Space], lowering: &mut Lowering<'_>) -> Result<Lowered, Error>;
}

/// A parameter of an operation, fixed when it is made; never an input.
///
/// Parameters are compared, and hashed, by value, a float by its bits, since they name what is
/// stored type-erased.
#[derive(Clone, Debug)]
pub enum Param {
    Integer(i64),
    Float(f64),
    Text(Arc<str>),
    Space(Space),
    List(Vec<Param>),
}

impl PartialEq for Param {
    fn eq(&self, other: &Param) -> bool {
        match (self, other) {
            (Param::Integer(a), Param::Integer(b)) => a == b,
            (Param::Float(a), Param::Float(b)) => a.to_bits() == b.to_bits(),
            (Param::Text(a), Param::Text(b)) => a == b,
            (Param::Space(a), Param::Space(b)) => a == b,
            (Param::List(a), Param::List(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for Param {}

impl Hash for Param {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Param::Integer(a) => a.hash(state),
            Param::Float(a) => a.to_bits().hash(state),
            Param::Text(a) => a.hash(state),
            Param::Space(a) => a.hash(state),
            Param::List(a) => a.hash(state),
        }
    }
}

/// What computes an operation on one target.
#[derive(Clone)]
pub enum Lowered {
    Kernel(Kernel),
}

/// What an operation is lowered with: the target.
pub struct Lowering<'a> {
    target: &'a Target,
}

impl<'a> Lowering<'a> {
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the testing checks and the compiler lower, in later tasks"
        )
    )]
    pub(crate) fn new(target: &'a Target) -> Lowering<'a> {
        Lowering { target }
    }

    pub fn target(&self) -> &Target {
        self.target
    }
}

/// An operation on one target. The variant is its order.
#[derive(Clone)]
pub enum Kernel {
    Zero(Arc<dyn Forward>),
    One(Arc<dyn FirstOrder>),
    Two(Arc<dyn SecondOrder>),
}

/// How far a kernel differentiates.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Order {
    Zero,
    One,
    Two,
}

pub trait Forward: Send + Sync + 'static {
    fn forward(&self, inputs: &[Value]) -> Value;
}

pub trait FirstOrder: Forward {
    /// The value and the tangent frame of the output. A missing frame is zero.
    fn tangent(&self, inputs: &[Value], tangents: &[Option<Frame>]) -> (Value, Option<Frame>);
    /// The value and the pull-back at these inputs, into the inputs marked differentiable.
    fn record(&self, inputs: &[Value], differentiable: &[bool]) -> (Value, Box<dyn Pullback>);
}

pub trait SecondOrder: FirstOrder {
    /// As `record`, with the inputs' velocities, and returning the output's.
    fn record_second(
        &self,
        inputs: &[Value],
        differentiable: &[bool],
        velocities: &[Option<Frame>],
    ) -> (Value, Option<Frame>, Box<dyn SecondPullback>);
}

pub trait Pullback: Send {
    /// One frame per input: `None` for an input that is not differentiable.
    fn cotangent(&self, alpha: &Frame) -> Vec<Option<Frame>>;
}

pub trait SecondPullback: Pullback {
    /// `hvp_j(alpha)` per input, at the velocities recorded: `alpha` has one column, each result
    /// as many columns as the velocities.
    fn second(&self, alpha: &Frame) -> Vec<Option<Frame>>;
}

impl Kernel {
    pub fn order(&self) -> Order {
        match self {
            Kernel::Zero(_) => Order::Zero,
            Kernel::One(_) => Order::One,
            Kernel::Two(_) => Order::Two,
        }
    }

    /// Every kernel's forward map, by upcasting a `One` or a `Two`.
    pub fn forward(&self) -> &dyn Forward {
        match self {
            Kernel::Zero(kernel) => kernel.as_ref(),
            Kernel::One(kernel) => kernel.as_ref(),
            Kernel::Two(kernel) => kernel.as_ref(),
        }
    }

    /// The first-order maps of a `One`, and of a `Two` by upcasting.
    pub fn first(&self) -> Option<&dyn FirstOrder> {
        match self {
            Kernel::Zero(_) => None,
            Kernel::One(kernel) => Some(kernel.as_ref()),
            Kernel::Two(kernel) => Some(kernel.as_ref()),
        }
    }

    pub fn second(&self) -> Option<&dyn SecondOrder> {
        match self {
            Kernel::Two(kernel) => Some(kernel.as_ref()),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use skoro_tensor::{Points, Target};

    use super::*;

    /// Doubles its one input: the smallest kernel of order two.
    struct Double;

    struct DoublePullback;

    fn double(value: &Value) -> Value {
        let Value::Continuous(points) = value else {
            unreachable!("the test gives a continuous value")
        };
        let target = Target::host();
        let doubled = points.clone().into_burn().mul_scalar(2.0);
        Value::Continuous(Points::from_burn(doubled, &target).unwrap())
    }

    fn double_frame(frame: &Frame) -> Frame {
        Frame::from_burn(frame.clone().into_burn().mul_scalar(2.0), &Target::host()).unwrap()
    }

    impl Forward for Double {
        fn forward(&self, inputs: &[Value]) -> Value {
            double(&inputs[0])
        }
    }

    impl FirstOrder for Double {
        fn tangent(&self, inputs: &[Value], tangents: &[Option<Frame>]) -> (Value, Option<Frame>) {
            (double(&inputs[0]), tangents[0].as_ref().map(double_frame))
        }

        fn record(&self, inputs: &[Value], _: &[bool]) -> (Value, Box<dyn Pullback>) {
            (double(&inputs[0]), Box::new(DoublePullback))
        }
    }

    impl SecondOrder for Double {
        fn record_second(
            &self,
            inputs: &[Value],
            _: &[bool],
            velocities: &[Option<Frame>],
        ) -> (Value, Option<Frame>, Box<dyn SecondPullback>) {
            let velocity = velocities[0].as_ref().map(double_frame);
            (double(&inputs[0]), velocity, Box::new(DoublePullback))
        }
    }

    impl Pullback for DoublePullback {
        fn cotangent(&self, alpha: &Frame) -> Vec<Option<Frame>> {
            vec![Some(double_frame(alpha))]
        }
    }

    impl SecondPullback for DoublePullback {
        fn second(&self, _: &Frame) -> Vec<Option<Frame>> {
            vec![None]
        }
    }

    #[test]
    fn kernels_reach_their_maps_by_upcasting() {
        let target = Target::host();
        let x = Value::Continuous(Points::from_host(&[1.5, -2.0], 1, 2, &target));
        let kernels = [
            Kernel::Zero(Arc::new(Double)),
            Kernel::One(Arc::new(Double)),
            Kernel::Two(Arc::new(Double)),
        ];
        let orders: Vec<Order> = kernels.iter().map(Kernel::order).collect();
        assert_eq!(orders, [Order::Zero, Order::One, Order::Two]);
        assert!(Order::Zero < Order::One && Order::One < Order::Two);
        for kernel in &kernels {
            let Value::Continuous(y) = kernel.forward().forward(std::slice::from_ref(&x)) else {
                unreachable!("a continuous value")
            };
            assert_eq!(y.to_host(), [3.0, -4.0]);
            assert_eq!(kernel.first().is_some(), kernel.order() >= Order::One);
            assert_eq!(kernel.second().is_some(), kernel.order() == Order::Two);
        }
        assert_eq!(Lowering::new(&target).target(), &target);
        let alpha = Frame::from_host(&[1.0, 1.0], 1, 2, 1, &target);
        let first = kernels[2].first().unwrap();
        let (_, pullback) = first.record(std::slice::from_ref(&x), &[true]);
        assert_eq!(
            pullback.cotangent(&alpha)[0].as_ref().unwrap().to_host(),
            [2.0, 2.0]
        );
    }

    #[test]
    fn params_compare_floats_by_bits() {
        use std::collections::HashSet;
        let params = [
            Param::Float(0.0),
            Param::Float(-0.0),
            Param::Float(f64::NAN),
            Param::Integer(0),
            Param::List(vec![Param::Space(Space::Real), Param::Text("a".into())]),
        ];
        let set: HashSet<Param> = params.iter().cloned().collect();
        assert_eq!(set.len(), params.len());
        assert_eq!(Param::Float(f64::NAN), Param::Float(f64::NAN));
        assert_ne!(Param::Float(0.0), Param::Float(-0.0));
    }
}
