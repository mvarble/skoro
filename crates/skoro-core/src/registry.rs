use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use crate::primitive::{
    Add, Atan2, Concatenate, Cos, Exp, IntegerAdd, IntegerEq, IntegerLe, IntegerLt, IntegerMul,
    IntegerSub, IsNonnegative, Log, MatMul, Mul, Neg, PositiveAsReal, PositiveExp, PositiveLog,
    PositiveMul, PositiveRecip, Scale, Select, Sin, Slice, Sub, Sum, ToReal, Transpose,
};
use crate::{Embedded, Error, Operation, Param};

/// Makes an operation from its parameters.
pub type MakeOperation = fn(&[Param]) -> Result<Arc<dyn Operation>, Error>;

/// Makes an embedded declaration from its parameters.
pub type MakeEmbedded = fn(&[Param]) -> Result<Arc<Embedded>, Error>;

/// How a name finds its code again: constructors by name. A value, not global state; a feature
/// crate exports a function that adds its names to one.
#[derive(Clone, Debug, Default)]
pub struct Registry {
    operations: HashMap<&'static str, MakeOperation>,
    embedded: HashMap<&'static str, MakeEmbedded>,
}

/// An operation of no parameters, made by its default.
fn plain<O: Operation + Default>(name: &'static str) -> (&'static str, MakeOperation) {
    fn make<O: Operation + Default>(params: &[Param]) -> Result<Arc<dyn Operation>, Error> {
        let operation = O::default();
        if params.is_empty() {
            Ok(Arc::new(operation))
        } else {
            Err(invalid(operation.name(), "it takes no parameters"))
        }
    }
    (name, make::<O>)
}

fn invalid(name: &str, reason: &str) -> Error {
    Error::InvalidDeclaration {
        name: name.to_string(),
        reason: reason.to_string(),
    }
}

fn range(param: &Param) -> Option<Range<usize>> {
    match param {
        Param::List(bounds) => match bounds.as_slice() {
            [Param::Integer(start), Param::Integer(end)] => {
                Some(usize::try_from(*start).ok()?..usize::try_from(*end).ok()?)
            }
            _ => None,
        },
        _ => None,
    }
}

fn slice(params: &[Param]) -> Result<Arc<dyn Operation>, Error> {
    match params {
        [Param::Space(output), storage, tangent] => match (range(storage), range(tangent)) {
            (Some(storage), Some(tangent)) => {
                Ok(Arc::new(Slice::new(output.clone(), storage, tangent)))
            }
            _ => Err(invalid(
                "skoro.slice",
                "its ranges are not two pairs of bounds",
            )),
        },
        _ => Err(invalid("skoro.slice", "it takes a space and two ranges")),
    }
}

fn concatenate(params: &[Param]) -> Result<Arc<dyn Operation>, Error> {
    match params {
        [Param::Space(output)] => Ok(Arc::new(Concatenate::new(output.clone()))),
        _ => Err(invalid("skoro.concatenate", "it takes a space")),
    }
}

impl Registry {
    /// An empty registry.
    pub fn new() -> Registry {
        Registry::default()
    }

    /// The core's operations.
    pub fn core() -> Registry {
        let operations: [(&'static str, MakeOperation); 29] = [
            plain::<Add>("skoro.add"),
            plain::<Sub>("skoro.sub"),
            plain::<Neg>("skoro.neg"),
            plain::<Mul>("skoro.mul"),
            plain::<Scale>("skoro.scale"),
            plain::<MatMul>("skoro.matmul"),
            plain::<Sum>("skoro.sum"),
            plain::<Exp>("skoro.exp"),
            plain::<Log>("skoro.log"),
            plain::<Sin>("skoro.sin"),
            plain::<Cos>("skoro.cos"),
            plain::<Atan2>("skoro.atan2"),
            plain::<Transpose>("skoro.transpose"),
            ("skoro.slice", slice),
            ("skoro.concatenate", concatenate),
            plain::<Select>("skoro.select"),
            plain::<PositiveAsReal>("skoro.positive.as_real"),
            plain::<PositiveExp>("skoro.positive.exp"),
            plain::<PositiveLog>("skoro.positive.log"),
            plain::<PositiveMul>("skoro.positive.mul"),
            plain::<PositiveRecip>("skoro.positive.recip"),
            plain::<IntegerAdd>("skoro.integer.add"),
            plain::<IntegerSub>("skoro.integer.sub"),
            plain::<IntegerMul>("skoro.integer.mul"),
            plain::<IntegerEq>("skoro.integer.eq"),
            plain::<IntegerLt>("skoro.integer.lt"),
            plain::<IntegerLe>("skoro.integer.le"),
            plain::<ToReal>("skoro.to_real"),
            plain::<IsNonnegative>("skoro.is_nonnegative"),
        ];
        Registry {
            operations: operations.into_iter().collect(),
            embedded: HashMap::new(),
        }
    }

    /// Adds `name`, refusing one already registered (`DuplicateName`).
    pub fn register_operation(
        &mut self,
        name: &'static str,
        make: MakeOperation,
    ) -> Result<(), Error> {
        if self.operations.contains_key(name) {
            return Err(Error::DuplicateName {
                name: name.to_string(),
            });
        }
        self.operations.insert(name, make);
        Ok(())
    }

    /// The operation registered as `name`, made from `params` (`UnknownOperation` for a name the
    /// registry lacks).
    pub fn operation(&self, name: &str, params: &[Param]) -> Result<Arc<dyn Operation>, Error> {
        let make = self
            .operations
            .get(name)
            .ok_or_else(|| Error::UnknownOperation {
                name: name.to_string(),
            })?;
        make(params)
    }

    /// Adds the embedded declaration `name`, refusing one already registered (`DuplicateName`).
    pub fn register_embedded(
        &mut self,
        name: &'static str,
        make: MakeEmbedded,
    ) -> Result<(), Error> {
        if self.embedded.contains_key(name) {
            return Err(Error::DuplicateName {
                name: name.to_string(),
            });
        }
        self.embedded.insert(name, make);
        Ok(())
    }

    /// The embedded declaration registered as `name`, made from `params` (`UnknownOperation` for
    /// a name the registry lacks).
    pub fn embedded(&self, name: &str, params: &[Param]) -> Result<Arc<Embedded>, Error> {
        let make = self
            .embedded
            .get(name)
            .ok_or_else(|| Error::UnknownOperation {
                name: name.to_string(),
            })?;
        make(params)
    }

    /// The names registered, in no particular order.
    pub fn names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.operations.keys().copied()
    }
}
