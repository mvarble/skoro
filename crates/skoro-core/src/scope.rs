//! The building scope: the thread-local stack of frames, and the functions that build into the
//! innermost scope.
//!
//! Every building function does four things. With no scope open it returns a handle marked as
//! made outside a scope. Given a failed handle it returns one, recording `OutsideScope` or
//! `FailedOperand` if the draft has no failure yet. It reads its inputs' spaces from the draft
//! (`UnknownNode`) and asks the operation for its output space. And on an error it records the
//! error and the caller's line, if the draft has none yet, and returns a failed handle; otherwise
//! it inserts the node, folds it if its inputs are fixed in the model, and returns its handle.

use std::cell::RefCell;
use std::panic::Location;
use std::sync::Arc;

use skoro_tensor::{HostData, HostValue, Target, Value};

pub use crate::ops::{concatenate, product, select};

use crate::model::{Model, Node, NodeKind};
use crate::{
    Address, Declares, Error, Failure, IntoHostValue, Lowered, Lowering, NodeId, Operation, Q,
    QDyn, Space, Typed,
};

/// A frame of the thread's stack.
pub(crate) enum Frame {
    /// A building scope, with its draft.
    Scope(Draft),
    /// A namespace, prefixing the addresses of the quantities made in it.
    Namespace(Address),
}

/// The model a scope builds, and the first failure it recorded.
pub(crate) struct Draft {
    pub(crate) model: Model,
    pub(crate) failure: Option<Failure>,
}

thread_local! {
    static STACK: RefCell<Vec<Frame>> = const { RefCell::new(Vec::new()) };
}

/// Pops the frames down to its depth when dropped, on return or on unwinding.
struct Guard {
    depth: usize,
}

impl Drop for Guard {
    fn drop(&mut self) {
        // Frames dropped here hold no handle on the stack, so dropping them cannot re-enter it.
        let dropped = STACK.with(|stack| stack.borrow_mut().split_off(self.depth));
        drop(dropped);
    }
}

/// Runs `body` with `frame` pushed, and gives the frame back when it returns.
pub(crate) fn open<R>(frame: Frame, body: impl FnOnce() -> R) -> (R, Frame) {
    let depth = STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        stack.push(frame);
        stack.len() - 1
    });
    let guard = Guard { depth };
    let result = body();
    let mut frames = STACK.with(|stack| stack.borrow_mut().split_off(depth));
    std::mem::forget(guard);
    if frames.is_empty() {
        unreachable!("a frame is popped only by the guard of the call that pushed it");
    }
    (result, frames.swap_remove(0))
}

/// The number of frames open on this thread.
#[doc(hidden)]
pub fn depth() -> usize {
    STACK.with(|stack| stack.borrow().len())
}

impl Draft {
    pub(crate) fn new(model: Model) -> Draft {
        Draft {
            model,
            failure: None,
        }
    }

    fn record(&mut self, error: Error, location: &'static Location<'static>) {
        if self.failure.is_none() {
            self.failure = Some(Failure { error, location });
        }
    }

    /// The spaces of `inputs`, refusing a failed handle and one the draft does not hold.
    pub(crate) fn spaces(&self, inputs: &[QDyn]) -> Result<Vec<Space>, Error> {
        if inputs.iter().any(|q| q.id() == NodeId::OUTSIDE_SCOPE) {
            return Err(Error::OutsideScope);
        }
        if inputs.iter().any(QDyn::is_failed) {
            return Err(Error::FailedOperand);
        }
        inputs
            .iter()
            .map(|q| {
                self.model
                    .nodes
                    .get(&q.id())
                    .map(|node| node.space().clone())
                    .ok_or(Error::UnknownNode { id: q.id() })
            })
            .collect()
    }

    pub(crate) fn mapped(
        &mut self,
        operation: Arc<dyn Operation>,
        inputs: &[QDyn],
    ) -> Result<NodeId, Error> {
        let spaces = self.spaces(inputs)?;
        let space = operation.output(&spaces)?;
        let inputs: Box<[NodeId]> = inputs.iter().map(QDyn::id).collect();
        let folded = self.fold(operation.as_ref(), &inputs, &spaces)?;
        let id = NodeId::fresh();
        self.model.insert(Node::new(
            id,
            space,
            NodeKind::Mapped {
                operation,
                inputs,
                folded,
            },
        ));
        Ok(id)
    }

    /// The value of a node whose inputs are all fixed in the model, computed once on the host in
    /// double precision; `None` when an input is not fixed.
    fn fold(
        &self,
        operation: &dyn Operation,
        inputs: &[NodeId],
        spaces: &[Space],
    ) -> Result<Option<HostValue>, Error> {
        let values: Option<Vec<&HostValue>> = inputs
            .iter()
            .map(|id| self.model.nodes.get(id).and_then(|node| node.value()))
            .collect();
        let Some(values) = values else {
            return Ok(None);
        };
        let mut pinned = values.iter().filter_map(|value| value.rows());
        let batch = pinned.next();
        if let Some(found) = pinned.find(|&rows| Some(rows) != batch) {
            let expected = batch.unwrap_or(1);
            return Err(Error::BatchSize { expected, found });
        }
        let target = Target::host();
        let Lowered::Kernel(kernel) = operation.lower(spaces, &mut Lowering::new(&target))?;
        let rows = batch.unwrap_or(1);
        let loaded = values
            .iter()
            .map(|value| {
                value.load(rows, &target)?.ok_or(Error::BatchSize {
                    expected: rows,
                    found: value.rows().unwrap_or(1),
                })
            })
            .collect::<Result<Vec<Value>, Error>>()?;
        let output = kernel.forward().forward(&loaded);
        Ok(Some(host_value(&output, batch)))
    }

    fn quantity(&mut self, address: Address, space: Space) -> Result<NodeId, Error> {
        if self.model.addresses.contains_key(&address) {
            return Err(Error::AddressTaken { address });
        }
        let id = NodeId::fresh();
        self.model
            .insert(Node::new(id, space, NodeKind::Quantity { address }));
        Ok(id)
    }

    pub(crate) fn constant(
        &mut self,
        space: Space,
        value: Option<HostValue>,
    ) -> Result<NodeId, Error> {
        let value = value
            .filter(|value| fits(&space, value))
            .ok_or_else(|| Error::ValueShape {
                space: space.clone(),
            })?;
        let id = NodeId::fresh();
        self.model
            .insert(Node::new(id, space, NodeKind::Constant { value }));
        Ok(id)
    }
}

/// Whether `value` is a value of `space`: its storage, its kind, a category in range, and a
/// positive value positive.
pub(crate) fn fits(space: &Space, value: &HostValue) -> bool {
    if value.storage() != space.storage() {
        return false;
    }
    match (space, value.data()) {
        (Space::Categorical(n), HostData::Discrete(data)) => data
            .iter()
            .all(|&k| usize::try_from(k).is_ok_and(|k| k < *n)),
        (Space::Integer, HostData::Discrete(_)) => true,
        (Space::Positive(_), HostData::Continuous(data)) => data.iter().all(|&x| x > 0.0),
        (space, HostData::Continuous(_)) => space.is_continuous(),
        _ => false,
    }
}

/// A value read back to the host: shared when the batch was not pinned, one per row otherwise.
fn host_value(value: &Value, batch: Option<usize>) -> HostValue {
    let (storage, data) = match value {
        Value::Continuous(points) => (points.storage(), HostData::Continuous(points.to_host())),
        Value::Discrete(integers) => (integers.storage(), HostData::Discrete(integers.to_host())),
    };
    match HostValue::new(batch, storage, data) {
        Some(value) => value,
        None => unreachable!("a kernel's output holds its batch of rows"),
    }
}

/// Runs `step` on the innermost scope's draft, with the prefix of the namespaces opened inside
/// it, and gives the identifier of what it built: the mark of a handle made outside a scope when
/// none is open, and of a failed one when `step` fails, recording the failure at `location`.
pub(crate) fn build(
    location: &'static Location<'static>,
    step: impl FnOnce(&mut Draft, &Address) -> Result<NodeId, Error>,
) -> NodeId {
    STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        let Some(at) = stack
            .iter()
            .rposition(|frame| matches!(frame, Frame::Scope(_)))
        else {
            return NodeId::OUTSIDE_SCOPE;
        };
        let (scope, namespaces) = stack.split_at_mut(at + 1);
        let prefix = namespaces
            .iter()
            .fold(Address::root(), |prefix, frame| match frame {
                Frame::Namespace(namespace) => prefix.join(namespace),
                Frame::Scope(_) => prefix,
            });
        let Some(Frame::Scope(draft)) = scope.last_mut() else {
            unreachable!("the frame found is a scope")
        };
        match step(draft, &prefix) {
            Ok(id) => id,
            Err(error) => {
                draft.record(error, location);
                NodeId::FAILED
            }
        }
    })
}

/// A quantity of `marker`'s space at `address`, under the open namespaces (`AddressTaken`).
#[track_caller]
pub fn quantity<T: Declares>(address: impl Into<Address>, marker: T) -> Q<T> {
    let (address, space) = (address.into(), marker.space());
    let location = Location::caller();
    Q::from_id(build(location, |draft, prefix| {
        draft.quantity(prefix.join(&address), space)
    }))
}

/// A quantity of `space` at `address`, under the open namespaces (`AddressTaken`).
#[track_caller]
pub fn quantity_in(address: impl Into<Address>, space: Space) -> QDyn {
    let address = address.into();
    let location = Location::caller();
    QDyn::from_id(build(location, |draft, prefix| {
        draft.quantity(prefix.join(&address), space)
    }))
}

/// A constant of `marker`'s space (`ValueShape` when the value does not fit it).
#[track_caller]
pub fn constant<T: Declares>(marker: T, value: impl IntoHostValue<T>) -> Q<T> {
    let (space, value) = (marker.space(), value.into_host_value());
    let location = Location::caller();
    Q::from_id(build(location, |draft, _| draft.constant(space, value)))
}

/// Runs `body` with `prefix` prefixed to the addresses of the quantities it makes and looks up.
pub fn namespace<R>(prefix: impl Into<Address>, body: impl FnOnce() -> R) -> R {
    open(Frame::Namespace(prefix.into()), body).0
}

/// The quantity at `address` under the open namespaces (`UnknownAddress`), handled as `T`
/// (`SpaceMismatch` when its space is not `T`'s).
#[track_caller]
pub fn lookup<T: Typed>(address: impl Into<Address>) -> Q<T> {
    let address = address.into();
    let location = Location::caller();
    Q::from_id(build(location, |draft, prefix| {
        let address = prefix.join(&address);
        let id = *draft
            .model
            .addresses
            .get(&address)
            .ok_or(Error::UnknownAddress { address })?;
        admit::<T>(draft, id, "lookup")
    }))
}

/// A node of `operation` applied to `inputs`.
#[track_caller]
pub fn apply(operation: Arc<dyn Operation>, inputs: &[QDyn]) -> QDyn {
    let location = Location::caller();
    QDyn::from_id(build(location, |draft, _| draft.mapped(operation, inputs)))
}

/// The node `id` handled as `T`, checked against its space in the draft.
#[track_caller]
pub(crate) fn narrow<T: Typed>(id: NodeId) -> NodeId {
    let location = Location::caller();
    build(location, |draft, _| admit::<T>(draft, id, "narrow"))
}

fn admit<T: Typed>(draft: &Draft, id: NodeId, operation: &'static str) -> Result<NodeId, Error> {
    let space = draft.spaces(&[QDyn::from_id(id)])?.remove(0);
    if T::admits(&space) {
        Ok(id)
    } else {
        Err(Error::SpaceMismatch {
            operation,
            inputs: vec![space],
        })
    }
}

/// A literal: a constant of `space` whose every entry is `c` (`ValueShape` for a discrete space,
/// or a positive one and `c` not positive).
pub(crate) fn literal(draft: &mut Draft, space: Space, c: f64) -> Result<NodeId, Error> {
    let storage = space.storage();
    let value = space
        .is_continuous()
        .then(|| HostValue::new(None, storage, HostData::Continuous(vec![c; storage])))
        .flatten();
    draft.constant(space, value)
}
