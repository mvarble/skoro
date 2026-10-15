use std::fmt;
use std::panic::Location;
use std::sync::Arc;

use crate::model::{Model, Node, NodeKind};
use crate::scope::{self, Draft, Frame};
use crate::{Error, Failure, Lowered, Lowering, NodeId, Operation, Param, QDyn, Space};

/// A named subgraph with designated inputs and one output, built by the code a model would use.
pub struct Composite {
    name: &'static str,
    params: Vec<Param>,
    inputs: Vec<Space>,
    output: Space,
    body: Body,
}

/// A composite's nodes, the nodes standing for its inputs, and its output node.
pub(crate) struct Body {
    pub(crate) model: Model,
    pub(crate) inputs: Vec<NodeId>,
    pub(crate) output: NodeId,
}

impl Composite {
    /// Runs `body` with one handle per input; the operations inside build into the composite,
    /// and the handle `body` returns is its output. A quantity cannot be made there, nor looked
    /// up (`NotInAModel`), and a handle from outside is not one of the composite's (`UnknownNode`).
    #[track_caller]
    pub fn define(
        name: &'static str,
        params: Vec<Param>,
        inputs: &[Space],
        body: impl FnOnce(&[QDyn]) -> QDyn,
    ) -> Result<Arc<Composite>, Failure> {
        let location = Location::caller();
        let mut draft = Draft::definition(Model::new());
        let handles: Vec<QDyn> = inputs
            .iter()
            .map(|space| {
                let id = NodeId::fresh();
                let node = Node::new(id, space.clone(), NodeKind::Input);
                draft.model.insert(node);
                QDyn::from_id(id)
            })
            .collect();
        let (output, frame) = scope::open(Frame::Definition(draft), || body(&handles));
        let Frame::Definition(draft) = frame else {
            unreachable!("define gets back the definition it opened")
        };
        if let Some(failure) = draft.failure {
            return Err(failure);
        }
        let space = draft
            .model
            .nodes
            .get(&output.id())
            .map(|node| node.space().clone())
            .ok_or(Failure {
                error: Error::UnknownNode { id: output.id() },
                location,
            })?;
        Ok(Arc::new(Composite {
            name,
            params,
            inputs: inputs.to_vec(),
            output: space,
            body: Body {
                model: draft.model,
                inputs: handles.iter().map(QDyn::id).collect(),
                output: output.id(),
            },
        }))
    }

    /// Adds one node to the open scope: this composite applied to `inputs`.
    #[track_caller]
    pub fn call(self: &Arc<Self>, inputs: &[QDyn]) -> QDyn {
        scope::apply(Arc::new(Call(self.clone())), inputs)
    }

    /// The operation of a call of this composite: what `call` applies.
    pub fn operation(self: &Arc<Self>) -> Arc<dyn Operation> {
        Arc::new(Call(self.clone()))
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn params(&self) -> &[Param] {
        &self.params
    }

    pub fn inputs(&self) -> &[Space] {
        &self.inputs
    }

    pub fn output(&self) -> &Space {
        &self.output
    }

    #[allow(
        dead_code,
        reason = "the compiler expands a composite's body, in the next task"
    )]
    pub(crate) fn body(&self) -> &Body {
        &self.body
    }

    /// Whether `self` and `other` are the same composite: the identity by which everything stored
    /// type-erased is named, its name and its parameters.
    pub fn same(&self, other: &Composite) -> bool {
        self.name == other.name && self.params == other.params
    }
}

impl fmt::Debug for Composite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Composite")
            .field("name", &self.name)
            .field("params", &self.params)
            .field("inputs", &self.inputs)
            .field("output", &self.output)
            .finish_non_exhaustive()
    }
}

/// A call of a composite: the operation of the node `Composite::call` adds.
#[derive(Debug)]
struct Call(Arc<Composite>);

impl Operation for Call {
    fn name(&self) -> &'static str {
        self.0.name
    }

    fn params(&self) -> Vec<Param> {
        self.0.params.clone()
    }

    fn output(&self, inputs: &[Space]) -> Result<Space, Error> {
        if inputs == self.0.inputs.as_slice() {
            Ok(self.0.output.clone())
        } else {
            Err(Error::SpaceMismatch {
                operation: self.0.name,
                inputs: inputs.to_vec(),
            })
        }
    }

    fn lower(&self, inputs: &[Space], _: &mut Lowering<'_>) -> Result<Lowered, Error> {
        self.output(inputs)?;
        Ok(Lowered::Composite(self.0.clone()))
    }
}
