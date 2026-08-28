//! Registering deterministic transformations of quantities as graph nodes.
//!
//! The graph accepts operations only through [`DynGraphOperation`], with the two adapters
//! [`DiffGraphOperation`] and [`DiscreteGraphOperation`] as the only implementors. Wrapping
//! at every call site would put an adapter in front of every kernel in the workspace, so it
//! happens here instead: the [`register_mapN`](register_map1) family takes a bare
//! [`DiffOperation`], the [`register_discrete_mapN`](register_discrete_map1) family takes a
//! bare [`DiscreteOperation`], and each wraps before crossing into the client.
//!
//! Two families rather than one trait with two blanket impls: `impl<Op: DiffOperation<B>>`
//! alongside an impl for any other type trips E0119, since nothing tells coherence that a
//! downstream crate will not implement `DiffOperation` for the other wrapper.

use skoro_backend::Backend;
use skoro_core::{error::SkoroError, kind::Kind};
use skoro_graph::{
    operation::{DiffGraphOperation, DiscreteGraphOperation},
    quantity::QuantityType,
};
use skoro_operation::{diff::DiffOperation, discrete::DiscreteOperation};

use skoro_core::error::SkoroResult;

use crate::q::Q;

/// Register a unary mapped quantity from a differentiable kernel.
pub fn register_map1<B, I0, O, K, Op>(
    quantity_type: O,
    op: Op,
    x0: &Q<B, I0>,
) -> SkoroResult<Q<B, O>>
where
    B: Backend,
    O: QuantityType<B, K>,
    K: Kind<B>,
    Op: DiffOperation<B>,
{
    x0.client
        .add_mapped(quantity_type, DiffGraphOperation(op), vec![x0.id])
}

/// Register a binary mapped quantity from a differentiable kernel.
pub fn register_map2<B, I0, I1, O, K, Op>(
    quantity_type: O,
    op: Op,
    inputs: (&Q<B, I0>, &Q<B, I1>),
) -> SkoroResult<Q<B, O>>
where
    B: Backend,
    O: QuantityType<B, K>,
    K: Kind<B>,
    Op: DiffOperation<B>,
{
    let (x0, x1) = inputs;
    if x0.client != x1.client {
        return Err(SkoroError::ModelMismatch {
            lhs: x0.client.server_id,
            rhs: x1.client.server_id,
        });
    }
    x0.client
        .add_mapped(quantity_type, DiffGraphOperation(op), vec![x0.id, x1.id])
}

/// Register a unary mapped quantity from a forward-only discrete kernel.
///
/// The discrete sibling of [`register_map1`]. A [`DiscreteOperation`] has no derivatives at
/// all, which is a property of the kernel rather than of the output type, so it arrives
/// through its own entry point rather than through a wrapper at the call site.
pub fn register_discrete_map1<B, I0, O, K, Op>(
    quantity_type: O,
    op: Op,
    x0: &Q<B, I0>,
) -> SkoroResult<Q<B, O>>
where
    B: Backend,
    O: QuantityType<B, K>,
    K: Kind<B>,
    Op: DiscreteOperation<B>,
{
    x0.client
        .add_mapped(quantity_type, DiscreteGraphOperation(op), vec![x0.id])
}

/// Register a binary mapped quantity from a forward-only discrete kernel.
///
/// See [`register_discrete_map1`].
pub fn register_discrete_map2<B, I0, I1, O, K, Op>(
    quantity_type: O,
    op: Op,
    inputs: (&Q<B, I0>, &Q<B, I1>),
) -> SkoroResult<Q<B, O>>
where
    B: Backend,
    O: QuantityType<B, K>,
    K: Kind<B>,
    Op: DiscreteOperation<B>,
{
    let (x0, x1) = inputs;
    if x0.client != x1.client {
        return Err(SkoroError::ModelMismatch {
            lhs: x0.client.server_id,
            rhs: x1.client.server_id,
        });
    }
    x0.client.add_mapped(
        quantity_type,
        DiscreteGraphOperation(op),
        vec![x0.id, x1.id],
    )
}

/// Register a mapped quantity over a **runtime-variadic** input list.
///
/// The `register_mapN` family fixes arity in the type system and lets each input have its own
/// quantity type; this trades that for an arity only known at run time, at the cost of one shared
/// input type. Every layer below is already variadic -- `MappedNode::parent_ids` is a `Vec<Id>`,
/// and `DiffOperation::forward` takes a `Vec<Input<B>>` -- so this closes the only fixed-arity
/// seam in the stack.
///
/// The invariant the macro-generated siblings enforce is enforced here too: every input must
/// belong to the same client.
///
/// An empty `inputs` names no client and no graph, so there is nothing to register against; it is
/// reported as [`SkoroError::ShapeMismatch`] rather than panicking.
pub fn register_map_slice<B, I, O, K, Op>(
    quantity_type: O,
    op: Op,
    inputs: &[&Q<B, I>],
) -> SkoroResult<Q<B, O>>
where
    B: Backend,
    O: QuantityType<B, K>,
    K: Kind<B>,
    Op: DiffOperation<B>,
{
    let Some(first) = inputs.first() else {
        return Err(SkoroError::ShapeMismatch {
            expected: 1,
            found: 0,
        });
    };
    for input in &inputs[1..] {
        if first.client != input.client {
            return Err(SkoroError::ModelMismatch {
                lhs: first.client.server_id,
                rhs: input.client.server_id,
            });
        }
    }
    first.client.add_mapped(
        quantity_type,
        DiffGraphOperation(op),
        inputs.iter().map(|q| q.id).collect(),
    )
}

macro_rules! impl_register_map {
    (
        // The name of the function (e.g., register_map3)
        $fn_name:ident,
        // The name of the tuple type alias to generate (e.g., QTuple3)
        $tuple_name:ident,
        // The first argument and its type (e.g., x0: I0)
        $first_arg:ident: $first_type:ident
        // The rest of the arguments, if any
        $(, $rest_arg:ident: $rest_type:ident)*
    ) => {
        // 1. Generate the type alias (e.g. pub type QTuple3<B, I0, I1, I2> = ...)
        // Note: The trailing comma ensures 1-element tuples compile correctly as (T,)
        pub type $tuple_name<'a, B, $first_type $(, $rest_type)*> = (
            &'a Q<B, $first_type>,
            $(&'a Q<B, $rest_type>,)*
        );

        // 2. Generate the function
        pub fn $fn_name<B, $first_type, $($rest_type,)* O, K, Op>(
            quantity_type: O,
            op: Op,
            inputs: $tuple_name<B, $first_type $(, $rest_type)*>,
        ) -> SkoroResult<Q<B, O>>
        where
            B: Backend,
            O: QuantityType<B, K>,
            K: Kind<B>,
            Op: DiffOperation<B>,
        {
            // Destructure the tuple back into x0, x1, x2...
            let ($first_arg, $($rest_arg,)*) = inputs;

            // Assert clients match the first argument's client
            $(
                if $first_arg.client != $rest_arg.client {
                    return Err(SkoroError::ModelMismatch {
                        lhs: $first_arg.client.server_id,
                        rhs: $rest_arg.client.server_id,
                    });
                }
            )*

            $first_arg.client.add_mapped(
                    quantity_type,
                    DiffGraphOperation(op),
                    vec![$first_arg.id $(, $rest_arg.id)*]
            )
        }
    };
}

// Now, invoke the macro to generate the type aliases AND the functions!

impl_register_map!(register_map3, QTuple3, x0: I0, x1: I1, x2: I2);
impl_register_map!(register_map4, QTuple4, x0: I0, x1: I1, x2: I2, x3: I3);
impl_register_map!(register_map5, QTuple5, x0: I0, x1: I1, x2: I2, x3: I3, x4: I4);
impl_register_map!(register_map6, QTuple6, x0: I0, x1: I1, x2: I2, x3: I3, x4: I4, x5: I5);
impl_register_map!(register_map7, QTuple7, x0: I0, x1: I1, x2: I2, x3: I3, x4: I4, x5: I5, x6: I6);
impl_register_map!(register_map8, QTuple8, x0: I0, x1: I1, x2: I2, x3: I3, x4: I4, x5: I5, x6: I6, x7: I7);
impl_register_map!(register_map9, QTuple9, x0: I0, x1: I1, x2: I2, x3: I3, x4: I4, x5: I5, x6: I6, x7: I7, x8: I8);
impl_register_map!(register_map10, QTuple10, x0: I0, x1: I1, x2: I2, x3: I3, x4: I4, x5: I5, x6: I6, x7: I7, x8: I8, x9: I9);
impl_register_map!(register_map11, QTuple11, x0: I0, x1: I1, x2: I2, x3: I3, x4: I4, x5: I5, x6: I6, x7: I7, x8: I8, x9: I9, x10: I10);
impl_register_map!(register_map12, QTuple12, x0: I0, x1: I1, x2: I2, x3: I3, x4: I4, x5: I5, x6: I6, x7: I7, x8: I8, x9: I9, x10: I10, x11: I11);
