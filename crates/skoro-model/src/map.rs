use skoro_backend::Backend;
use skoro_core::{error::SkoroError, kind::Kind};
use skoro_graph::{operation::GraphOperation, quantity::QuantityType};

use skoro_core::error::SkoroResult;

use crate::q::Q;

/// Register a unary mapped quantity, continuous or discrete.
pub fn register_map1<B, I0, O, K, Op>(
    quantity_type: O,
    op: Op,
    x0: &Q<B, I0>,
) -> SkoroResult<Q<B, O>>
where
    B: Backend,
    O: QuantityType<B, K>,
    K: Kind<B>,
    Op: GraphOperation<B>,
{
    x0.guard.set_mapped_leaf(false);
    x0.client.add_mapped(quantity_type, op, vec![x0.id])
}

/// Register a binary mapped quantity, continuous or discrete.
pub fn register_map2<B, I0, I1, O, K, Op>(
    quantity_type: O,
    op: Op,
    inputs: (&Q<B, I0>, &Q<B, I1>),
) -> SkoroResult<Q<B, O>>
where
    B: Backend,
    O: QuantityType<B, K>,
    K: Kind<B>,
    Op: GraphOperation<B>,
{
    let (x0, x1) = inputs;
    if x0.client != x1.client {
        return Err(SkoroError::ModelMismatch {
            lhs: x0.client.server_id,
            rhs: x1.client.server_id,
        });
    }
    x0.guard.set_mapped_leaf(false);
    x1.guard.set_mapped_leaf(false);
    x0.client.add_mapped(quantity_type, op, vec![x0.id, x1.id])
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
            Op: GraphOperation<B>,
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

            $first_arg.guard.set_mapped_leaf(false);
            $($rest_arg.guard.set_mapped_leaf(false);)*
            $first_arg.client.add_mapped(
                    quantity_type,
                    op,
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
