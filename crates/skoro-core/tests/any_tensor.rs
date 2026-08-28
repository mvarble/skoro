//! `AnyTensor` and `QuantityValue`: the shared tensor type for the two kinds, and the
//! batch-agnostic/batch-pinned distinction layered on top of it.
//!
//! `AnyTensor` is where the two element types stop being a compile-time fact and become a runtime
//! one. Its accessors come in two flavours on purpose: `as_*_opt` for asking, `unwrap_*` for
//! asserting. The second panics, because a caller that reached for the wrong kind has a bug the
//! value cannot repair.

use skoro_backend::{Int, NdArrayBackend, Tensor, TensorData};
use skoro_core::{
    any_tensor::{AnyTensor, AnyTensorMap, QuantityValue},
    error::SkoroError,
    id::IdGenerator,
};

type B = NdArrayBackend<f64, i32>;

fn floats<const D: usize>(data: Vec<f64>, dims: [usize; D]) -> Tensor<B, D> {
    Tensor::from_data(TensorData::new(data, dims), &Default::default())
}

fn ints<const D: usize>(data: Vec<i32>, dims: [usize; D]) -> Tensor<B, D, Int> {
    Tensor::from_data(TensorData::new(data, dims), &Default::default())
}

fn float2(data: Vec<f64>, batch: usize, components: usize) -> AnyTensor<B, 2> {
    AnyTensor::float(floats(data, [batch, components]))
}

fn int2(data: Vec<i32>, batch: usize, components: usize) -> AnyTensor<B, 2> {
    AnyTensor::int(ints(data, [batch, components]))
}

/// The continuous/discrete boundary: a value knowing its own kind, and refusing the other one
/// rather than reinterpreting its bytes. The two accessor families are the point -- `as_*_opt`
/// answers, `unwrap_*` asserts -- so each is checked in both directions.
mod discrimination {
    use super::*;

    #[test]
    fn a_value_knows_its_own_kind() {
        assert!(!float2(vec![1.0], 1, 1).is_discrete());
        assert!(int2(vec![1], 1, 1).is_discrete());
    }

    /// The asking accessors answer `None` for the other kind rather than panicking.
    #[test]
    fn the_optional_accessors_answer_rather_than_assert() {
        let float = float2(vec![1.0, 2.0], 1, 2);
        let int = int2(vec![3, 4], 1, 2);

        assert!(float.as_float_opt().is_some());
        assert!(float.as_int_opt().is_none());
        assert!(int.as_int_opt().is_some());
        assert!(int.as_float_opt().is_none());
    }

    /// The asserting accessors, in both borrowing and consuming forms.
    #[test]
    fn the_unwrapping_accessors_return_the_matching_kind() {
        let float = float2(vec![1.5], 1, 1);
        assert_eq!(float.unwrap_float_ref().dims(), [1, 1]);
        assert_eq!(
            float
                .clone()
                .unwrap_float()
                .into_data()
                .to_vec::<f64>()
                .unwrap(),
            vec![1.5]
        );

        let int = int2(vec![7], 1, 1);
        assert_eq!(int.unwrap_int_ref().dims(), [1, 1]);
        assert_eq!(
            int.clone()
                .unwrap_int()
                .into_data()
                .to_vec::<i32>()
                .unwrap(),
            vec![7]
        );
    }

    /// Four panics, one per wrong-kind accessor. Each names which kind it wanted, so a stack
    /// trace is enough to locate the mistake.
    mod wrong_kind_panics {
        use super::*;

        #[test]
        #[should_panic(expected = "expected a continuous (float) tensor, found a discrete one")]
        fn unwrap_float() {
            int2(vec![0], 1, 1).unwrap_float();
        }

        #[test]
        #[should_panic(expected = "expected a continuous (float) tensor, found a discrete one")]
        fn unwrap_float_ref() {
            let _ = int2(vec![0], 1, 1).unwrap_float_ref();
        }

        #[test]
        #[should_panic(expected = "expected a discrete (int) tensor, found a continuous one")]
        fn unwrap_int() {
            float2(vec![0.0], 1, 1).unwrap_int();
        }

        #[test]
        #[should_panic(expected = "expected a discrete (int) tensor, found a continuous one")]
        fn unwrap_int_ref() {
            let _ = float2(vec![0.0], 1, 1).unwrap_int_ref();
        }
    }

    /// Shape and device are readable without knowing the kind, which is what lets the compiler
    /// check batches uniformly.
    #[test]
    fn shape_and_device_are_kind_agnostic() {
        for value in [float2(vec![1.0, 2.0], 1, 2), int2(vec![1, 2], 1, 2)] {
            assert_eq!(value.dims(), [1, 2]);
            let _ = value.device();
        }
    }

    #[test]
    fn tensors_convert_in_at_any_rank() {
        let rank1: AnyTensor<B, 1> = floats(vec![1.0], [1]).into();
        assert!(!rank1.is_discrete());
        let rank1: AnyTensor<B, 1> = ints(vec![1], [1]).into();
        assert!(rank1.is_discrete());

        let rank3: AnyTensor<B, 3> = floats(vec![1.0], [1, 1, 1]).into();
        assert_eq!(rank3.dims(), [1, 1, 1]);
        assert!(format!("{rank3:?}").contains("Float"));

        let rank3: AnyTensor<B, 3> = ints(vec![1], [1, 1, 1]).into();
        assert!(format!("{rank3:?}").contains("Int"));
        assert!(!format!("{:?}", rank3.clone()).is_empty());
    }

    /// The by-id map alias is what a forward pass hands back.
    #[test]
    fn the_map_alias_keys_by_id() {
        let id = IdGenerator::new().issue_id();
        let mut map: AnyTensorMap<B, 2> = AnyTensorMap::new();
        map.insert(id, float2(vec![1.0], 1, 1));
        assert_eq!(map[&id].dims(), [1, 1]);
    }
}

/// The batch axis of an `AnyTensor`: what its dims report, and what stacking a rank-1 value up
/// to a batch does to them. A tensor that lost its batch axis here still has a plausible shape,
/// so the dims are asserted rather than only the values.
mod batching {
    use super::*;

    /// The single batch check in the workspace: everything that is checked rather than broadcast
    /// goes through it, so the three call sites cannot come to disagree about what matches.
    #[test]
    fn ensure_batch_accepts_a_match_and_names_a_mismatch() {
        let value = float2(vec![1.0, 2.0], 2, 1);
        assert_eq!(value.ensure_batch(2).unwrap().dims(), [2, 1]);
        assert_eq!(
            value.ensure_batch(3).err(),
            Some(SkoroError::BatchSizeMismatch {
                expected: 3,
                found: 2
            })
        );
    }

    /// Rank-1 storage expands to whatever batch a pass runs at, by repetition.
    #[test]
    fn unbatched_storage_expands_by_repetition() {
        let value: AnyTensor<B, 1> = floats(vec![1.0, 2.0], [2]).into();
        let expanded = value.expand(3);
        assert_eq!(expanded.dims(), [3, 2]);
        assert_eq!(
            expanded.unwrap_float().into_data().to_vec::<f64>().unwrap(),
            vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
        );

        let value: AnyTensor<B, 1> = ints(vec![5], [1]).into();
        assert_eq!(
            value
                .expand(2)
                .unwrap_int()
                .into_data()
                .to_vec::<i32>()
                .unwrap(),
            vec![5, 5]
        );
    }

    /// And rank-2 storage of a single row collapses back, which is how a fold that pinned
    /// nothing is stored batch-agnostically again.
    #[test]
    fn a_single_row_unbatches() {
        assert_eq!(
            float2(vec![1.0, 2.0], 1, 2)
                .unbatch()
                .unwrap_float()
                .into_data()
                .to_vec::<f64>()
                .unwrap(),
            vec![1.0, 2.0]
        );
        assert_eq!(
            int2(vec![3], 1, 1)
                .unbatch()
                .unwrap_int()
                .into_data()
                .to_vec::<i32>()
                .unwrap(),
            vec![3]
        );
    }

    /// Expanding and unbatching are inverse for a single row.
    #[test]
    fn expand_and_unbatch_round_trip() {
        let value: AnyTensor<B, 1> = floats(vec![1.0, 2.0], [2]).into();
        let round_tripped = value.expand(1).unbatch();
        assert_eq!(
            round_tripped
                .unwrap_float()
                .into_data()
                .to_vec::<f64>()
                .unwrap(),
            vec![1.0, 2.0]
        );
    }
}

/// `QuantityValue`, the batch-agnostic/batch-pinned distinction layered over `AnyTensor`.
/// Which half a value falls in is a property of the value, not of the call site: an `Unbatched`
/// is rank-1 storage the compiler will expand, a `Batched` is rank-2 storage it will check.
mod quantity_values {
    use super::*;

    /// The single place the two representations diverge: `Unbatched` expands, `Batched` checks.
    #[test]
    fn materializing_expands_or_checks_according_to_the_variant() {
        let shared = QuantityValue::Unbatched(floats(vec![7.0], [1]).into());
        assert_eq!(shared.materialize(4).unwrap().dims(), [4, 1]);
        assert_eq!(
            shared
                .materialize(4)
                .unwrap()
                .unwrap_float()
                .into_data()
                .to_vec::<f64>()
                .unwrap(),
            vec![7.0; 4]
        );

        let pinned = QuantityValue::Batched(float2(vec![1.0, 2.0, 3.0], 3, 1));
        assert_eq!(pinned.materialize(3).unwrap().dims(), [3, 1]);
    }

    /// A `Batched` value of batch 1 in a batch-`n` pass is an **error**, not a broadcast.
    ///
    /// Broadcasting is what `Unbatched` already is. A rule where batch 1 broadcasts but batch `n`
    /// does not is the failure mode where a flat manifold's `Addition` broadcasts happily and
    /// returns a silently wrong answer.
    #[test]
    fn a_pinned_value_never_broadcasts() {
        let single_row = QuantityValue::Batched(float2(vec![1.0], 1, 1));
        assert_eq!(
            single_row.materialize(3).err(),
            Some(SkoroError::BatchSizeMismatch {
                expected: 3,
                found: 1
            })
        );

        let three_rows = QuantityValue::Batched(float2(vec![1.0, 2.0, 3.0], 3, 1));
        assert_eq!(
            three_rows.materialize(1).err(),
            Some(SkoroError::BatchSizeMismatch {
                expected: 1,
                found: 3
            })
        );
    }

    /// `pinned_batch` is how a graph discovers what batch it is locked to.
    #[test]
    fn only_batched_values_pin() {
        let shared: QuantityValue<B> = QuantityValue::Unbatched(floats(vec![1.0], [1]).into());
        assert_eq!(shared.pinned_batch(), None);

        let pinned = QuantityValue::Batched(float2(vec![1.0, 2.0], 2, 1));
        assert_eq!(pinned.pinned_batch(), Some(2));
    }

    #[test]
    fn kind_and_width_are_readable_through_either_variant() {
        let unbatched_float: QuantityValue<B> =
            QuantityValue::Unbatched(floats(vec![1.0, 2.0, 3.0], [3]).into());
        assert!(!unbatched_float.is_discrete());
        assert_eq!(unbatched_float.num_components(), 3);

        let unbatched_int: QuantityValue<B> =
            QuantityValue::Unbatched(ints(vec![1, 2], [2]).into());
        assert!(unbatched_int.is_discrete());
        assert_eq!(unbatched_int.num_components(), 2);

        // For rank-2 storage the component axis is the *second* one, not the first.
        let batched_float = QuantityValue::Batched(float2(vec![1.0, 2.0, 3.0, 4.0], 2, 2));
        assert!(!batched_float.is_discrete());
        assert_eq!(batched_float.num_components(), 2);

        let batched_int = QuantityValue::Batched(int2(vec![1, 2, 3, 4], 2, 2));
        assert!(batched_int.is_discrete());
        assert_eq!(batched_int.num_components(), 2);
    }

    /// The two variants hold tensors of different rank, so there is no accessor that serves both
    /// -- a caller reaching for the value has to say which it expects.
    #[test]
    fn each_accessor_answers_only_for_its_own_variant() {
        let shared: QuantityValue<B> = QuantityValue::Unbatched(floats(vec![1.0], [1]).into());
        assert!(shared.as_unbatched().is_some());
        assert!(shared.as_batched().is_none());

        let pinned = QuantityValue::Batched(float2(vec![1.0], 1, 1));
        assert!(pinned.as_unbatched().is_none());
        assert!(pinned.as_batched().is_some());
    }

    /// The rank of the tensor decides the variant, which is what makes
    /// `add_constant` / `add_batched_constant` differ at the call site alone.
    #[test]
    fn rank_selects_the_variant_on_conversion() {
        let from_rank1: QuantityValue<B> = floats(vec![1.0], [1]).into();
        assert!(from_rank1.as_unbatched().is_some() && !from_rank1.is_discrete());

        let from_rank1_int: QuantityValue<B> = ints(vec![1], [1]).into();
        assert!(from_rank1_int.as_unbatched().is_some() && from_rank1_int.is_discrete());

        let from_rank2: QuantityValue<B> = floats(vec![1.0, 2.0], [2, 1]).into();
        assert_eq!(from_rank2.pinned_batch(), Some(2));

        let from_rank2_int: QuantityValue<B> = ints(vec![1, 2], [2, 1]).into();
        assert_eq!(from_rank2_int.pinned_batch(), Some(2));
        assert!(from_rank2_int.is_discrete());

        assert!(format!("{from_rank2:?}").contains("Batched"));
        assert!(format!("{:?}", from_rank1.clone()).contains("Unbatched"));
    }
}
