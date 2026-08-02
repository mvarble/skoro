use skoro_backend::{Backend, CotangentFrame};

use crate::step::{DynHvpStep, DynVjpStep};

use super::DiffOperation;

/// A [`DiffOperation`] with a memoized VJP state that will be used to call back-propagation of
/// first-order cotangent vectors.
pub struct VjpStep<B: Backend, Op: DiffOperation<B>> {
    pub operation: Op,
    pub state: Op::VjpState,
    pub input_indices: Vec<usize>,
    pub output_index: usize,
}

impl<B: Backend, Op: DiffOperation<B>> DynVjpStep<B> for VjpStep<B, Op> {
    fn step(&self, cotangents: &mut [Option<CotangentFrame<B>>]) {
        if let Some(omega) = cotangents[self.output_index].take() {
            let new_cotangents = self.operation.vjp(&self.state, omega);
            for (&index, cotangent) in self.input_indices.iter().zip(new_cotangents) {
                if let Some(cotangent) = cotangent {
                    take_accumulate(cotangents, index, cotangent);
                }
            }
        }
    }

    fn input_indices(&self) -> &[usize] {
        &self.input_indices
    }

    fn output_index(&self) -> usize {
        self.output_index
    }
}

/// A [`DiffOperation`] with memoized VJP and HVP states that will be used to call back-propagation of
/// first-order and second-order cotangent vectors.
pub struct HvpStep<B: Backend, Op: DiffOperation<B>> {
    pub operation: Op,
    pub vjp_state: Op::VjpState,
    pub hvp_state: Op::HvpState,
    pub input_indices: Vec<usize>,
    pub output_index: usize,
}

impl<B: Backend, Op: DiffOperation<B>> DynHvpStep<B> for HvpStep<B, Op> {
    fn step(
        &self,
        ord1_cotangents: &mut [Option<CotangentFrame<B>>],
        ord2_cotangents: &mut [Option<CotangentFrame<B>>],
    ) {
        let alpha_opt = ord1_cotangents[self.output_index].take();
        let beta_opt = ord2_cotangents[self.output_index].take();

        if alpha_opt.is_none() && beta_opt.is_none() {
            return;
        }

        let alpha =
            alpha_opt.expect("Order-1 cotangent vector must be seeded if Order-2 is seeded.");
        let beta = beta_opt.expect("Order-2 cotangent vector must be seeded if Order-1 is seeded.");

        let new_ord1_cotangents = self.operation.vjp(&self.vjp_state, alpha.clone());
        let new_ord2_cotangents0 = self.operation.hvp(&self.vjp_state, &self.hvp_state, alpha);
        let new_ord2_cotangents1 = self.operation.vjp(&self.vjp_state, beta);

        for (((&index, alpha), beta0), beta1) in self
            .input_indices
            .iter()
            .zip(new_ord1_cotangents)
            .zip(new_ord2_cotangents0)
            .zip(new_ord2_cotangents1)
        {
            if let Some(alpha) = alpha {
                take_accumulate(ord1_cotangents, index, alpha);
            }
            let beta = match (beta0, beta1) {
                (Some(b0), Some(b1)) => Some(b0 + b1),
                (Some(b), None) | (None, Some(b)) => Some(b),
                (None, None) => None,
            };
            if let Some(b) = beta {
                take_accumulate(ord2_cotangents, index, b);
            }
        }
    }

    fn input_indices(&self) -> &[usize] {
        &self.input_indices
    }

    fn output_index(&self) -> usize {
        self.output_index
    }
}

#[inline]
fn take_accumulate<B: Backend>(
    vec: &mut [Option<CotangentFrame<B>>],
    index: usize,
    value: CotangentFrame<B>,
) {
    match vec[index].take() {
        Some(existing) => {
            vec[index] = Some(existing + value);
        }
        None => {
            vec[index] = Some(value);
        }
    }
}
