use std::ops::Range;

use crate::Error;

/// The factor structure: blocks of given dimensions, and for each factor the blocks it involves,
/// in increasing order.
#[derive(Clone, PartialEq, Debug)]
pub struct Structure {
    dimensions: Vec<usize>,
    factors: Vec<Vec<usize>>,
    /// Where each block's coordinates start in the stacked tangent.
    offsets: Vec<usize>,
}

impl Structure {
    /// Refuses a factor that names a block twice, out of increasing order, or one that does not
    /// exist (`Structure`).
    pub fn new(dimensions: Vec<usize>, factors: Vec<Vec<usize>>) -> Result<Structure, Error> {
        for (f, blocks) in factors.iter().enumerate() {
            let increasing = blocks.windows(2).all(|w| w[0] < w[1]);
            if !increasing || blocks.iter().any(|&k| k >= dimensions.len()) {
                return Err(Error::Structure { factor: f });
            }
        }
        let mut offsets = Vec::with_capacity(dimensions.len());
        let mut start = 0;
        for d in &dimensions {
            offsets.push(start);
            start += d;
        }
        Ok(Structure {
            dimensions,
            factors,
            offsets,
        })
    }

    /// `D`, the sum of the blocks' dimensions.
    pub fn dimension(&self) -> usize {
        self.dimensions.iter().sum()
    }

    /// The blocks' dimensions.
    pub fn blocks(&self) -> &[usize] {
        &self.dimensions
    }

    pub fn factors(&self) -> &[Vec<usize>] {
        &self.factors
    }

    /// Block `k`'s coordinates in the stacked tangent.
    pub fn range(&self, k: usize) -> Range<usize> {
        self.offsets[k]..self.offsets[k] + self.dimensions[k]
    }

    /// The number of columns of factor `f`'s rows.
    pub(crate) fn width(&self, f: usize) -> usize {
        self.factors[f].iter().map(|&k| self.dimensions[k]).sum()
    }

    /// The coordinates of `tangent` at factor `f`'s blocks, in order.
    pub(crate) fn gather(&self, f: usize, tangent: &[f64]) -> Vec<f64> {
        self.factors[f]
            .iter()
            .flat_map(|&k| tangent[self.range(k)].iter().copied())
            .collect()
    }
}
