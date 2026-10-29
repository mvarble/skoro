use std::ops::Range;
use std::sync::Arc;

use crate::{Embedded, Error};

/// A quantity's type in the mathematics' sense: a closed enumeration, dimensioned at run time.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Space {
    Real,
    Vector(usize),
    /// Rows and columns, stored column by column.
    Matrix(usize, usize),
    Positive(usize),
    /// Continuous factors only: made by [`Space::product`], which refuses a discrete one.
    Product(Arc<[Space]>),
    /// A curved type, equal to another when their declarations' names and parameters are.
    Embedded(Arc<Embedded>),
    Categorical(usize),
    Integer,
}

/// A factor of a product, with its ranges in the product's storage and tangent coordinates.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Block {
    pub space: Space,
    pub storage: Range<usize>,
    pub tangent: Range<usize>,
}

impl Space {
    /// `Categorical(2)`.
    pub const BOOLEAN: Space = Space::Categorical(2);

    /// The product of `factors`, which may themselves be products. Refuses a discrete factor
    /// (`NotContinuous`).
    pub fn product(factors: impl IntoIterator<Item = Space>) -> Result<Space, Error> {
        let factors: Arc<[Space]> = factors.into_iter().collect();
        match factors.iter().find(|factor| !factor.is_continuous()) {
            Some(factor) => Err(Error::NotContinuous {
                space: factor.clone(),
            }),
            None => Ok(Space::Product(factors)),
        }
    }

    pub fn is_continuous(&self) -> bool {
        !matches!(self, Space::Categorical(_) | Space::Integer)
    }

    /// The number of entries a value takes in a row of a tensor.
    pub fn storage(&self) -> usize {
        match self {
            Space::Real | Space::Categorical(_) | Space::Integer => 1,
            Space::Vector(n) | Space::Positive(n) => *n,
            Space::Matrix(m, n) => m * n,
            Space::Product(factors) => factors.iter().map(Space::storage).sum(),
            Space::Embedded(declaration) => declaration.storage(),
        }
    }

    /// The dimension of the tangent space; `None` for a discrete space.
    pub fn dimension(&self) -> Option<usize> {
        match self {
            Space::Categorical(_) | Space::Integer => None,
            Space::Product(factors) => factors.iter().map(Space::dimension).sum(),
            Space::Embedded(declaration) => Some(declaration.dimension()),
            continuous => Some(continuous.storage()),
        }
    }

    /// The number of values of a categorical space; `None` for any other.
    pub fn cardinality(&self) -> Option<usize> {
        match self {
            Space::Categorical(n) => Some(*n),
            _ => None,
        }
    }

    /// A product's factors with their ranges in storage and in tangent coordinates; any other
    /// space is one block of itself, whose tangent range is empty when it is discrete.
    pub fn blocks(&self) -> Vec<Block> {
        let factors = match self {
            Space::Product(factors) => factors.to_vec(),
            other => vec![other.clone()],
        };
        let (mut storage, mut tangent) = (0, 0);
        factors
            .into_iter()
            .map(|space| {
                let block = Block {
                    storage: storage..storage + space.storage(),
                    tangent: tangent..tangent + space.dimension().unwrap_or(0),
                    space,
                };
                storage = block.storage.end;
                tangent = block.tangent.end;
                block
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn product_refuses_a_discrete_factor() {
        for discrete in [Space::Integer, Space::BOOLEAN] {
            let result = Space::product([Space::Real, discrete.clone()]);
            assert_eq!(result, Err(Error::NotContinuous { space: discrete }));
        }
        let inner = Space::product([Space::Real, Space::Positive(2)]).unwrap();
        assert!(Space::product([inner, Space::Matrix(2, 3)]).is_ok());
    }

    #[test]
    fn storage_dimension_and_cardinality_of_the_flat_spaces() {
        let cases = [
            (Space::Real, 1, Some(1), None),
            (Space::Vector(3), 3, Some(3), None),
            (Space::Matrix(2, 3), 6, Some(6), None),
            (Space::Positive(4), 4, Some(4), None),
            (Space::Categorical(5), 1, None, Some(5)),
            (Space::Integer, 1, None, None),
        ];
        for (space, storage, dimension, cardinality) in cases {
            assert_eq!(space.storage(), storage, "{space:?}");
            assert_eq!(space.dimension(), dimension, "{space:?}");
            assert_eq!(space.cardinality(), cardinality, "{space:?}");
        }
    }

    #[test]
    fn blocks_of_a_nested_product() {
        let inner = Space::product([Space::Real, Space::Positive(2)]).unwrap();
        let space = Space::product([Space::Vector(2), inner.clone()]).unwrap();
        assert_eq!((space.storage(), space.dimension()), (5, Some(5)));
        assert_eq!(
            space.blocks(),
            [
                Block {
                    space: Space::Vector(2),
                    storage: 0..2,
                    tangent: 0..2
                },
                Block {
                    space: inner,
                    storage: 2..5,
                    tangent: 2..5
                },
            ]
        );
        let integer = Space::Integer.blocks();
        assert_eq!(integer[0].tangent, 0..0);
    }
}
