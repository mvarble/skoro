use crate::{Integers, Points, Target, TensorError, Value};

/// A value on the host: what a model stores and what a query observes.
#[derive(Clone, PartialEq, Debug)]
pub struct HostValue {
    rows: Option<usize>,
    storage: usize,
    data: HostData,
}

/// The values of a [`HostValue`], in row-major order.
#[derive(Clone, PartialEq, Debug)]
pub enum HostData {
    Continuous(Vec<f64>),
    Discrete(Vec<i64>),
}

impl HostData {
    pub fn len(&self) -> usize {
        match self {
            HostData::Continuous(data) => data.len(),
            HostData::Discrete(data) => data.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl HostValue {
    /// One value shared by every row of a batch (`rows: None`), or one value per row, pinning the
    /// batch to `rows` (`Some(rows)`). `None` when `data` does not hold that many values of
    /// `storage` entries.
    pub fn new(rows: Option<usize>, storage: usize, data: HostData) -> Option<HostValue> {
        (rows.unwrap_or(1).checked_mul(storage) == Some(data.len())).then_some(HostValue {
            rows,
            storage,
            data,
        })
    }

    /// The batch this value pins, or `None` when it is shared by every row.
    pub fn rows(&self) -> Option<usize> {
        self.rows
    }

    pub fn storage(&self) -> usize {
        self.storage
    }

    pub fn data(&self) -> &HostData {
        &self.data
    }

    /// The value for a pass at `batch` rows on the target: a shared value loaded as
    /// `[1, storage]` and expanded to the batch, a per-row value as it is. `Ok(None)` when the value
    /// pins another batch, even of one row, which is never broadcast.
    pub fn load(&self, batch: usize, target: &Target) -> Result<Option<Value>, TensorError> {
        if self.rows.is_some_and(|rows| rows != batch) {
            return Ok(None);
        }
        let rows = self.rows.unwrap_or(1);
        let value = match &self.data {
            HostData::Continuous(data) => {
                let points = Points::from_host(data, rows, self.storage, target).into_burn();
                Value::Continuous(Points::from_burn(
                    points.expand([batch, self.storage]),
                    target,
                )?)
            }
            HostData::Discrete(data) => {
                let integers = Integers::from_host(data, rows, self.storage, target)?.into_burn();
                Value::Discrete(Integers::from_burn(
                    integers.expand([batch, self.storage]),
                    target,
                )?)
            }
        };
        Ok(Some(value))
    }
}
