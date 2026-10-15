use skoro_tensor::{HostData, HostValue};

use crate::{
    Boolean, Categorical, CategoricalDyn, Integer, Matrix, MatrixDyn, Positive, PositiveDyn, Real,
    Vector, VectorDyn,
};

/// A Rust value that a marker suggests, as a value on the host shared by every row; `None` when
/// it cannot be one.
pub trait IntoHostValue<T> {
    fn into_host_value(self) -> Option<HostValue>;
}

/// One value per row, pinning the batch to their number.
#[derive(Clone, Debug, PartialEq)]
pub struct Rows<V>(pub Vec<V>);

fn continuous(data: Vec<f64>) -> Option<HostValue> {
    HostValue::new(None, data.len(), HostData::Continuous(data))
}

fn discrete(value: i64) -> Option<HostValue> {
    HostValue::new(None, 1, HostData::Discrete(vec![value]))
}

impl IntoHostValue<Real> for f64 {
    fn into_host_value(self) -> Option<HostValue> {
        continuous(vec![self])
    }
}

impl<const N: usize> IntoHostValue<Vector<N>> for [f64; N] {
    fn into_host_value(self) -> Option<HostValue> {
        continuous(self.to_vec())
    }
}

impl<const N: usize> IntoHostValue<Positive<N>> for [f64; N] {
    fn into_host_value(self) -> Option<HostValue> {
        continuous(self.to_vec())
    }
}

/// The matrix's rows, stored column by column.
impl<const M: usize, const N: usize> IntoHostValue<Matrix<M, N>> for [[f64; N]; M] {
    fn into_host_value(self) -> Option<HostValue> {
        continuous(
            (0..N)
                .flat_map(|j| self.iter().map(move |row| row[j]))
                .collect(),
        )
    }
}

impl IntoHostValue<VectorDyn> for &[f64] {
    fn into_host_value(self) -> Option<HostValue> {
        continuous(self.to_vec())
    }
}

/// In storage order: column by column.
impl IntoHostValue<MatrixDyn> for &[f64] {
    fn into_host_value(self) -> Option<HostValue> {
        continuous(self.to_vec())
    }
}

impl IntoHostValue<PositiveDyn> for &[f64] {
    fn into_host_value(self) -> Option<HostValue> {
        continuous(self.to_vec())
    }
}

impl IntoHostValue<Boolean> for bool {
    fn into_host_value(self) -> Option<HostValue> {
        discrete(i64::from(self))
    }
}

impl<const N: usize> IntoHostValue<Categorical<N>> for usize {
    fn into_host_value(self) -> Option<HostValue> {
        discrete(i64::try_from(self).ok()?)
    }
}

impl IntoHostValue<CategoricalDyn> for usize {
    fn into_host_value(self) -> Option<HostValue> {
        discrete(i64::try_from(self).ok()?)
    }
}

impl IntoHostValue<Integer> for i64 {
    fn into_host_value(self) -> Option<HostValue> {
        discrete(self)
    }
}

impl<T, V: IntoHostValue<T>> IntoHostValue<T> for Rows<V> {
    fn into_host_value(self) -> Option<HostValue> {
        let rows = self.0.len();
        let values = self
            .0
            .into_iter()
            .map(IntoHostValue::into_host_value)
            .collect::<Option<Vec<HostValue>>>()?;
        let storage = values.first()?.storage();
        let data = match values.first()?.data() {
            HostData::Continuous(_) => HostData::Continuous(
                values
                    .iter()
                    .map(|v| match v.data() {
                        HostData::Continuous(data) => Some(data.clone()),
                        HostData::Discrete(_) => None,
                    })
                    .collect::<Option<Vec<_>>>()?
                    .concat(),
            ),
            HostData::Discrete(_) => HostData::Discrete(
                values
                    .iter()
                    .map(|v| match v.data() {
                        HostData::Discrete(data) => Some(data.clone()),
                        HostData::Continuous(_) => None,
                    })
                    .collect::<Option<Vec<_>>>()?
                    .concat(),
            ),
        };
        HostValue::new(Some(rows), storage, data)
    }
}
