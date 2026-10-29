use burn::tensor::{Bool, DType, FloatDType, Int, IntDType, Tensor, TensorData};

use crate::{Target, TensorError};

/// A batch of continuous values, `[rows, storage]`, at its target's precision.
#[derive(Clone, Debug)]
pub struct Points(Tensor<2>);

/// A batch of frames of tangents or cotangents, `[rows, dimension, columns]`, at its target's
/// precision.
#[derive(Clone, Debug)]
pub struct Frame(Tensor<3>);

/// A batch of discrete values, `[rows, storage]`, in its target's integers.
#[derive(Clone, Debug)]
pub struct Integers(Tensor<2, Int>);

/// A batch of values of either kind.
#[derive(Clone, Debug)]
pub enum Value {
    Continuous(Points),
    Discrete(Integers),
}

impl Points {
    /// `rows × storage` values in row-major order, at the target's precision.
    ///
    /// # Panics
    ///
    /// When `data` does not hold `rows × storage` values.
    pub fn from_host(data: &[f64], rows: usize, storage: usize, target: &Target) -> Points {
        Points(float_tensor(data, [rows, storage], target))
    }

    pub fn zeros(rows: usize, storage: usize, target: &Target) -> Points {
        Points(Tensor::zeros(
            [rows, storage],
            (target.device(), target.float_dtype()),
        ))
    }

    /// Gives the tensor up to burn, for a kernel to compute with.
    pub fn into_burn(self) -> Tensor<2> {
        self.0
    }

    /// Takes a tensor back from burn, refusing one of another data type or device than the
    /// target's (`ForeignTensor`).
    pub fn from_burn(tensor: Tensor<2>, target: &Target) -> Result<Points, TensorError> {
        check_foreign(
            tensor.dtype(),
            &tensor.device(),
            target.float_dtype(),
            target,
        )?;
        Ok(Points(tensor))
    }

    pub fn rows(&self) -> usize {
        self.0.dims()[0]
    }

    pub fn storage(&self) -> usize {
        self.0.dims()[1]
    }

    /// The values in row-major order, in double precision whatever the target's.
    pub fn to_host(&self) -> Vec<f64> {
        read(self.0.to_data())
    }
}

impl Frame {
    /// `rows × dimension × columns` values in row-major order, at the target's precision.
    ///
    /// # Panics
    ///
    /// When `data` does not hold `rows × dimension × columns` values.
    pub fn from_host(
        data: &[f64],
        rows: usize,
        dimension: usize,
        columns: usize,
        target: &Target,
    ) -> Frame {
        Frame(float_tensor(data, [rows, dimension, columns], target))
    }

    pub fn zeros(rows: usize, dimension: usize, columns: usize, target: &Target) -> Frame {
        Frame(Tensor::zeros(
            [rows, dimension, columns],
            (target.device(), target.float_dtype()),
        ))
    }

    /// Gives the tensor up to burn, for a kernel to compute with.
    pub fn into_burn(self) -> Tensor<3> {
        self.0
    }

    /// Takes a tensor back from burn, refusing one of another data type or device than the
    /// target's (`ForeignTensor`).
    pub fn from_burn(tensor: Tensor<3>, target: &Target) -> Result<Frame, TensorError> {
        check_foreign(
            tensor.dtype(),
            &tensor.device(),
            target.float_dtype(),
            target,
        )?;
        Ok(Frame(tensor))
    }

    pub fn rows(&self) -> usize {
        self.0.dims()[0]
    }

    pub fn dimension(&self) -> usize {
        self.0.dims()[1]
    }

    pub fn columns(&self) -> usize {
        self.0.dims()[2]
    }

    /// The values in row-major order, in double precision whatever the target's.
    pub fn to_host(&self) -> Vec<f64> {
        read(self.0.to_data())
    }
}

impl Integers {
    /// `rows × storage` values in row-major order, in the target's integers. A value that does
    /// not fit them is refused (`IntegerOutOfRange`).
    ///
    /// # Panics
    ///
    /// When `data` does not hold `rows × storage` values.
    pub fn from_host(
        data: &[i64],
        rows: usize,
        storage: usize,
        target: &Target,
    ) -> Result<Integers, TensorError> {
        if target.int_dtype() == DType::I32
            && let Some(&value) = data.iter().find(|&&v| i32::try_from(v).is_err())
        {
            return Err(TensorError::IntegerOutOfRange { value });
        }
        let data = TensorData::new(data.to_vec(), [rows, storage]);
        Ok(Integers(Tensor::from_data(
            data,
            (target.device(), target.int_dtype()),
        )))
    }

    pub fn zeros(rows: usize, storage: usize, target: &Target) -> Integers {
        Integers(Tensor::zeros(
            [rows, storage],
            (target.device(), target.int_dtype()),
        ))
    }

    /// The comparison `mask` as integers, one for true and zero for false, in the target's
    /// integers rather than the device's default. Refuses a mask on another device
    /// (`ForeignTensor`).
    pub fn from_mask(mask: Tensor<2, Bool>, target: &Target) -> Result<Integers, TensorError> {
        let device = mask.device();
        if &device != target.device() {
            return Err(foreign(target.int_dtype(), target, mask.dtype(), &device));
        }
        Ok(Integers(mask.cast(IntDType::from(target.int_dtype()))))
    }

    /// The values as points at the target's precision, rather than the device's default.
    pub fn to_points(&self, target: &Target) -> Points {
        Points(self.0.clone().cast(FloatDType::from(target.float_dtype())))
    }

    /// Gives the tensor up to burn, for a kernel to compute with.
    pub fn into_burn(self) -> Tensor<2, Int> {
        self.0
    }

    /// Takes a tensor back from burn, refusing one of another data type or device than the
    /// target's (`ForeignTensor`).
    pub fn from_burn(tensor: Tensor<2, Int>, target: &Target) -> Result<Integers, TensorError> {
        check_foreign(tensor.dtype(), &tensor.device(), target.int_dtype(), target)?;
        Ok(Integers(tensor))
    }

    pub fn rows(&self) -> usize {
        self.0.dims()[0]
    }

    pub fn storage(&self) -> usize {
        self.0.dims()[1]
    }

    /// The values in row-major order.
    pub fn to_host(&self) -> Vec<i64> {
        self.0.to_data().iter::<i64>().collect()
    }
}

impl Value {
    pub fn rows(&self) -> usize {
        match self {
            Value::Continuous(points) => points.rows(),
            Value::Discrete(integers) => integers.rows(),
        }
    }

    pub fn storage(&self) -> usize {
        match self {
            Value::Continuous(points) => points.storage(),
            Value::Discrete(integers) => integers.storage(),
        }
    }
}

fn float_tensor<const D: usize>(data: &[f64], shape: [usize; D], target: &Target) -> Tensor<D> {
    let data = TensorData::new(data.to_vec(), shape);
    Tensor::from_data(data, (target.device(), target.float_dtype()))
}

fn read(data: TensorData) -> Vec<f64> {
    data.iter::<f64>().collect()
}

fn check_foreign(
    dtype: DType,
    device: &burn::tensor::Device,
    expected: DType,
    target: &Target,
) -> Result<(), TensorError> {
    if dtype == expected && device == target.device() {
        Ok(())
    } else {
        Err(foreign(expected, target, dtype, device))
    }
}

fn foreign(
    expected: DType,
    target: &Target,
    dtype: DType,
    device: &burn::tensor::Device,
) -> TensorError {
    TensorError::ForeignTensor {
        expected: format!("{expected:?} on {:?}", target.device()),
        found: format!("{dtype:?} on {device:?}"),
    }
}
