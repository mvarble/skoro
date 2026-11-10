/// A dense matrix, row by row: the crate's own, so that no faer type is public.
#[derive(Clone, PartialEq, Debug)]
pub struct Matrix {
    rows: usize,
    columns: usize,
    data: Vec<f64>,
}

impl Matrix {
    /// The matrix of `rows` rows and `columns` columns whose entries are `data`, row by row;
    /// `None` when `data` does not hold that many.
    pub fn new(rows: usize, columns: usize, data: Vec<f64>) -> Option<Matrix> {
        (rows.checked_mul(columns) == Some(data.len())).then_some(Matrix {
            rows,
            columns,
            data,
        })
    }

    pub fn zeros(rows: usize, columns: usize) -> Matrix {
        Matrix {
            rows,
            columns,
            data: vec![0.0; rows * columns],
        }
    }

    pub fn identity(n: usize) -> Matrix {
        let mut matrix = Matrix::zeros(n, n);
        for i in 0..n {
            matrix.set(i, i, 1.0);
        }
        matrix
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn get(&self, row: usize, column: usize) -> f64 {
        self.data[row * self.columns + column]
    }

    /// The entries, row by row.
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// The entries, column by column, as a skoro `MatrixDyn` stores them.
    pub fn by_columns(&self) -> Vec<f64> {
        (0..self.columns)
            .flat_map(|j| (0..self.rows).map(move |i| self.get(i, j)))
            .collect()
    }

    /// `A x`.
    pub fn apply(&self, x: &[f64]) -> Vec<f64> {
        (0..self.rows)
            .map(|i| (0..self.columns).map(|j| self.get(i, j) * x[j]).sum())
            .collect()
    }

    pub(crate) fn set(&mut self, row: usize, column: usize, value: f64) {
        self.data[row * self.columns + column] = value;
    }
}
