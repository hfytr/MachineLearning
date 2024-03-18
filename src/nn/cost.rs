use crate::algebra::Matrix;

pub trait Cost {
    fn calc(pred: &Matrix<f64>, actual: &Matrix<f64>) -> f64;
    fn prime(pred: &Matrix<f64>, actual: &Matrix<f64>) -> Matrix<f64>;
}

pub struct SumSquared {}
impl Cost for SumSquared {
    fn calc(pred: &Matrix<f64>, actual: &Matrix<f64>) -> f64 {
        pred.iter()
            .zip(actual.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum()
    }

    fn prime(pred: &Matrix<f64>, actual: &Matrix<f64>) -> Matrix<f64> {
        Matrix::new(
            pred.iter()
                .zip(actual.iter())
                .map(|(x, y)| 2.0 * (x - y))
                .collect(),
            1,
            pred.h(),
        )
    }
}
