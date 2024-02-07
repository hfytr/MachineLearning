use super::activations::Activation;
use crate::algebra::{Matrix, Vector};

mod train;

pub struct FFNet<T: Activation> {
    layers: Vec<Layer<T>>,
}

struct Layer<T: Activation> {
    weights: Matrix<f64>,
    biases: Vector<f64>,
    in_shape: usize,
    out_shape: usize,
    output: Vector<f64>,
    unactivated: Vector<f64>, // value of output before activation function
}

impl<T: Activation> Layer<T> {
    pub fn new(in_shape: usize, out_shape: usize) -> Self {
        Self {
            weights: Matrix::new_uniform(1_f64, in_shape, out_shape), // initialise to 1 for testing purposes
            in_shape,
            out_shape,
            biases: Vector::new_uniform(0.0, out_shape),
            output: Vector::new_uniform(0.0, out_shape),
            unactivated: Vector::new_uniform(0.0, out_shape),
        }
    }

    pub fn pred(&self, x: &Vector<f64>) -> Vector<f64> {
        (&self.weights * x + self.biases).apply(|x| T::calc(x))
    }
}

impl<T: Activation> FFNet<T> {
    pub fn new(shape: Vector<usize>) -> Self {
        let layers = (1..shape.shape())
            .map(|i| Layer::<T>::new(shape[i - 1], shape[i]))
            .collect();
        Self { layers }
    }

    pub fn pred_single(&self, x: &Vector<f64>) -> Vector<f64> {
        let mut output: Vector<f64> = x.to_vec();
        for layer in &self.layers {
            output = layer.pred(&output);
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::FFNet;
    use crate::algebra::Vector;
    use crate::nn::activations::*;
    const ERROR_MARGIN: f64 = 0.00001;

    macro_rules! test_with_activation {
        ($type: ty, $expected: expr, $input: ident) => {
            let net: FFNet<$type> = FFNet::<$type>::new(Vector::new(vec![2_usize, 1_usize]));
            let result = net.pred_single(&$input);
            assert!(
                result[0] <= $expected + ERROR_MARGIN,
                "result[0] = {}",
                result[0]
            );
            assert!(
                result[0] >= $expected - ERROR_MARGIN,
                "result[0] = {}",
                result[0]
            );
        };
    }

    #[test]
    fn computation() {
        let input: Vector<f64> = Vector::new(vec![-1.9, 2.5]);
        let expected: Vector<f64> = Vector::new(vec![0.6, 1.03748_f64, 0.64565_f64]);
        test_with_activation!(ReLU, expected[0], input);
        test_with_activation!(Softplus, expected[1], input);
        test_with_activation!(Sigmoid, expected[2], input);
    }
}
