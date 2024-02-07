use super::{Activation, FFNet, Layer};
use crate::algebra::{Matrix, Vector};

impl<T: Activation> FFNet<T> {
    pub fn train(data: Matrix<f64>) {}

    fn randomize_weights(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.randomize_weights();
        }
    }
}

impl<T: Activation> Layer<T> {
    pub fn randomize_weights(&mut self) {
        self.weights = Matrix::random(-0.3_f64, 0.3_f64, self.weights.w(), self.weights.h());
    }

    // a_wrt_b = partial derivative of a with respect to b
    pub fn calculate_grad(
        &mut self,
        input: Vector<f64>,
        cost_wrt_output: Vector<f64>,
    ) -> (Matrix<f64>, Vector<f64>, Vector<f64>) {
        assert_eq!(
            cost_wrt_output.shape(),
            self.out_shape,
            "cost_wrt_output does not match out_shape"
        );
        assert_eq!(
            input.shape(),
            self.in_shape,
            "input len does not match in_shape"
        );
        self.unactivated = self.weights * input + self.biases;
        let output_wrt_unactivated = self.unactivated.iter().map(|i| T::prime(*i)).collect();
        let prefix = cost_wrt_output.mul_element_wise(output_wrt_unactivated);
        let weight_grad = prefix.outer_prod(input);
        let cost_wrt_input = self.weights * prefix;
        (weight_grad, prefix, cost_wrt_input)
    }
}
