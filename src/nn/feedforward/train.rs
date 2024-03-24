use super::{Activation, Cost, FFNet, Layer};
use crate::algebra::Matrix;

impl<A: Activation, C: Cost> FFNet<A, C> {
    pub fn sgd(&mut self, x: &Matrix<f64>, y: &Matrix<f64>, batch_size: usize, learning_rate: f64) {
        self.randomize_params();
        let (mut weight_grad, mut bias_grad) = self.init_params();
        for i in 0..x.h() {
            let (case_weight, case_bias) =
                self.single_case_grad(x.clone_row(i).transpose(), &y.clone_row(i));
            for j in (0..self.layers.len()).rev() {
                weight_grad[j] =
                    &weight_grad[j] - &case_weight[j] * (learning_rate / batch_size as f64);
                bias_grad[j] = &bias_grad[j] - &case_bias[j] * (learning_rate / batch_size as f64);
            }
            println!("case_weight: {}case_bias: {}", case_weight[1], case_bias[1]);
            if i % batch_size == 0 {
                self.apply_grad(&weight_grad, &bias_grad);
                println!(
                    "weight: {}bias: {}",
                    self.layers[1].weights, self.layers[1].biases
                );
                (weight_grad, bias_grad) = self.init_params();
            }
        }
    }

    fn single_case_grad(
        &mut self,
        input: &Matrix<f64>,
        output: &Matrix<f64>,
    ) -> (Vec<Matrix<f64>>, Vec<Matrix<f64>>) {
        let result = self.pred_single(input.clone());
        let (mut weight_grad, mut bias_grad);
        let mut cost_wrt_output: Matrix<f64> = C::prime(result, output);
        let mut weight_grads = vec![Matrix::<f64>::default(); self.unactivated.len()];
        let mut bias_grads = vec![Matrix::<f64>::default(); self.unactivated.len()];

        for (i, layer) in self.layers.iter_mut().enumerate().rev() {
            (weight_grad, bias_grad, cost_wrt_output) = layer.calculate_grad(
                &self.activated[i],
                &self.unactivated[i],
                &cost_wrt_output,
                i == 0,
            );
            weight_grads[i] = weight_grad;
            bias_grads[i] = bias_grad;
        }
        (weight_grads, bias_grads)
    }

    fn init_params(&self) -> (Vec<Matrix<f64>>, Vec<Matrix<f64>>) {
        (
            self.layers
                .iter()
                .map(|layer| Matrix::<f64>::new_uniform(0.0, layer.in_shape, layer.out_shape))
                .collect(),
            self.layers
                .iter()
                .map(|layer| Matrix::<f64>::new_uniform(0.0, 1, layer.out_shape))
                .collect(),
        )
    }

    fn apply_grad(&mut self, weight: &[Matrix<f64>], bias: &[Matrix<f64>]) {
        for i in 0..self.layers.len() {
            self.layers[i].apply_grad(&weight[i], &bias[i]);
        }
    }

    fn randomize_params(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.randomize_params();
        }
    }
}

impl<A: Activation, C: Cost> Layer<A, C> {
    // a_wrt_b = partial derivative of a with respect to b
    /// @return (weight gradient, bias gradient, cost_wrt_input = cost_wrt_output for next layer)
    pub fn calculate_grad(
        &self,
        input: &Matrix<f64>,              // row
        unactivated_output: &Matrix<f64>, // col
        cost_wrt_output: &Matrix<f64>,    // col
        is_first_layer: bool,
    ) -> (Matrix<f64>, Matrix<f64>, Matrix<f64>) {
        assert_eq!(
            cost_wrt_output.h(),
            self.out_shape,
            "cost_wrt_output does not match out_shape"
        );
        assert_eq!(
            input.w(),
            self.in_shape,
            "input len does not match in_shape"
        );

        let output_wrt_unactivated = unactivated_output.apply(|x| A::prime(x));
        /*println!(
            "output_wrt_unactivated: {}cost_wrt_output: {}",
            output_wrt_unactivated, cost_wrt_output
        );*/
        let mut cost_wrt_unactivated = cost_wrt_output.mul_element_wise(output_wrt_unactivated);
        let weight_grad = &cost_wrt_unactivated * input;

        // cost_wrt_input will be passed too next layer as cost_wrt_output, so its unneeded if this
        // is the first layer
        if is_first_layer {
            (weight_grad, cost_wrt_unactivated, Matrix::<f64>::default())
        } else {
            let mut cost_wrt_input = cost_wrt_unactivated.transpose() * &self.weights;
            cost_wrt_unactivated.transpose();
            cost_wrt_input.transpose();
            (weight_grad, cost_wrt_unactivated, cost_wrt_input)
        }
    }

    pub fn apply_grad(&mut self, weight: &Matrix<f64>, bias: &Matrix<f64>) {
        self.weights = &self.weights + weight;
        self.biases = &self.biases + bias;
    }

    pub fn randomize_params(&mut self) {
        self.weights = Matrix::random(-0.3_f64, 0.3_f64, self.weights.w(), self.weights.h());
        self.biases = Matrix::random(-0.3_f64, 0.3_f64, self.biases.w(), self.biases.h());
    }
}
