use super::super::matrix::Matrix;

pub struct FFNet {
    layers: Vec<Layer>,
    activation: fn(f64) -> f64,
}

struct Layer {
    weights: Matrix<f64>,
    in_shape: usize,
    out_shape: usize,
    biases: Vec<f64>,
    activation: fn(f64) -> f64,
}

impl Layer {
    pub fn new(in_shape: usize, out_shape: usize, activation: fn(f64) -> f64) -> Self {
        Self { weights: Matrix::new_uniform(1_f64, in_shape, out_shape), // initialise to 1 for testing purposes
            in_shape,
            out_shape,
            biases: vec![1_f64; out_shape], 
            activation
        }
    }

    pub fn pred(&self, x: &Vec<f64>) -> Vec<f64> {
        &self.weights * x
    }
}

impl FFNet {
    pub fn new(shape: Vec<usize>, activation: fn(f64) -> f64) -> Self {
        let layers = (1..shape.len()).map(|i| Layer::new(shape[i-1], shape[i], activation)).collect();
        Self { layers, activation }
    }

    pub fn pred_single(&self, x: &Vec<f64>) -> Vec<f64> {
        let mut output = x.to_vec();
        for layer in &self.layers {
            output = layer.pred(&output);
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use crate::nn::activations::*;
    use super::FFNet;

    #[test]
    fn layer_computation_works(){
        const ERROR_MARGIN: f64 = 0.00001;
        let input: Vec<f64> = vec![-1.9, 2.5];
        let expected: Vec<f64> = vec![0.6, 1.03748_f64, 0.64565_f64];
        let activations = vec![relu, softplus, sigmoid];
        for (i, activation) in activations.iter().enumerate() {
            let net = FFNet::new(vec![2,1], *activation);
            let result = net.pred_single(&input);
            // dont want to deal with float imprecision
            assert!(result[0] <= expected[i] + ERROR_MARGIN, "result[0] = {}", result[0]);
            assert!(result[0] >= expected[i] - ERROR_MARGIN, "result[0] = {}", result[0]);
        }
    }
}
