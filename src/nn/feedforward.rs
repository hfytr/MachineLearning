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
        Self { weights: Matrix::new_uniform(0_f64, out_shape, in_shape),
            in_shape,
            out_shape,
            biases: vec![0_f64; out_shape], 
            activation
        }
    }

    pub fn pred(&self, x: Vec<f64>) -> Vec<f64> { &self.weights * x }
}

impl FFNet {
    pub fn new(shape: Vec<usize>, activation: fn(f64) -> f64) -> Self {
        let layers = (1..shape.len()).map(|i| Layer::new(shape[i-1], shape[i], activation)).collect();
        Self { layers, activation }
    }

    pub fn pred_single(&self, x: Vec<f64>) -> Vec<f64> {
        let mut output = x;
        for layer in &self.layers {
            output = layer.pred(output);
        }
        output
    }
}
