pub mod feedforward;
pub mod activations {
    use std::f64::consts::E;
    pub fn softmax(x: f64) -> f64   { (E.powf(x) + 1.0).log(E) }
    pub fn sigmoid(x: f64) -> f64   { 1.0 / (1.0 + E.powf(-x)) }
    pub fn relu(x: f64) -> f64      { x.max(0.0) }
}
