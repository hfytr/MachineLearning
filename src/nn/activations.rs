pub trait Activation {
    fn prime(x: f64) -> f64;
    fn calc(x: f64) -> f64;
}

pub struct Softplus;
impl Activation for Softplus {
    fn calc(x: f64) -> f64 {
        if x > 420.0 {
            x
        } else {
            (x.exp() + 1.0).ln()
        }
    }
    fn prime(x: f64) -> f64 {
        Sigmoid::calc(x)
    }
}

pub struct Sigmoid;
impl Activation for Sigmoid {
    fn calc(x: f64) -> f64 {
        if x < -420.0 {
            0.0
        } else {
            1.0 / (1.0 + (-x).exp())
        }
    }
    fn prime(x: f64) -> f64 {
        Self::calc(x) * (1.0 - Self::calc(x))
    }
}

pub struct ReLU;
impl Activation for ReLU {
    fn calc(x: f64) -> f64 {
        x.max(0.0)
    }
    fn prime(x: f64) -> f64 {
        if x > 0.0 {
            1.0
        } else {
            0.0
        }
    }
}
