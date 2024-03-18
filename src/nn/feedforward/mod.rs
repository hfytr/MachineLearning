use super::activations::Activation;
use super::cost::Cost;
use crate::algebra::Matrix;
use std::marker::PhantomData;

mod train;

pub struct FFNet<A: Activation, C: Cost> {
    layers: Vec<Layer<A, C>>,
    activated: Vec<Matrix<f64>>,   // row vec
    unactivated: Vec<Matrix<f64>>, // col vec
    spine_chilling: PhantomData<A>,
    gut_wrneching: PhantomData<C>,
}

struct Layer<A: Activation, C: Cost> {
    weights: Matrix<f64>,
    biases: Matrix<f64>,
    in_shape: usize,
    out_shape: usize,
    spine_chilling: PhantomData<A>,
    gut_wrneching: PhantomData<C>,
}

impl<A: Activation, C: Cost> Layer<A, C> {
    pub fn new(in_shape: usize, out_shape: usize) -> Self {
        Self {
            weights: Matrix::new_uniform(1_f64, in_shape, out_shape), // initialise to 1 for testing purposes
            in_shape,
            out_shape,
            biases: Matrix::new_uniform(0.0, 1, out_shape),
            spine_chilling: PhantomData,
            gut_wrneching: PhantomData,
        }
    }

    pub fn pred(&self, input: &Matrix<f64>) -> (Matrix<f64>, Matrix<f64>) {
        let unactivated = &self.weights * input + &self.biases;
        let activated = unactivated.apply(|x| A::calc(x));
        (unactivated, activated)
    }
}

impl<A: Activation, C: Cost> FFNet<A, C> {
    pub fn new(shape: Vec<usize>) -> Self {
        let layers = (1..shape.len())
            .map(|i| Layer::<A, C>::new(shape[i - 1], shape[i]))
            .collect();
        let activated = (0..shape.len())
            .map(|i| Matrix::<f64>::new_uniform(0.0, 1, shape[i]))
            .collect();
        let unactivated = (1..shape.len())
            .map(|i| Matrix::<f64>::new_uniform(0.0, 1, shape[i]))
            .collect();
        Self {
            layers,
            activated,
            unactivated,
            spine_chilling: PhantomData,
            gut_wrneching: PhantomData,
        }
    }

    pub fn pred_single(&mut self, input: Matrix<f64>) -> &Matrix<f64> {
        self.activated[0] = input;
        let layers = self.layers.iter().enumerate();
        for (i, layer) in layers {
            (self.unactivated[i], self.activated[i + 1]) = layer.pred(&self.activated[i]);
            self.activated[i].transpose();
        }
        self.activated.last_mut().unwrap().transpose();
        self.activated.last().unwrap()
    }
}
