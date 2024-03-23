#[cfg(test)]
mod tests {
    use ml::algebra::Matrix;
    use ml::data::Dataset;
    use ml::nn::{activations::*, cost::SumSquared, feedforward::FFNet};
    const ERROR_MARGIN: f64 = 0.00001;

    macro_rules! test_with_activation {
        ($type: ty, $expected: expr, $input: ident) => {
            let mut net = FFNet::<$type, SumSquared>::new(vec![2, 1]);
            let result = net.pred_single($input.clone());
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
        let input: Matrix<f64> = Matrix::new(vec![-1.9, 2.5], 1, 2);
        let expected: Matrix<f64> = Matrix::new(vec![0.6, 1.03748_f64, 0.64565_f64], 1, 3);
        test_with_activation!(ReLU, expected[0], input);
        test_with_activation!(Softplus, expected[1], input);
        test_with_activation!(Sigmoid, expected[2], input);
    }

    #[test]
    fn training_works() {
        let train_path = String::from("data/mnist_small.csv");
        let mut data = Dataset::from_csv(&train_path);
        let y_keys = data.one_hot_encode("label", 1);
        let x_keys: Vec<String> = data
            .keys()
            .iter()
            .filter(|&value| value != "label")
            .cloned()
            .collect();
        let x = data.to_matrix(&x_keys);
        let y = data.to_matrix(&y_keys);
        println!("{}", y);
        let mut net: FFNet<ReLU, SumSquared> = FFNet::new(vec![784_usize, 2_usize, 10_usize]);
        net.sgd(&x, &y, 4, 0.1);
        panic!();
    }
}
