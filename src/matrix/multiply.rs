use std::ops::{AddAssign, Mul};
use super::*;

macro_rules! mat_vec_mul {
    ($type: ty) => (
        fn mul(self, rhs: $type) -> Vec<T> {
            assert_eq!(rhs.len(), self.w, "checking dimensionality");
            let mut output: Vec<T> = vec![T::default(); self.h];
            for i in 0..self.w {
                for j in 0..self.h {
                    output[i] += rhs[i] * self[(i,j)];
                }
            }
            output
        }
    );
}

impl<T> Mul<Vec<T>> for Matrix<T> 
where T: Mul<Output = T> + AddAssign + Copy + Default {
    type Output = Vec<T>;
    mat_vec_mul!(Vec<T>);
}

impl<T> Mul<&Vec<T>> for Matrix<T> 
where T: Mul<Output = T> + AddAssign + Copy + Default {
    type Output = Vec<T>;
    mat_vec_mul!(&Vec<T>);
}

impl<T> Mul<Vec<T>> for &Matrix<T> 
where T: Mul<Output = T> + AddAssign + Copy + Default {
    type Output = Vec<T>;
    mat_vec_mul!(Vec<T>);
}

impl<T> Mul<&Vec<T>> for &Matrix<T> 
where T: Mul<Output = T> + AddAssign + Copy + Default {
    type Output = Vec<T>;
    mat_vec_mul!(&Vec<T>);
}

macro_rules! mat_mat_mul {
    ($type: ty) => {
        fn mul(self, rhs: $type) -> Matrix<T> {
            assert_eq!(self.w, rhs.h, "checking dimensionality");
            let mut output: Matrix<T> = Matrix::new_uniform(T::default(), self.h, rhs.w);
            for i in 0..self.h {
                for j in 0..rhs.w {
                    for k in 0..rhs.h {
                        output[(i,j)] += self[(i,k)] * rhs[(k,j)];
                    }
                }
            }
            output
        }
    };
}

impl<T> Mul<Matrix<T>> for Matrix<T> 
where T: Mul<Output = T> + AddAssign + Copy + Default {
    type Output = Matrix<T>;
    mat_mat_mul!(Matrix<T>);
}

impl<T> Mul<&Matrix<T>> for Matrix<T> 
where T: Mul<Output = T> + AddAssign + Copy + Default {
    type Output = Matrix<T>;
    mat_mat_mul!(&Matrix<T>);
}

impl<T> Mul<Matrix<T>> for &Matrix<T> 
where T: Mul<Output = T> + AddAssign + Copy + Default {
    type Output = Matrix<T>;
    mat_mat_mul!(Matrix<T>);
}

impl<T> Mul<&Matrix<T>> for &Matrix<T> 
where T: Mul<Output = T> + AddAssign + Copy + Default {
    type Output = Matrix<T>;
    mat_mat_mul!(&Matrix<T>);
}
