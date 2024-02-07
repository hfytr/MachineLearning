use super::Vector;
use rand::{distributions::uniform::SampleUniform, Rng};
use std::{
    iter::Sum,
    ops::{AddAssign, Index, IndexMut, Mul},
};

pub struct Matrix<T: Mul> {
    data: Vec<T>,
    w: usize,
    h: usize,
}

impl<T: Mul<Output = T> + Copy> Matrix<T> {
    pub fn new(input: Vec<T>, w: usize, h: usize) -> Self {
        assert_eq!(w * h, input.len(), "input size does not match shape");
        Self { data: input, w, h }
    }

    pub fn new_uniform(val: T, w: usize, h: usize) -> Self {
        Self {
            data: vec![val; w * h],
            w,
            h,
        }
    }

    pub fn w(&self) -> usize {
        self.w
    }
    pub fn h(&self) -> usize {
        self.h
    }
}

impl<T> Matrix<T>
where
    T: Mul<Output = T> + Copy + PartialOrd + SampleUniform,
{
    pub fn random(low: T, high: T, w: usize, h: usize) -> Matrix<T> {
        let mut rng = rand::thread_rng();
        Self {
            data: (0..(w * h)).map(|_| rng.gen_range(low..high)).collect(),
            w,
            h,
        }
    }
}

impl<T: Mul<Output = T>> Index<(usize, usize)> for Matrix<T> {
    type Output = T;
    fn index(&self, index: (usize, usize)) -> &T {
        &self.data[index.0 * self.w + index.1]
    }
}

impl<T: Mul<Output = T>> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut T {
        &mut self.data[index.0 * self.w + index.1]
    }
}

macro_rules! mat_vec_mul {
    ($type: ty) => {
        fn mul(self, rhs: $type) -> Vector<T> {
            assert_eq!(rhs.shape(), self.w, "checking dimensionality");
            let mut output = Vector::<T>::new(vec![T::default(); self.h]);
            for i in 0..self.h {
                for j in 0..self.w {
                    output[i] += rhs[j] * self[(i, j)];
                }
            }
            output
        }
    };
}

impl<T> Mul<Vector<T>> for Matrix<T>
where
    T: Mul<Output = T> + AddAssign + Copy + Default + Sum,
{
    type Output = Vector<T>;
    mat_vec_mul!(Vector<T>);
}

impl<T> Mul<&Vector<T>> for Matrix<T>
where
    T: Mul<Output = T> + AddAssign + Copy + Default + Sum,
{
    type Output = Vector<T>;
    mat_vec_mul!(&Vector<T>);
}

impl<T> Mul<Vector<T>> for &Matrix<T>
where
    T: Mul<Output = T> + AddAssign + Copy + Default + Sum,
{
    type Output = Vector<T>;
    mat_vec_mul!(Vector<T>);
}

impl<T> Mul<&Vector<T>> for &Matrix<T>
where
    T: Mul<Output = T> + AddAssign + Copy + Default + Sum,
{
    type Output = Vector<T>;
    mat_vec_mul!(&Vector<T>);
}

macro_rules! mat_mat_mul {
    ($type: ty) => {
        fn mul(self, rhs: $type) -> Matrix<T> {
            assert_eq!(self.w, rhs.h, "checking dimensionality");
            let mut output: Matrix<T> = Matrix::new_uniform(T::default(), self.h, rhs.w);
            for i in 0..self.h {
                for j in 0..rhs.w {
                    for k in 0..rhs.h {
                        output[(i, j)] += self[(i, k)] * rhs[(k, j)];
                    }
                }
            }
            output
        }
    };
}

impl<T> Mul<Matrix<T>> for Matrix<T>
where
    T: Mul<Output = T> + AddAssign + Copy + Default,
{
    type Output = Matrix<T>;
    mat_mat_mul!(Matrix<T>);
}

impl<T> Mul<&Matrix<T>> for Matrix<T>
where
    T: Mul<Output = T> + AddAssign + Copy + Default,
{
    type Output = Matrix<T>;
    mat_mat_mul!(&Matrix<T>);
}

impl<T> Mul<Matrix<T>> for &Matrix<T>
where
    T: Mul<Output = T> + AddAssign + Copy + Default,
{
    type Output = Matrix<T>;
    mat_mat_mul!(Matrix<T>);
}

impl<T> Mul<&Matrix<T>> for &Matrix<T>
where
    T: Mul<Output = T> + AddAssign + Copy + Default,
{
    type Output = Matrix<T>;
    mat_mat_mul!(&Matrix<T>);
}
