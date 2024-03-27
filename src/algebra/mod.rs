mod matrix;
pub use crate::algebra::matrix::Matrix;
use rand::distributions::uniform::SampleUniform;
use std::ops::{Add, AddAssign, Index, Mul, Sub};

#[allow(clippy::len_without_is_empty)]
pub trait MatLike: Index<(usize, usize)> {
    type Item;
    fn w(&self) -> usize;
    fn h(&self) -> usize;
    fn iter(&self) -> std::slice::Iter<'_, Self::Item>;
    fn len(&self) -> usize {
        self.w() * self.h()
    }
}

pub struct MatSlice<'a, T> {
    mat: &'a Matrix<T>,
    w1: usize,
    h1: usize,
    w2: usize,
    h2: usize,
}

impl<'a, T> Index<(usize, usize)> for MatSlice<'a, T> {
    type Output = T;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        assert_eq!(
            index.1, self.w2,
            "index.1 too big. index.1: {}, self.w2: {}",
            index.1, self.w2
        );
        assert_eq!(
            index.0, self.h2,
            "index.0 too big. index.0: {}, self.h2: {}",
            index.0, self.h2
        );
        &self.mat[(self.h1 + index.0, self.w1 + index.1)]
    }
}

impl<T: Mul<Output = T> + Copy + Clone + PartialOrd + SampleUniform> Matrix<T> {
    pub fn mul_element_wise(&self, rhs: Matrix<T>) -> Matrix<T> {
        assert_eq!(self.w(), rhs.w(), "width does not match");
        assert_eq!(self.h(), rhs.h(), "height does not match");
        Matrix::<T>::new(
            self.iter()
                .zip(rhs.iter())
                .map(|(x, y)| (*x) * (*y))
                .collect(),
            self.w(),
            self.h(),
        )
    }
}

macro_rules! mat_mat_add {
    ($type: ty, $name: ident, $op: tt) => {
        type Output = Matrix<T>;
        fn $name(self, rhs: $type) -> Matrix<T> {
            assert_eq!(self.w(), rhs.w(), "widths do not match");
            assert_eq!(self.h(), rhs.h(), "heights do not match");
            Matrix::<T>::new(
                self
                    .iter()
                    .zip(rhs.iter())
                    .map(|(x, y)| *x $op *y)
                    .collect(),
                self.w(),
                self.h(),
            )
        }
    };
}

macro_rules! mat_mat_mul {
    ($type: ty) => {
        type Output = Matrix<T>;
        fn mul(self, rhs: $type) -> Matrix<T> {
            assert_eq!(
                self.w(),
                rhs.h(),
                "dimensions do not match (lhs w: {}, rhs h: {})",
                self.w(),
                rhs.h()
            );
            let mut output: Matrix<T> = Matrix::new_uniform(T::default(), rhs.w(), self.h());
            for i in 0..self.h() {
                for j in 0..rhs.w() {
                    for k in 0..rhs.h() {
                        output[(i, j)] += self[(i, k)] * rhs[(k, j)];
                    }
                }
            }
            output
        }
    };
}

macro_rules! mat_const_mul {
    () => {
        type Output = Matrix<T>;
        fn mul(self, rhs: T) -> Matrix<T> {
            Matrix::<T>::new(
                self.iter().map(|x| (*x) * rhs).collect(),
                self.w(),
                self.h(),
            )
        }
    };
}

impl<T> Mul<T> for Matrix<T>
where
    T: Copy + Clone + Mul<Output = T> + PartialOrd,
{
    mat_const_mul!();
}

impl<T> Mul<T> for &Matrix<T>
where
    T: Copy + Clone + Mul<Output = T> + PartialOrd,
{
    mat_const_mul!();
}

impl<T> Mul<Matrix<T>> for Matrix<T>
where
    T: Mul<Output = T> + AddAssign + Copy + Default + PartialOrd + SampleUniform,
{
    mat_mat_mul!(Matrix<T>);
}

impl<T> Mul<&Matrix<T>> for Matrix<T>
where
    T: Mul<Output = T> + AddAssign + Copy + Default + PartialOrd,
{
    mat_mat_mul!(&Matrix<T>);
}

impl<T> Mul<Matrix<T>> for &Matrix<T>
where
    T: Mul<Output = T> + AddAssign + Copy + Default + PartialOrd + SampleUniform,
{
    mat_mat_mul!(Matrix<T>);
}

impl<T> Mul<&mut Matrix<T>> for Matrix<T>
where
    T: Mul<Output = T> + AddAssign + Copy + Default + PartialOrd + SampleUniform,
{
    mat_mat_mul!(&mut Matrix<T>);
}

impl<T> Mul<&mut Matrix<T>> for &Matrix<T>
where
    T: Mul<Output = T> + AddAssign + Copy + Default + PartialOrd + SampleUniform,
{
    mat_mat_mul!(&mut Matrix<T>);
}

impl<T> Mul<&Matrix<T>> for &mut Matrix<T>
where
    T: Mul<Output = T> + AddAssign + Copy + Default + PartialOrd + SampleUniform,
{
    mat_mat_mul!(&Matrix<T>);
}

impl<T> Mul<&Matrix<T>> for &Matrix<T>
where
    T: Mul<Output = T> + AddAssign + Copy + Default + PartialOrd + SampleUniform,
{
    mat_mat_mul!(&Matrix<T>);
}

impl<T> Add<Matrix<T>> for Matrix<T>
where
    T: Copy + Default + PartialOrd + SampleUniform + Add<Output = T>,
{
    mat_mat_add!(Matrix<T>,add,  +);
}

impl<T> Add<Matrix<T>> for &Matrix<T>
where
    T: Copy + Default + PartialOrd + SampleUniform + Add<Output = T>,
{
    mat_mat_add!(Matrix<T>, add, +);
}

impl<T> Add<&Matrix<T>> for Matrix<T>
where
    T: Copy + Default + PartialOrd + SampleUniform + Add<Output = T>,
{
    mat_mat_add!(&Matrix<T>, add, +);
}

impl<T> Add<&Matrix<T>> for &Matrix<T>
where
    T: Copy + Default + PartialOrd + SampleUniform + Add<Output = T>,
{
    mat_mat_add!(&Matrix<T>, add, +);
}

impl<T> Sub<Matrix<T>> for Matrix<T>
where
    T: Copy + Default + PartialOrd + SampleUniform + Sub<Output = T>,
{
    mat_mat_add!(Matrix<T>, sub, -);
}

impl<T> Sub<Matrix<T>> for &Matrix<T>
where
    T: Copy + Default + PartialOrd + SampleUniform + Sub<Output = T>,
{
    mat_mat_add!(Matrix<T>, sub, -);
}

impl<T> Sub<&Matrix<T>> for Matrix<T>
where
    T: Copy + Default + PartialOrd + SampleUniform + Sub<Output = T>,
{
    mat_mat_add!(&Matrix<T>, sub, -);
}

impl<T> Sub<&Matrix<T>> for &Matrix<T>
where
    T: Copy + Default + PartialOrd + SampleUniform + Sub<Output = T>,
{
    mat_mat_add!(&Matrix<T>, sub, -);
}
