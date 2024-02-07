use crate::algebra::Matrix;
use std::{
    iter::Sum,
    ops::{Add, Index, IndexMut, Mul},
};

pub struct Vector<T> {
    data: Vec<T>,
    shape: usize,
}

impl<T: Mul<Output = T>> Index<usize> for Vector<T> {
    type Output = T;
    fn index(&self, index: usize) -> &T {
        &self.data[index]
    }
}

impl<T: Mul<Output = T>> IndexMut<usize> for Vector<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut self.data[index]
    }
}

impl<T: Copy + Mul<Output = T> + Sum + Default> Vector<T> {
    pub fn new(v: Vec<T>) -> Self {
        Self {
            data: v,
            shape: v.len(),
        }
    }
    pub fn new_uniform(v: T, shape: usize) -> Self {
        Self {
            data: vec![v; shape],
            shape,
        }
    }
    pub fn shape(&self) -> usize {
        self.shape
    }
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.data.iter()
    }
    pub fn to_vec(&self) -> Vector<T> {
        Vector::new(self.data.to_vec())
    }

    pub fn mul_element_wise(&self, rhs: Vector<T>) -> Vector<T> {
        assert_eq!(self.shape, rhs.shape(), "shapes do not match");
        self.iter()
            .zip(rhs.iter())
            .map(|(x, y)| (*x) * (*y))
            .collect()
    }

    pub fn apply<F: Fn(T) -> T>(&self, function: F) -> Vector<T> {
        self.iter().map(|x| function(*x)).collect()
    }

    pub fn inner_prod(&self, rhs: Vector<T>) -> T {
        self.iter().zip(rhs.iter()).map(|(x, y)| (*x) * (*y)).sum()
    }

    pub fn outer_prod(&self, rhs: Vector<T>) -> Matrix<T> {
        let mut output = Matrix::<T>::new_uniform(T::default(), rhs.shape, self.shape);
        for i in 0..self.shape {
            for j in 0..self.shape {
                output[(i, j)] = self[i] * rhs[j];
            }
        }
        output
    }
}

impl<T: Copy + Mul<Output = T> + Add<Output = T> + Sum + Default> Add for Vector<T> {
    type Output = Vector<T>;
    fn add(self, rhs: Vector<T>) -> Vector<T> {
        assert_eq!(self.shape, rhs.shape, "shapes do not match");
        self.iter().zip(rhs.iter()).map(|(x, y)| *x + *y).collect()
    }
}

impl<T: Copy + Mul<Output = T> + Default + Sum> FromIterator<T> for Vector<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Vector<T> {
        let mut a = Vec::<T>::new();
        for i in iter {
            a.push(i);
        }
        Vector::<T>::new(a)
    }
}
