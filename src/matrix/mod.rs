use std::ops::{Mul, Index, IndexMut};
mod multiply;

pub struct Matrix<T: Mul> {
    data: Vec<T>,
    w: usize,
    h: usize,
}

impl<T: Mul<Output = T> + Copy> Matrix<T> {
    pub fn new(input: Vec<T>, w: usize, h: usize) -> Self    { Self { data: input, w, h } }
    pub fn new_uniform(val: T, w: usize, h: usize) -> Self   { Self { data: vec![val; w * h], w, h} }
}

impl<T: Mul<Output = T>> Index<(usize, usize)> for Matrix<T> {
    type Output = T;
    fn index(&self, index: (usize, usize)) -> &T {
        &self.data[index.0 * self.h + index.1]
    }
}

impl<T: Mul<Output = T>> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut T {
        &mut self.data[index.0 * self.h + index.1]
    }
}
