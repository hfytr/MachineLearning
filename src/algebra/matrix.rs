use super::MatLike;
use rand::{distributions::uniform::SampleUniform, Rng};
use std::fmt::Display;
use std::ops::{Index, IndexMut};
#[derive(Clone, Default)]
pub struct Matrix<T> {
    data: Vec<T>,
    w: usize,
    h: usize,
}

impl<T> MatLike for Matrix<T> {
    type Item = T;
    fn w(&self) -> usize {
        self.w
    }
    fn h(&self) -> usize {
        self.h
    }
    fn iter(&self) -> std::slice::Iter<'_, T> {
        self.data.iter()
    }
}

// row major
impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;
    fn index(&self, index: (usize, usize)) -> &T {
        assert!(
            index.1 < self.w,
            "index.1 too big. index.1: {}, self.w: {}",
            index.1,
            self.w
        );
        assert!(
            index.0 < self.h,
            "index.0 too big. index.0: {}, self.h: {}",
            index.0,
            self.h
        );
        &self.data[index.0 * self.w + index.1]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut T {
        assert!(
            index.0 * self.w + index.1 < self.data.len(),
            "index does not match, i: {}, j: {}, w: {}, h: {}",
            index.0,
            index.1,
            self.w,
            self.h,
        );
        &mut self.data[index.0 * self.w + index.1]
    }
}

impl<T: Display> Display for Matrix<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Matrix: w: {}, h: {} [\n    ", self.w(), self.h())?;
        for (i, x) in self.iter().enumerate() {
            write!(f, "{}, ", x)?;
            if (i + 1) % self.w() == 0 && i != 0 {
                writeln!(f)?;
                if i != self.len() - 1 {
                    write!(f, "    ")?;
                }
            }
        }
        writeln!(f, "]")?;
        Ok(())
    }
}

impl<T: Copy + Clone + PartialOrd> Matrix<T> {
    pub fn new(data: Vec<T>, w: usize, h: usize) -> Self {
        assert_eq!(w * h, data.len(), "input size does not match shape");
        Self { data, w, h }
    }

    pub fn new_from_2d(data: Vec<Vec<T>>, row_major: bool) -> Self {
        if row_major {
            Self {
                w: data[0].len(),
                h: data.len(),
                data: data.into_iter().flatten().collect(),
            }
        } else {
            Self {
                w: data[0].len(),
                h: data.len(),
                data: data.into_iter().flatten().collect(),
            }
            .new_transposed()
        }
    }

    pub fn new_transposed(&self) -> Matrix<T> {
        let mut positions: Vec<usize> = (0..self.h).map(|x| x * self.w).collect();
        Self {
            w: self.h,
            h: self.w,
            data: (0..self.w)
                .flat_map(|_| {
                    positions
                        .iter_mut()
                        .map(|x| {
                            *x += 1;
                            self.data[*x - 1]
                        })
                        .collect::<Vec<T>>()
                })
                .collect(),
        }
    }

    pub fn transpose(&mut self) -> &mut Matrix<T> {
        std::mem::swap(&mut self.w, &mut self.h);
        if self.w == 1_usize || self.h == 1_usize {
            return self;
        }
        self.data = self.new_transposed().data;
        self
    }

    pub fn new_uniform(val: T, w: usize, h: usize) -> Self {
        Self {
            data: vec![val; w * h],
            w,
            h,
        }
    }

    pub fn apply<F: Fn(T) -> T>(&self, function: F) -> Matrix<T> {
        Self {
            data: self.iter().map(|x| function(*x)).collect(),
            w: self.w,
            h: self.h,
        }
    }

    pub fn clone_row(&self, i: usize) -> Matrix<T> {
        Matrix::new(
            self.data[(self.w * i)..(self.w * (i + 1))].to_vec(),
            self.w,
            1,
        )
    }
}

impl<T: Copy + SampleUniform + Clone + PartialOrd> Matrix<T> {
    pub fn random(low: T, high: T, w: usize, h: usize) -> Matrix<T> {
        let mut rng = rand::thread_rng();
        Self {
            data: (0..(w * h)).map(|_| rng.gen_range(low..high)).collect(),
            w,
            h,
        }
    }
}
