use rand::{distributions::uniform::SampleUniform, Rng};
use std::{
    fmt::Display,
    iter::Sum,
    ops::{Add, AddAssign, Index, IndexMut, Mul},
};

#[derive(Clone, Default)]
pub struct Matrix<T> {
    data: Vec<T>,
    w: usize,
    h: usize,
}

impl<T: Display> Display for Matrix<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Matrix: w: {}, h: {} [\n    ", self.w, self.h)?;
        for (i, x) in self.data.iter().enumerate() {
            write!(f, "{}, ", x)?;
            if (i + 1) % self.w == 0 && i != 0 {
                writeln!(f)?;
                if i != self.data.len() - 1 {
                    write!(f, "    ")?;
                }
            }
        }
        writeln!(f, "]")?;
        Ok(())
    }
}

impl<T: Mul<Output = T> + Copy> Matrix<T> {
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

    pub fn w(&self) -> usize {
        self.w
    }
    pub fn h(&self) -> usize {
        self.h
    }

    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.data.iter()
    }

    pub fn apply<F: Fn(T) -> T>(&self, function: F) -> Matrix<T> {
        Self {
            data: self.iter().map(|x| function(*x)).collect(),
            w: self.w,
            h: self.h,
        }
    }
}

impl<T> Matrix<T>
where
    T: Mul<Output = T> + Copy + PartialOrd + SampleUniform + Default + Sum,
{
    pub fn random(low: T, high: T, w: usize, h: usize) -> Matrix<T> {
        let mut rng = rand::thread_rng();
        Self {
            data: (0..(w * h)).map(|_| rng.gen_range(low..high)).collect(),
            w,
            h,
        }
    }

    pub fn clone_row(&self, i: usize) -> Matrix<T> {
        Matrix::new(
            self.data[(self.w * i)..(self.w * (i + 1))].to_vec(),
            self.w,
            1,
        )
    }

    pub fn mul_element_wise(&self, rhs: Matrix<T>) -> Matrix<T> {
        assert_eq!(self.w, rhs.w(), "width does not match");
        assert_eq!(self.h, rhs.h(), "height does not match");
        Self {
            data: self
                .iter()
                .zip(rhs.iter())
                .map(|(x, y)| (*x) * (*y))
                .collect(),
            w: self.w,
            h: self.h,
        }
    }
}

// row major
impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;
    fn index(&self, index: (usize, usize)) -> &T {
        assert!(
            index.0 * self.w + index.1 < self.data.len(),
            "index does not match, i: {}, j: {}, w: {}, h: {}",
            index.0,
            index.1,
            self.w,
            self.h,
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

impl<T> Index<usize> for Matrix<T> {
    type Output = T;
    fn index(&self, index: usize) -> &T {
        assert!(
            index < self.data.len(),
            "index does not match, i: {}, len: {}",
            index,
            self.data.len()
        );
        &self.data[index]
    }
}

impl<T> IndexMut<usize> for Matrix<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        assert!(
            index < self.data.len(),
            "index does not match, i: {}, len: {}",
            index,
            self.data.len()
        );
        &mut self.data[index]
    }
}

macro_rules! mat_mat_add {
    ($type: ty) => {
        type Output = Matrix<T>;
        fn add(self, rhs: $type) -> Matrix<T> {
            assert_eq!(self.w, rhs.w, "widths do not match");
            assert_eq!(self.h, rhs.h, "heights do not match");
            Matrix::<T>::new(
                self.data
                    .iter()
                    .zip(rhs.data.iter())
                    .map(|(x, y)| *x + *y)
                    .collect(),
                self.w,
                self.h,
            )
        }
    };
}

macro_rules! mat_mat_mul {
    ($type: ty) => {
        type Output = Matrix<T>;
        fn mul(self, rhs: $type) -> Matrix<T> {
            assert_eq!(
                self.w, rhs.h,
                "dimensions do not match (lhs w: {}, rhs h: {})",
                self.w, rhs.h
            );
            let mut output: Matrix<T> = Matrix::new_uniform(T::default(), rhs.w, self.h);
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

macro_rules! mat_const_mul {
    () => {
        type Output = Matrix<T>;
        fn mul(self, rhs: T) -> Matrix<T> {
            Matrix::<T>::new(
                self.data.iter().map(|x| (*x) * rhs).collect(),
                self.w,
                self.h,
            )
        }
    };
}

impl<T: Copy + Mul<Output = T> + Add<Output = T> + Sum + Default> Mul<T> for Matrix<T> {
    mat_const_mul!();
}

impl<T: Copy + Mul<Output = T> + Add<Output = T> + Sum + Default> Mul<T> for &Matrix<T> {
    mat_const_mul!();
}

impl<T> Mul<Matrix<T>> for Matrix<T>
where
    T: Mul<Output = T> + AddAssign + Copy + Default,
{
    mat_mat_mul!(Matrix<T>);
}

impl<T> Mul<&Matrix<T>> for Matrix<T>
where
    T: Mul<Output = T> + AddAssign + Copy + Default,
{
    mat_mat_mul!(&Matrix<T>);
}

impl<T> Mul<Matrix<T>> for &Matrix<T>
where
    T: Mul<Output = T> + AddAssign + Copy + Default,
{
    mat_mat_mul!(Matrix<T>);
}

impl<T> Mul<&mut Matrix<T>> for Matrix<T>
where
    T: Mul<Output = T> + AddAssign + Copy + Default,
{
    mat_mat_mul!(&mut Matrix<T>);
}

impl<T> Mul<&mut Matrix<T>> for &Matrix<T>
where
    T: Mul<Output = T> + AddAssign + Copy + Default,
{
    mat_mat_mul!(&mut Matrix<T>);
}

impl<T> Mul<&Matrix<T>> for &mut Matrix<T>
where
    T: Mul<Output = T> + AddAssign + Copy + Default,
{
    mat_mat_mul!(&Matrix<T>);
}

impl<T> Mul<&Matrix<T>> for &Matrix<T>
where
    T: Mul<Output = T> + AddAssign + Copy + Default,
{
    mat_mat_mul!(&Matrix<T>);
}

impl<T: Add<Output = T> + Copy + Mul<Output = T>> Add<Matrix<T>> for Matrix<T> {
    mat_mat_add!(Matrix<T>);
}

impl<T: Add<Output = T> + Copy + Mul<Output = T>> Add<Matrix<T>> for &Matrix<T> {
    mat_mat_add!(Matrix<T>);
}

impl<T: Add<Output = T> + Copy + Mul<Output = T>> Add<&Matrix<T>> for Matrix<T> {
    mat_mat_add!(&Matrix<T>);
}

impl<T: Add<Output = T> + Copy + Mul<Output = T>> Add<&Matrix<T>> for &Matrix<T> {
    mat_mat_add!(&Matrix<T>);
}
