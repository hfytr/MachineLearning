use std::{
    collections::{HashMap, HashSet},
    fs::read_to_string,
};

use crate::algebra::Matrix;

pub enum StringEncode {
    Integer,
    OneHot,
}

#[derive(Debug)]
pub enum DataType {
    Categorical(String),
    Numerical(f64),
}

pub struct Dataset {
    data: HashMap<String, Vec<DataType>>,
    keys: Vec<String>,
}

impl Dataset {
    pub fn to_matrix(&self, columns: &[String]) -> Matrix<f64> {
        Matrix::new_from_2d(columns.iter().map(|x| {
            self.data.get(x).unwrap_or_else(|| panic!("failed to get {x}")).iter().map(|y| match y {
                DataType::Numerical(value) => *value,
                DataType::Categorical(_) => { panic!("Categorical variables in dataset. Please use Dataset::encode before trying to get a matrix."); }
            }).collect()
        }).collect(), false)
    }

    pub fn keys(&self) -> &[String] {
        &self.keys
    }

    pub fn from_csv(path: &String) -> Dataset {
        let file_string: String = read_to_string(path).expect("couldn't read {path}");
        let mut lines: Vec<Vec<String>> = file_string
            .split('\n')
            .map(|x| {
                x.split(',')
                    .map(|x| x.trim().to_string())
                    .collect::<Vec<String>>()
            })
            .collect();
        let keys: Vec<String> = lines.remove(0);
        let mut data: HashMap<String, Vec<DataType>> = HashMap::new();
        for (i, col) in keys.iter().enumerate().rev() {
            let mut column: Vec<DataType> = Vec::new();
            for j in lines.iter_mut() {
                // guard against empty/incomplete lines
                if j.len() != i + 1 {
                    j.pop(); // ensures that the line cannot be accepted on a future iteration
                    continue;
                }
                let moved: String = j.pop().unwrap();
                let parsed = moved.parse::<f64>();
                column.push(match parsed {
                    Result::Err(_) => DataType::Categorical(moved),
                    Result::Ok(result) => DataType::Numerical(result),
                })
            }
            data.insert(col.to_string(), column);
        }
        Dataset { data, keys }
    }

    pub fn encode(&mut self, encoding: StringEncode) {
        let categorical: Vec<&String> = self
            .keys
            .iter()
            .filter(|&x| match self.data.get(x).expect("failed to get {x}")[0] {
                DataType::Numerical(_) => false,
                DataType::Categorical(_) => true,
            })
            .collect();
        match encoding {
            StringEncode::OneHot => {
                for i in categorical.into_iter() {
                    let unique: Vec<String> = self.unique_values(i);
                    for j in unique.into_iter() {
                        self.data
                            .insert(j.clone(), self.is_in_category(i.to_string(), j));
                    }
                }
            }
            StringEncode::Integer => {
                for i in categorical.into_iter() {
                    let suffix: &str = " - encoded";
                    let unique: Vec<String> = self.unique_values(i);
                    self.data
                        .insert(i.to_owned() + suffix, self.integer_encode(i, unique));
                }
            }
        }
    }

    fn integer_encode(&self, column: &String, unique: Vec<String>) -> Vec<DataType> {
        let mut unique_hashmap: HashMap<String, usize> = HashMap::new();
        for i in unique.into_iter().enumerate() {
            unique_hashmap.insert(i.1, i.0);
        }
        self.data
            .get(column)
            .expect("failed to get {column}")
            .to_owned()
            .iter()
            .map(|x| {
                if let DataType::Categorical(y) = x {
                    DataType::Numerical(*unique_hashmap.get(y).expect("failed to get {y}") as f64)
                } else {
                    unreachable!()
                }
            })
            .collect()
    }

    fn is_in_category(&self, column: String, category: String) -> Vec<DataType> {
        self.data
            .get(&column)
            .expect("failed to get {&column}")
            .iter()
            .map(|x| {
                if let DataType::Categorical(y) = x {
                    if *y == category {
                        DataType::Numerical(1.0)
                    } else {
                        DataType::Numerical(0.0)
                    }
                } else {
                    unreachable!()
                }
            })
            .collect()
    }

    fn unique_values(&self, column: &str) -> Vec<String> {
        let mut hashset = HashSet::<String>::new();
        for j in self
            .data
            .get(column)
            .expect("failed to get {column}")
            .iter()
        {
            if let DataType::Categorical(x) = j {
                hashset.insert(x.to_string());
            }
        }
        Vec::from_iter(hashset)
    }
}
