use std::{
    collections::{HashMap, HashSet},
    fs::read_to_string,
};

use crate::algebra::Matrix;

#[derive(Debug, Clone, PartialEq)]
pub enum DataType {
    Categorical(String),
    Numerical(f64),
}

#[derive(Debug)]
pub struct Dataset {
    data: HashMap<String, usize>,
    container: Vec<Vec<DataType>>,
    keys: Vec<String>,
}

impl Dataset {
    pub fn to_matrix(&self, columns: &[String]) -> Matrix<f64> {
        Matrix::new_from_2d(columns.iter().map(|x| {
            self.get_col(x).unwrap_or_else(|| panic!("failed to get {x}")).iter().map(|y| match y {
                DataType::Numerical(value) => *value,
                DataType::Categorical(_) => { panic!("Categorical variables in dataset. Please use encode data before trying to get a matrix."); }
            }).collect()
        }).collect(), false)
    }

    pub fn get_col(&self, col: &str) -> Option<&[DataType]> {
        Some(&self.container[*self.data.get(col)?])
    }

    pub fn get_row(&self, row: usize) -> Vec<&DataType> {
        self.keys()
            .iter()
            .map(|x| &self.get_col(x).unwrap()[row])
            .collect()
    }

    pub fn keys(&self) -> &[String] {
        &self.keys
    }

    pub fn blazing_from_csv(path: &String) -> Dataset {
        let file_string = read_to_string(path).expect("couldn't read {path}");
        let mut file_iter = file_string.chars();
        let mut c = file_iter.next().unwrap();
        let mut keys: Vec<String> = Vec::new();
        let mut data: HashMap<String, usize> = HashMap::new();
        let mut s: String = String::new();
        while c != '\n' {
            if c != ',' {
                s.push(c);
            } else {
                keys.push(s.trim().to_string());
                data.insert(keys.last().unwrap().to_string(), keys.len() - 1);
            }
            c = file_iter.next().unwrap();
        }
        println!("{:?}", keys);
        let mut container: Vec<Vec<DataType>> = Vec::new();
        let mut cur_line: Vec<DataType> = Vec::new();
        let mut cur_entry = String::new();
        for c in file_iter {
            if c == ',' || c == '\n' {
                cur_entry = cur_entry.trim().to_string();
                cur_line.push(match cur_entry.parse::<f64>() {
                    Result::Ok(x) => DataType::Numerical(x),
                    Result::Err(_) => DataType::Categorical(cur_entry),
                });
                cur_entry = String::new();
            } else {
                cur_entry.push(c);
            }
            if c == '\n' {
                if cur_line.len() == keys.len() {
                    for i in (0..cur_line.len()).rev() {
                        container[i].push(cur_line.pop().unwrap());
                    }
                }
                cur_line.clear();
                cur_entry.clear();
            }
        }
        Dataset {
            data,
            container,
            keys,
        }
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
        let mut data: HashMap<String, usize> = HashMap::new();
        let mut container: Vec<Vec<DataType>> = Vec::new();
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
            data.insert(col.to_string(), container.len());
            container.push(column);
        }
        Dataset {
            data,
            container,
            keys,
        }
    }

    pub fn one_hot_encode(&mut self, col: &str, categories: &[DataType]) -> Vec<String> {
        let mut new_keys = Vec::<String>::new();
        for category in categories.iter() {
            let mut new_col = col.to_string();
            new_col.push_str(" - ");
            new_col.push_str(&match category {
                DataType::Numerical(x) => x.to_string(),
                DataType::Categorical(ref x) => x.to_string(),
            });
            self.data.insert(new_col.clone(), self.container.len());
            self.container.push(
                self.is_in_category(col, category)
                    .expect("Invalid column name"),
            );
            new_keys.push(new_col);
        }
        new_keys
    }

    fn is_in_category(&self, col: &str, category: &DataType) -> Option<Vec<DataType>> {
        Some(
            self.get_col(col)?
                .iter()
                .map(|x| {
                    if x == category {
                        DataType::Numerical(1.0)
                    } else {
                        DataType::Numerical(0.0)
                    }
                })
                .collect(),
        )
    }

    fn integer_encode(&mut self, col: &String) {
        let unique: Vec<String> = self
            .unique_values(col, 0)
            .unwrap()
            .into_iter()
            .map(|x| {
                if let DataType::Categorical(y) = x {
                    y
                } else {
                    unreachable!()
                }
            })
            .collect();
        let mut unique_hashmap: HashMap<String, usize> = HashMap::new();
        for i in unique.into_iter().enumerate() {
            unique_hashmap.insert(i.1, i.0);
        }
        col.clone().push_str(" - encoded");
        self.data.insert(col.to_string(), self.container.len());
        self.container.push(
            self.get_col(col)
                .unwrap()
                .iter()
                .map(|x| {
                    if let DataType::Categorical(y) = x {
                        DataType::Numerical(
                            *unique_hashmap.get(y).expect("failed to get {y}") as f64
                        )
                    } else {
                        unreachable!()
                    }
                })
                .collect(),
        )
    }

    fn unique_values(&self, col_name: &str, accuracy: i32) -> Option<Vec<DataType>> {
        let mut unique = HashSet::<String>::new();
        let col = self.get_col(col_name)?;
        let is_num = match col[0] {
            DataType::Categorical(_) => false,
            DataType::Numerical(_) => true,
        };
        for i in col {
            unique.insert(match i {
                DataType::Categorical(x) => x.to_string(),
                DataType::Numerical(x) => Self::string_with_accuracy(*x, accuracy),
            });
        }
        Some(
            unique
                .into_iter()
                .map(|x| {
                    if is_num {
                        DataType::Numerical(x.parse().unwrap())
                    } else {
                        DataType::Categorical(x)
                    }
                })
                .collect(),
        )
    }

    fn string_with_accuracy(x: f64, accuracy: i32) -> String {
        ((x * 10_f64.powi(accuracy)).round() / 10_f64.powi(accuracy)).to_string()
    }
}
