use serde::{Deserialize, Serialize};
use ndarray::Array1;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct User {
    pub id: i32,
    pub user_face_bin: Vec<u8>,
}

impl User {
    pub fn new(id: i32, face_features: Array1<f32>) -> Self {
        let face_bin = bincode::serialize(&face_features.to_vec()).expect("Failed to serialize face features");
        User {
            id,
            user_face_bin: face_bin,
        }
    }

    pub fn get_face_features(&self) -> Array1<f32> {
       let v:Vec<f32> =  bincode::deserialize(&self.user_face_bin).expect("Failed to deserialize face features");
        Array1::from_vec(v)
    }
}