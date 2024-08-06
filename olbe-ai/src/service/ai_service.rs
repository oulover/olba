use std::fmt::Debug;
use anyhow::Result;
use image::DynamicImage;
use serde::{Deserialize, Serialize};

pub mod r#impl;

pub type OlAiService = std::sync::Arc<dyn AiService>;

#[async_trait::async_trait]
pub trait AiService: Sync + Send + Debug {
    async fn search_face(&self, face_img: DynamicImage)-> Result<Vec<UserFaceFind>>;
    async fn register_face(&self,user_id:i64, face_img: DynamicImage) -> Result<bool>;
}


#[derive(Debug,Clone,Serialize,Deserialize,Default)]
pub struct UserFaceFind{
    pub id: i64,
    pub user_id: i64,
    pub score:f32,
}
#[derive(Debug,Clone)]
pub struct UserFaceCreate{
    pub id: i64,
    pub user_id: i64,
    pub feature:Vec<f32>,
}
