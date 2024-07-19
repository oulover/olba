use std::fmt::Debug;
use anyhow::Result;
use axum::extract::Multipart;

pub trait DemoService: Sync + Send + Debug{
    async fn set_string(&self ) -> Result<String>;
    async fn upload_file(&self, multipart: Multipart) -> Result<String>;
}