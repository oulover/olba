use std::fmt::{Debug, Display, Formatter};
use std::str::FromStr;
use axum::response::IntoResponse;
use axum::{Extension, Router};
use axum::extract::Multipart;
use axum::routing::post;
use anyhow::Result;
use axum::body::Bytes;
use log::Level::Error;
use crate::AppContext;
use crate::controller::RespVO;
use crate::service::ai_service::{OlAiService, UserFaceFind};

pub(crate) fn router() -> Router {
    Router::new()
        .route("/search", post(search_resp))
        .route("/upload", post(upload_resp))
}

pub async fn search_resp(Extension(context): Extension<AppContext>, multipart: Multipart) -> impl IntoResponse {
    RespVO::from_result(search(context, multipart).await).json()
}
pub async fn search(context: AppContext, mut multipart: Multipart) -> Result<Vec<UserFaceFind>> {
    let file = multipart.next_field().await?.unwrap();
    let bb = file.bytes().await?;
    let bytes = bb.to_vec();
    let img = image::load_from_memory(&bytes)?;
    let service: OlAiService = context.container.resolve().await?;
    service.search_face(img).await
}

pub async fn upload_resp(Extension(context): Extension<AppContext>, multipart: Multipart) -> impl IntoResponse {
    RespVO::from_result(upload(context, multipart).await).json()
}
pub async fn upload(context: AppContext, mut multipart: Multipart) -> Result<Vec<UserFaceFind>> {
    let mut uid:Option<String> =None;
    let mut img:Option<Bytes> =None;
    while let Some(field) = multipart.next_field().await? {
       if let Some(name) = field.name(){
           if name.eq("user_id"){
               uid = Some(field.text().await?)
           }else if name.eq("image"){
               img = Some(field.bytes().await?)
           }
       }
    }
    if let Some(uid) = uid {
        if let Some(img) = img {
            let uid: i64 = i64::from_str(&uid)?;
            let bytes = img.to_vec();
            let img = image::load_from_memory(&bytes)?;
            let service: OlAiService = context.container.resolve().await?;
            Ok(service.search_face(img).await?)
        } else {
             Err(anyhow::Error::msg("123"))
        }
    } else {
        Err(anyhow::Error::msg("123"))
    }

}
pub struct MyError;



