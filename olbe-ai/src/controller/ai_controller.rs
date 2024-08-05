use std::fmt::{Debug, Display};
use std::str::FromStr;
use std::sync::Arc;

use crate::error::Result;
use axum::{Extension, Json, Router};
use axum::body::Bytes;
use axum::extract::Multipart;
use axum::response::IntoResponse;
use axum::routing::post;

use crate::AppContext;
use crate::controller::R;
use crate::error::AppError;
use crate::service::ai_service::{OlAiService, UserFaceFind};

pub(crate) fn router() -> Router {
    Router::new()
        .route("/search", post(search))
        .route("/upload", post(upload))
}



pub async fn search(Extension(context): Extension<Arc<AppContext>>, mut multipart: Multipart) -> Result<R<Vec<UserFaceFind>>> {
    let file = multipart.next_field().await?.ok_or(AppError::NotFound)?;
    let bb = file.bytes().await?;
    let bytes = bb.to_vec();
    let img = image::load_from_memory(&bytes)?;
    let service: OlAiService = context.container.resolve().await?;
    let r = service.search_face(img).await?;
    Ok(R::ok(Some(r)))
}


pub async fn upload(Extension(context): Extension<Arc<AppContext>>, mut multipart: Multipart)  -> Result<R<bool>>{
    let mut uid: Option<String> = None;
    let mut img: Option<Bytes> = None;
    while let Some(field) = multipart.next_field().await? {
        if let Some(name) = field.name() {
            if name.eq("user_id") {
                uid = Some(field.text().await?)
            } else if name.eq("image") {
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
            Ok(R::ok(Some(service.register_face(uid,img).await?)))
        } else {
            Err(anyhow::Error::msg("123").into())
        }
    } else {
        Err(anyhow::Error::msg("123").into())
    }
}




