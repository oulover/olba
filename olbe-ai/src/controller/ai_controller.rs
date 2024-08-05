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
use crate::controller::RespVO;
use crate::error::AppError;
use crate::service::ai_service::{OlAiService, UserFaceFind};

pub(crate) fn router() -> Router {
    Router::new()
        .route("/search", post(search))
        .route("/upload", post(upload_resp))
}



pub async fn search(Extension(context): Extension<Arc<AppContext>>, mut multipart: Multipart) -> Result<Json<Vec<UserFaceFind>>, AppError> {
    let file = multipart.next_field().await?.ok_or(AppError::NotFound)?;
    let bb = file.bytes().await?;
    let bytes = bb.to_vec();
    let img = image::load_from_memory(&bytes)?;
    let service: OlAiService = context.container.resolve().await?;
    let r = service.search_face(img).await?;
    Ok(Json(r))
}

pub async fn upload_resp(Extension(context): Extension<Arc<AppContext>>, multipart: Multipart) -> impl IntoResponse {
    RespVO::from_result(upload(context, multipart).await).json()
}
pub async fn upload(context: Arc<AppContext>, mut multipart: Multipart) -> Result<Vec<UserFaceFind>> {
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
            Ok(service.search_face(img).await?)
        } else {
            Err(anyhow::Error::msg("123").into())
        }
    } else {
        Err(anyhow::Error::msg("123").into())
    }
}




