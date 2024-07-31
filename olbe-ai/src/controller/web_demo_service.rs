use axum::response::{IntoResponse, Response};
use axum::{Form, Router};
use axum::extract::Multipart;
use axum::extract::multipart::{Field, MultipartError};
use axum::routing::{get, post};
use serde::Deserialize;
use crate::controller::RespVO;
use crate::service::service::DemoService;
use crate::service::service_impl::DemoServiceImpl;

pub(crate) fn router() -> Router {
    Router::new()
        .route("/get_info", get(get_info))
        .route("/upload_file", post(upload_file))
        .route("/register_face_web", post(register_face_web))
}


pub async fn get_info() -> impl IntoResponse {
    let svc = DemoServiceImpl;
    let r = svc.set_string().await;

    RespVO::from_result(r).json()
}

// #[derive(Deserialize)]
// struct UploadForm {
//     file: axum::extract::multipart::Multipart,
// }


pub async fn upload_file2(multipart: Multipart) -> anyhow::Result<String> {
    // let file = multipart.next_field().await;
    //
    // match file {
    //     Ok(ff) => {
    //         let tt = ff.unwrap();
    //
    //         //println!("ff---{:?}----{:?}-", tt.name(), tt.file_name());
    //         // tt.
    //
    //         let bb = tt.bytes().await?;
    //         //println!("--{:?}", String::from_utf8(bb.to_vec())?);
    //     }
    //     Err(_) => {}
    // }
    //
    // Ok(String::from("123"))

    let svc = DemoServiceImpl;
    svc.upload_file(multipart).await
}
pub async fn register_face(multipart: Multipart) -> anyhow::Result<Vec<f32>> {
    // let file = multipart.next_field().await;
    //
    // match file {
    //     Ok(ff) => {
    //         let tt = ff.unwrap();
    //
    //         //println!("ff---{:?}----{:?}-", tt.name(), tt.file_name());
    //         // tt.
    //
    //         let bb = tt.bytes().await?;
    //         //println!("--{:?}", String::from_utf8(bb.to_vec())?);
    //     }
    //     Err(_) => {}
    // }
    //
    // Ok(String::from("123"))

    let svc = DemoServiceImpl;
    svc.register_face(multipart).await
}
pub async fn register_face_web(mut multipart: Multipart) -> impl IntoResponse {
    RespVO::from_result(register_face(multipart).await).json()
}


pub async fn upload_file(mut multipart: Multipart) -> impl IntoResponse {
    RespVO::from_result(upload_file2(multipart).await).json()
}


