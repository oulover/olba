use axum::response::IntoResponse;
use axum::Router;
use axum::extract::Multipart;
use axum::routing::{get, post};
use crate::controller::R;
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

    R::result(r)
}

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
pub async fn register_face_web(multipart: Multipart) -> impl IntoResponse {
    R::result(register_face(multipart).await)
}


pub async fn upload_file(multipart: Multipart) -> impl IntoResponse {
    R::result(upload_file2(multipart).await)
}


