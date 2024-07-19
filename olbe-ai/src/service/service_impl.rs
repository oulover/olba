use std::fmt::{Debug, Formatter};
use axum::extract::Multipart;
use opencv::imgcodecs;
use opencv::imgcodecs::IMREAD_COLOR;
use crate::ai::AiSession;
use crate::service::service::DemoService;

#[derive(Debug)]
pub(crate) struct DemoServiceImpl;

impl DemoService for DemoServiceImpl{
    async fn set_string(&self) -> anyhow::Result<String> {
        AiSession::get_instance()?.get()?;
        Ok(String::from("-----demo-----"))
    }

    async fn upload_file(&self, multipart: Multipart) -> anyhow::Result<String> {
        upload_file2(multipart).await

        //Ok(String::from("-----demo-----"))
    }
}
use opencv::{highgui  };
pub async fn upload_file2(mut multipart: Multipart) -> anyhow::Result<String> {
    let file = multipart.next_field().await;

    match file {
        Ok(ff) => {
            let tt = ff.unwrap();

            println!("ff---{:?}----{:?}-", tt.name(), tt.file_name());
            // tt.

            let bb = tt.bytes().await?;
            let bytes = bb.to_vec();
            // image::open(Path::new(f)).unwrap();

            // let img =  image::load_from_memory(&bytes)?;

            let mat = imgcodecs::imdecode(&bytes.as_slice(), IMREAD_COLOR)?;
            highgui::named_window("hello opencv!", 0)?;
            highgui::imshow("hello opencv!", &mat)?;
            highgui::wait_key(10000)?;
            println!("--{:?}", String::from_utf8(bb.to_vec())?);
        }
        Err(_) => {}
    }

    Ok(String::from("123"))
}