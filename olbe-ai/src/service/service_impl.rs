use std::fmt::{Debug, Formatter};
use axum::extract::Multipart;
use opencv::imgcodecs;
use opencv::imgcodecs::IMREAD_COLOR;
use crate::ai::AiSession;
use crate::service::service::DemoService;

#[derive(Debug)]
pub(crate) struct DemoServiceImpl;

impl DemoService for DemoServiceImpl {
    async fn set_string(&self) -> anyhow::Result<String> {
        AiSession::get_instance()?.get()?;
        Ok(String::from("-----demo-----"))
    }

    async fn upload_file(&self, multipart: Multipart) -> anyhow::Result<String> {
        upload_file2(multipart).await

        //Ok(String::from("-----demo-----"))
    }
}

use opencv::{highgui};

pub async fn upload_file2(mut multipart: Multipart) -> anyhow::Result<String> {
    let file = multipart.next_field().await;

    let ai = crate::ai::AiSession::get_instance()?;

    match file {
        Ok(ff) => {
            let tt = ff.unwrap();

            println!("ff---{:?}----{:?}-", tt.name(), tt.file_name());
            // tt.

            let bb = tt.bytes().await?;
            let bytes = bb.to_vec();
            // image::open(Path::new(f)).unwrap();

            let img = image::load_from_memory(&bytes)?;
            let face_boxes = ai.get_face_boxes(&img)?;
            let mut face_img = AiSession::get_face_img(&img, &face_boxes);
            if face_img.len() > 0 {
                for x in &face_img {
                    let feat = ai.get_face_feature(x)?;
                    println!("{:?}", feat)
                }
            }
            if face_img.len() > 1 {
                let t1 = face_img.pop().unwrap();
                let t2 = face_img.pop().unwrap();

               let sim=  ai.get_sim(&t1,&t2)?;

                println!("------{sim}")
            }


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