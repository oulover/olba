use std::fmt::{Debug, Formatter};
use axum::extract::Multipart;
use image::DynamicImage;
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

use opencv::{
    core::{self, Mat, CV_8UC3},

    prelude::*,
};
fn dynamic_image_to_mat(image: &DynamicImage) -> Result<Mat, Box<dyn std::error::Error>> {
    // Convert the image to RGB8 format
    let rgb_image = image.to_rgb8();

    // Get the dimensions of the image
    let width = rgb_image.width() as i32;
    let height = rgb_image.height() as i32;

    // Create a Mat with the same dimensions and type
    let mut mat = Mat::new_rows_cols_with_default(height, width, CV_8UC3, opencv::core::Scalar::default())?;

    // Copy the pixel data from the image to the Mat
    for y in 0..height {
        for x in 0..width {
            let pixel = rgb_image.get_pixel(x as u32, y as u32);
            let b = pixel[0] as u8;
            let g = pixel[1] as u8;
            let r = pixel[2] as u8;
            let data = [r, g, b];
            mat.at_2d_mut::<core::Vec3b>(y, x)?.copy_from_slice(&data);
        }
    }

    Ok(mat)
}

pub async fn upload_file2(mut multipart: Multipart) -> anyhow::Result<String> {
    let file = multipart.next_field().await;

    let ai = AiSession::get_instance()?;

    match file {
        Ok(ff) => {
            let tt = ff.unwrap();

            //println!("ff---{:?}----{:?}-", tt.name(), tt.file_name());
            // tt.

            let bb = tt.bytes().await?;
            let bytes = bb.to_vec();
            // image::open(Path::new(f)).unwrap();

            let img = image::load_from_memory(&bytes)?;
            let face_boxes = ai.get_face_boxes(&img)?;
            let mut face_img = AiSession::get_face_img(&img, &face_boxes);
            if face_img.len() > 0 {
                for x in &face_img {
                    // let feat = ai.get_face_feature(x)?;
                    // //println!("{:?}", feat)

                    let mat = dynamic_image_to_mat(x).unwrap();

                    // let mat = imgcodecs::imdecode(&x., IMREAD_COLOR)?;
                    highgui::named_window("hello opencv!", 0)?;
                    highgui::imshow("hello opencv!", &mat)?;
                    highgui::wait_key(3000)?;
                }
            }
            if face_img.len() > 1 {
                let t1 = face_img.pop().unwrap();
                let t2 = face_img.pop().unwrap();

               let sim=  ai.get_sim(&t1,&t2)?;

                println!("------{sim}")
            }




        }
        Err(_) => {}
    }

    Ok(String::from("123"))
}