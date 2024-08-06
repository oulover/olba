use std::borrow::Cow;
use std::fmt::Debug;
use std::time::Instant;

use axum::extract::Multipart;
use image::DynamicImage;
use milvus::client::Client;
use milvus::index::MetricType;
use milvus::query::SearchOptions;
use milvus::value::Value;
use opencv::{
    core::{self, CV_8UC3, Mat},
    prelude::*,
};

use crate::ai::AiSession;
use crate::ai::milvus::UserFaceFeature;
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
    }

    async fn register_face(&self, mut multipart: Multipart) -> anyhow::Result<Vec<f32>> {
        let file = multipart.next_field().await?;


        let client = Client::new("http://120.46.194.67:19530").await?;


        let ai = AiSession::get_instance()?;

        match file {
            None => {}
            Some(tt) => {
                let bb = tt.bytes().await?;
                let bytes = bb.to_vec();


                let img = image::load_from_memory(&bytes)?;

                let face_boxes = ai.get_face_boxes(&img)?;

                let mut face_img = ai.get_face_img(&img, &face_boxes);

                let r = face_img.pop().unwrap();

                let feature = ai.get_face_feature(&r).unwrap();
                let feature: Vec<_> = feature.iter().map(|o| *o).collect();

                UserFaceFeature::insert_data(UserFaceFeature {
                    id: chrono::Utc::now().timestamp(),
                    feature: feature.clone(),
                    user_id: chrono::Utc::now().timestamp(),
                });

                let sea_p = SearchOptions::default().metric_type(MetricType::IP).output_fields(vec![String::from("user_id"), String::from("id")]);


                let s = client.search(UserFaceFeature::schema()?.name(), vec![Value::FloatArray(Cow::from(feature))], "feature", &sea_p).await?;

                for ss in s {
                    println!(
                        "result num: {:?}----{:?}",
                        ss.field, ss.score,
                    );

                    println!(
                        "result ss.id len: {:?}-",
                        ss.id.len(),
                    );
                    for s_id in ss.id {
                        match s_id {
                            Value::None => { println!("---None"); }
                            Value::Bool(one) => { println!("---{:?}", one); }
                            Value::Int8(one) => { println!("---{:?}", one); }
                            Value::Int16(one) => { println!("---{:?}", one); }
                            Value::Int32(one) => { println!("---{:?}", one); }
                            Value::Long(one) => { println!("---{:?}", one); }
                            Value::Float(one) => { println!("---{:?}", one); }
                            Value::Double(one) => { println!("---{:?}", one); }
                            Value::FloatArray(one) => { println!("---{:?}", one); }
                            Value::Binary(one) => { println!("---{:?}", one); }
                            Value::String(one) => { println!("---{:?}", one); }
                            Value::Json(one) => { println!("---{:?}", one); }
                            Value::Array(one) => { println!("---{:?}", one); }
                        }
                    }
                }
            }
        }


        Ok(vec![])
    }
}


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


            let bb = tt.bytes().await?;
            let bytes = bb.to_vec();

            let img = image::load_from_memory(&bytes)?;
            let start = Instant::now();
            let face_boxes = ai.get_face_boxes(&img)?;
            let duration = start.elapsed();
            println!("Time elapsed in ai.get_face_boxes(&img)?;            is: {:?} milliseconds", duration.as_millis());
            let mut face_img = ai.get_face_img(&img, &face_boxes);

            if face_img.len() > 1 {
                let t1 = face_img.pop().unwrap();
                let t2 = face_img.pop().unwrap();


                let start = Instant::now();
                let sim = ai.get_sim(&t1, &t2)?;
                let duration = start.elapsed();
                println!("Time elapsed in ai.get_face_boxes(&img)?;    get_sim        is: {:?} milliseconds", duration.as_millis());

                println!("------{sim}")
            }
        }
        Err(_) => {}
    }


    Ok(String::from("123"))
}