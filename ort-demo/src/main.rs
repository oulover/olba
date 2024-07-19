#![allow(clippy::manual_retain)]

pub mod nms;
pub mod face_det;

// use crate::nms;
use std::iter::Zip;
use std::path::Path;

use image::{GenericImageView, imageops::FilterType, RgbImage};
use ndarray::{Array, ArrayBase, Axis, s};
use ort::{CUDAExecutionProvider, inputs, Session, SessionOutputs};
use raqote::{DrawOptions, DrawTarget, LineJoin, PathBuilder, SolidSource, Source, StrokeStyle};
use show_image::{event, ImageInfo, ImageView, PixelFormat, WindowOptions};
use tracing_subscriber::filter::FilterExt;

#[derive(Debug, Clone, Copy)]
struct BoundingBox {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
}

const DET_10G_URL: &str = "D:\\temp\\aaa\\buffalo_l\\det_10g.onnx";

#[show_image::main]
fn main22() -> ort::Result<()> {
    tracing_subscriber::fmt::init();

    ort::init()
        .with_execution_providers([CUDAExecutionProvider::default().build()])
        .commit()?;

    // 打开图片文件
    let f = "D:\\Temp\\aaa\\w1.jpg";
    let original_img = image::open(Path::new(f)).unwrap();
    let original_img= original_img.resize_exact(1280, 886, image::imageops::FilterType::Lanczos3);
    let (img_width, img_height) = (original_img.width(), original_img.height());

    // 输入尺寸
    let input_size = (640, 640);
    let input_mean = 127.5;
    let input_std = 128.0;

    // 计算原图宽高比和模型宽高比
    let im_ratio = img_height as f32 / img_width as f32;
    let model_ratio = input_size.1 as f32 / input_size.0 as f32;

    // 计算新的宽度和高度
    let mut new_width = input_size.0;
    let mut new_height = (new_width as f32 * im_ratio) as u32;

    if im_ratio > model_ratio {
        new_height = input_size.1;
        new_width = (new_height as f32 / im_ratio) as u32;
    }

    // 缩放图片
    let resized_img = original_img.resize_exact(new_width, new_height, image::imageops::FilterType::Lanczos3);

    // 创建一个640x640的黑色图像
    let mut new_image = RgbImage::new(input_size.0, input_size.1);

    // 将缩放后的图像复制到新图像的左上角
    let roi = image::imageops::overlay(&mut new_image, &resized_img.to_rgb8(), 0, 0);

    // 将图像数据转换为ndarray
    let mut input = Array::zeros((1, 3, 640, 640));

    for (x, y, pixel) in new_image.enumerate_pixels() {



        let r = (pixel[0] as f32 - input_mean) / input_std;
        let g = (pixel[1] as f32 - input_mean) / input_std;
        let b = (pixel[2] as f32 - input_mean) / input_std;

        input[[0, 0, y as usize, x as usize]] = r;
        input[[0, 1, y as usize, x as usize]] = g;
        input[[0, 2, y as usize, x as usize]] = b;
    }

    // 现在 `input` 包含处理后的图像数据
    //  println!("{:?}", input);

    // println!(" rgb_img {}",input);


    let model = Session::builder()?.commit_from_file(DET_10G_URL)?;

    // 448 张量（分类分数）和 451 张量（边界框坐标）
    // （448，451，454）：这是步长为8的一组数据，
    // 448（12800x1=>1x80x80x2 ）,步长为8，高宽640，640/8=80，每行80个预测框，共80行。通过insightface的源码可以看到，num_anchors = 2，
    // 每个位置的目标框是两组,正常来说是黑白图两种，既然是同一个位置，那么可以合并一起，意思就是有2张图 ，每张图大小是80x80，有这么多分值。
    //
    // 451：bboxs: 1x8x80x80 每一个分数对应的四个点(x1,y1,x2,y2)*注意这个点是距离原点的相对值，
    // 还是需要计算的,这里1x8  前面1~4 是一个矩形框的点，后面的4~8是另一张图的矩形框坐标点，就是黑白图。
    //
    // 454：kps：1x20x80x80 每一个分数对应的五官坐标点（x,y）*注意这个点是距离原点的相对值，
    // 还是需要计算的，这里1~10 是一组坐标点，另外的10~20是另外一张图的一组坐标点，分开计算就行。

    let outputs: SessionOutputs = model.run(inputs!["input.1" => input.view()]?)?;

    // // 448（12800x1=>1x80x80x2 ）,步长为8，高宽640，640/8=80，每行80个预测框，共80行。通过insightface的源码可以看到，num_anchors = 2，
    // // 每个位置的目标框是两组,正常来说是黑白图两种，既然是同一个位置，那么可以合并一起，意思就是有2张图 ，每张图大小是80x80，有这么多分值。
    // let output_448 = outputs["448"].try_extract_tensor::<f32>()?.to_owned();
    //
    // // 451：bboxs: 1x8x80x80 每一个分数对应的四个点(x1,y1,x2,y2)*注意这个点是距离原点的相对值，
    // // 还是需要计算的,这里1x8  前面1~4 是一个矩形框的点，后面的4~8是另一张图的矩形框坐标点，就是黑白图。
    // let output_451 = outputs["451"].try_extract_tensor::<f32>()?.t().into_owned();
    // let output_451 = output_451.into_shape([12800, 4]).unwrap();
    // let output_451 = output_451 * 8.0;
    // let axis_451_0 = output_451.axis_iter(Axis(0));
    // println!("output_451 --  {:?}", output_451.shape());
    // // for axis_one in axis_451_0 {
    // //     println!("shape 448 -- {:?}", axis_one);
    // // }
    //
    // let mut input_boxes = Array::zeros((12800, 2));
    // for i in 0..(12800 / 2) {
    //     let index = i * 2;
    //     input_boxes[[index, 0]] = ((i * 8) % 640) as f32;
    //     input_boxes[[index + 1, 0]] = ((i * 8) % 640) as f32;
    //     // input_boxes[[index+1,1]] = 0.0;
    // }
    // // for axis_one in input_boxes .axis_iter(Axis(0)){
    // //      println!("input_boxes 448 -- {:?}", axis_one);
    // //  }
    // let points = input_boxes;
    // let distance = output_451.clone();
    // let temp_box = {
    //     // 确保输入点集和距离集的形状匹配
    //     // assert_eq!(points.shape(), distance.shape());
    //
    //     let n = points.len_of(Axis(0));
    //
    //     // 初始化输出数组
    //     let mut bboxes = Array::zeros((n, 4));
    //
    //     for i in 0..n {
    //         let x = points[[i, 0]];
    //         let y = points[[i, 1]];
    //
    //         let dx1 = distance[[i, 0]];
    //         let dy1 = distance[[i, 1]];
    //         let dx2 = distance[[i, 2]];
    //         let dy2 = distance[[i, 3]];
    //
    //         // 计算边界框的坐标
    //         bboxes[[i, 0]] = x - dx1; // x1
    //         bboxes[[i, 1]] = y - dy1; // y1
    //         bboxes[[i, 2]] = x + dx2; // x2
    //         bboxes[[i, 3]] = y + dy2; // y2
    //     }
    //
    //     bboxes
    // };
    // println!("{:?}", temp_box.shape()); // 正确了 bboxes = distance2bbox(anchor_centers, bbox_preds)
    // // for axis_one in temp_box .axis_iter(Axis(0)){
    // //      println!("temp_box 448 -- {:?}", axis_one);
    // //  }
    //
    // // 结果
    // // let mut scores_list = Vec::new();
    // // 16
    // // 471，474，477
    // // 471  float32[3200,1]  scores
    // // 474  float32[3200,4]  bbox_preds
    // // 477  float32[3200,10]
    let mut boxes_result = Vec::new();
    // ********************---------------*********************************************************************Start End 中
    {

        let stride = 16;
        let scores_471 = outputs["471"].try_extract_tensor::<f32>()?.to_owned();
        let bbox_preds_474 = outputs["474"].try_extract_tensor::<f32>()?.to_owned();
        let bbox_preds_474 = bbox_preds_474 * 16.0;


        let mut input_boxes = Array::zeros((3200, 2));
        let y = 0;

        for i in 0..(3200 / 2) {
            let index = i * 2;
            input_boxes[[index, 0]] = ((i * stride) % 640) as f32;
            input_boxes[[index + 1, 0]] = ((i * stride) % 640) as f32;
            let a = ((i * stride) / 640) as f32;
            input_boxes[[index, 1]] = a * stride as f32;
            input_boxes[[index + 1, 1]] = a * stride as f32;
        }
        let temp = input_boxes.axis_iter(Axis(0));

        let mut index_i = 0;
        for axis_one in temp {
            println!("{}   ----- {:?}",index_i, axis_one);
            index_i = index_i+1;
        }

        let points = input_boxes;
        let distance = bbox_preds_474.clone();
        let bbox_preds_474_temp_box = {
            // 确保输入点集和距离集的形状匹配
            // assert_eq!(points.shape(), distance.shape());

            let n = points.len_of(Axis(0));

            // 初始化输出数组
            let mut bboxes = Array::zeros((n, 4));

            for i in 0..n {
                let x = points[[i, 0]];
                let y = points[[i, 1]];

                let dx1 = distance[[i, 0]];
                let dy1 = distance[[i, 1]];
                let dx2 = distance[[i, 2]];
                let dy2 = distance[[i, 3]];

                // 计算边界框的坐标
                bboxes[[i, 0]] = x - dx1; // x1
                bboxes[[i, 1]] = y - dy1; // y1
                bboxes[[i, 2]] = x + dx2; // x2
                bboxes[[i, 3]] = y + dy2; // y2
            }

            bboxes
        };


        println!("bbox_preds_474_temp_box  shape {:?}", bbox_preds_474_temp_box.shape()); //  [3200, 4]
        for axis_one in bbox_preds_474_temp_box.axis_iter(Axis(0)) {
            println!("bbox_preds_474_temp_box -- {:?}", axis_one);
        }
        // OK

        let pos_index: Vec<usize> = scores_471
            .iter()
            .enumerate()
            .filter_map(|(i, &score)| if score >= 0.5 { Some(i) } else { None })
            .collect::<Vec<_>>();

        let pos_scores = scores_471.select(Axis(0), &pos_index[..]);
        let pos_bboxes = bbox_preds_474_temp_box.select(Axis(0), &pos_index[..]);
        let pos_bboxes = pos_bboxes * 2.0;
        println!("pos_scores  shape {:?}", pos_scores.shape()); //  [3200, 4]
        println!("pos_bboxes  shape {:?}", pos_bboxes.shape()); //  [3200, 4]

        for axis_one in pos_scores.axis_iter(Axis(0)) {
            println!("pos_scores -- {:?}", axis_one);
        }

        for axis_one in pos_bboxes.axis_iter(Axis(0)) {
            println!("pos_bboxes -- {:?}", axis_one);
        }






        pos_scores.axis_iter(Axis(0)).zip(pos_bboxes.axis_iter(Axis(0))).for_each(|(so, bo)| {
            println!("bo----{:?}--so----{:?}",bo,so);
            boxes_result.push((
                BoundingBox {
                    x1: bo[0],
                    y1: bo[1],
                    x2: bo[2],
                    y2: bo[3],
                }, "tlab", so[0]
            ))
        });
    }


    // ********************---------------*********************************************************************End 中


    // ********************---------------*********************************************************************Start   大
    {

        let stride = 32;
        let scores_471 = outputs["494"].try_extract_tensor::<f32>()?.to_owned();
        let bbox_preds_474 = outputs["497"].try_extract_tensor::<f32>()?.to_owned();
        let bbox_preds_474 = bbox_preds_474 * 32.;


        let mut input_boxes = Array::zeros((800, 2));
        let y = 0;

        for i in 0..(400) {
            let index = i * 2;
            input_boxes[[index, 0]] = ((i * stride) % 640) as f32;
            input_boxes[[index + 1, 0]] = ((i * stride) % 640) as f32;
            let a = ((i * stride) / 640) as f32;
            input_boxes[[index, 1]] = a * stride as f32;
            input_boxes[[index + 1, 1]] = a * stride as f32;
        }
        let temp = input_boxes.axis_iter(Axis(0));

        let mut index_i = 0;
        for axis_one in temp {
            println!("{}   ----- {:?}",index_i, axis_one);
            index_i = index_i+1;
        }

        let points = input_boxes;
        let distance = bbox_preds_474.clone();
        let bbox_preds_474_temp_box = {
            // 确保输入点集和距离集的形状匹配
            // assert_eq!(points.shape(), distance.shape());

            let n = points.len_of(Axis(0));

            // 初始化输出数组
            let mut bboxes = Array::zeros((n, 4));

            for i in 0..n {
                let x = points[[i, 0]];
                let y = points[[i, 1]];

                let dx1 = distance[[i, 0]];
                let dy1 = distance[[i, 1]];
                let dx2 = distance[[i, 2]];
                let dy2 = distance[[i, 3]];

                // 计算边界框的坐标
                bboxes[[i, 0]] = x - dx1; // x1
                bboxes[[i, 1]] = y - dy1; // y1
                bboxes[[i, 2]] = x + dx2; // x2
                bboxes[[i, 3]] = y + dy2; // y2
            }

            bboxes
        };


        println!("bbox_preds_474_temp_box  shape {:?}", bbox_preds_474_temp_box.shape()); //  [3200, 4]
        for axis_one in bbox_preds_474_temp_box.axis_iter(Axis(0)) {
            println!("bbox_preds_474_temp_box -- {:?}", axis_one);
        }
        // OK

        let pos_index: Vec<usize> = scores_471
            .iter()
            .enumerate()
            .filter_map(|(i, &score)| if score >= 0.5 { Some(i) } else { None })
            .collect::<Vec<_>>();

        let pos_scores = scores_471.select(Axis(0), &pos_index[..]);
        let pos_bboxes = bbox_preds_474_temp_box.select(Axis(0), &pos_index[..]);
        let pos_bboxes = pos_bboxes * 2.0;
        println!("pos_scores  shape {:?}", pos_scores.shape()); //  [3200, 4]
        println!("pos_bboxes  shape {:?}", pos_bboxes.shape()); //  [3200, 4]

        for axis_one in pos_scores.axis_iter(Axis(0)) {
            println!("pos_scores -- {:?}", axis_one);
        }

        for axis_one in pos_bboxes.axis_iter(Axis(0)) {
            println!("pos_bboxes -- {:?}", axis_one);
        }






        pos_scores.axis_iter(Axis(0)).zip(pos_bboxes.axis_iter(Axis(0))).for_each(|(so, bo)| {
            println!("bo----{:?}--so----{:?}",bo,so);
            boxes_result.push((
                BoundingBox {
                    x1: bo[0],
                    y1: bo[1],
                    x2: bo[2],
                    y2: bo[3],
                }, "tlab", so[0]
            ))
        });
    }


    // ********************---------------*********************************************************************End 中













    for o in &boxes_result{
        println!("pos_bboxes----{:?}---{}",o.0,o.2);
    }




    // --------------------------------------------------------------------------------------------------------------------
    let mut boxes = boxes_result.clone();

    boxes.sort_by(|box1, box2| box2.2.total_cmp(&box1.2));
    // let mut result = Vec::new();

    // while !boxes.is_empty() {
    //     if let Some(p) = boxes.pop(){
    //         result.push(p);
    //     }
    //
    //     boxes = boxes
    //         .iter()
    //         .skip(1)
    //         .filter(|box1| {
    //             let x = intersection(&boxes[0].0, &box1.0) / union(&boxes[0].0, &box1.0);
    //             println!("skip -- {}",x);
    //             x < 0.7
    //         })
    //         .copied()
    //         .collect();
    //     boxes.sort_by(|box1, box2| box2.2.total_cmp(&box1.2));
    // }

    let t_r =  nms::nms(boxes_result.clone().into_iter().map(|o|nms::BBox::new(
        o.0.x1,o.0.y1,o.0.x2,o.0.y2,o.2,1usize
    )).collect(), 0.5);

    let  mut result:Vec<_> = t_r.into_iter().map(|o|{
        (
            BoundingBox {
                x1: o.x1,
                y1: o.y1,
                x2: o.x2,
                y2: o.y2,
            }, "tlab", o.score
        )
    }).collect();

    println!( "result size is {}",result.len());
    for o in &result{
        println!("result----{:?}---{}",o.0,o.2);
    }





    let mut dt = DrawTarget::new(img_width as _, img_height as _);

    let mut faces = vec![];

    for (bbox, label, _confidence) in result {
        let mut pb = PathBuilder::new();
        //          let x1 = ((bbox.x1 / 640.0) * img_width as f32) as f32;
        //         let y1 = ((bbox.y1 / 640.0) * img_height as f32) as f32;
        //         let x2 = ((bbox.x2 / 640.0) * img_width as f32) as f32;
        //         let y2 = ((bbox.y2 / 640.0) * img_height as f32) as f32;
        //         pb.rect(x1, y1, x2 - x1, y2 - y1);
        pb.rect(bbox.x1, bbox.y1, bbox.x2 - bbox.x1, bbox.y2 - bbox.y1);

       let t =  original_img.crop_imm(bbox.x1 as _, bbox.y1 as _, (bbox.x2 - bbox.x1) as _, (bbox.y2 - bbox.y1) as _);
        faces.push(t);

        let path = pb.finish();
        let color = match label {
            "baseball bat" => SolidSource { r: 0x00, g: 0x10, b: 0x80, a: 0x80 },
            "baseball glove" => SolidSource { r: 0x20, g: 0x80, b: 0x40, a: 0x80 },
            _ => SolidSource { r: 0x80, g: 0x10, b: 0x40, a: 0x80 }
        };
        dt.stroke(
            &path,
            &Source::Solid(color),
            &StrokeStyle {
                join: LineJoin::Round,
                width: 4.,
                ..StrokeStyle::default()
            },
            &DrawOptions::new(),
        );
    }

    let overlay: show_image::Image = dt.into();


    face_det::main222(faces.pop().unwrap_or(original_img.clone()) .clone(),faces.pop().unwrap_or(original_img.clone()) .clone());

    let window = show_image::context()
        .run_function_wait(move |context| -> Result<_, String> {
            let mut window = context
                .create_window(
                    "ort + YOLOv8",
                    WindowOptions {
                        size: Some([img_width, img_height]),
                        ..WindowOptions::default()
                    },
                )
                .map_err(|e| e.to_string())?;
            // window.set_image("baseball", &original_img.as_image_view().map_err(|e| e.to_string())?);

            let tt = faces.pop().unwrap_or(original_img) .to_rgb8();
            let image_view = ImageView::new(ImageInfo::new(PixelFormat::Bgr8, tt.width(), tt.height()), tt.as_raw());


            window.set_image("baseball", &image_view);
            window.set_overlay("yolo", &overlay.as_image_view().map_err(|e| e.to_string())?, true);
            Ok(window.proxy())
        })
        .unwrap();

    for event in window.event_channel().unwrap() {
        if let event::WindowEvent::KeyboardInput(event) = event {
            if event.input.key_code == Some(event::VirtualKeyCode::Escape) && event.input.state.is_pressed() {
                break;
            }
        }
    }


    Ok(())
}

fn intersection(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
    (box1.x2.min(box2.x2) - box1.x1.max(box2.x1)) * (box1.y2.min(box2.y2) - box1.y1.max(box2.y1))
}

fn union(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
    ((box1.x2 - box1.x1) * (box1.y2 - box1.y1)) + ((box2.x2 - box2.x1) * (box2.y2 - box2.y1)) - intersection(box1, box2)
}