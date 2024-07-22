use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::sync::{Arc, Mutex, Once};
use ort::{CUDAExecutionProvider, inputs, Session, SessionOutputs};
use lazy_static::lazy_static;
use anyhow::Result;
use image::{DynamicImage, RgbImage};
use ndarray::{Array, Array1, Axis, Dim, IxDynImpl};
use raqote::{DrawOptions, DrawTarget, LineJoin, PathBuilder, SolidSource, Source, StrokeStyle};
use show_image::{event, ImageInfo, ImageView, PixelFormat, WindowOptions};

mod nms;

const DET_10G_URL: &str = "D:\\temp\\aaa\\buffalo_l\\det_10g.onnx";
const DET_R50G_URL: &str = "D:\\temp\\aaa\\buffalo_l\\w600k_r50.onnx";

pub fn init() -> Result<()> {
    ort::init()
        .with_execution_providers([CUDAExecutionProvider::default().build()])
        .commit()?;

    Ok(())
}

struct ModInfo {
    pub stride: usize,
    pub scores: String,
    pub boxes: String,
}

impl ModInfo {
    pub fn new(stride: usize, scores: &str, boxes: &str) -> Self {
        Self {
            stride,
            scores: scores.to_string(),
            boxes: boxes.to_string(),
        }
    }

    pub fn get_insight() -> Vec<Self> {
        vec![
            // Self::new(8, "448", "451"),
            Self::new(16, "471", "474"),
            Self::new(16, "494", "497"),
        ]
    }
}

pub struct AiSession {
    det_mod: Session,
    feature_mod: Session,
}

fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let dot_product = a.dot(b);
    let norm_a = a.dot(a);
    let norm_b = b.dot(b);
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0 // If either vector is zero, the cosine similarity is 0
    } else {
        (dot_product * dot_product / (norm_a * norm_b)).sqrt()
    }
}

fn convert_to_array1(arr: &Array<f32, Dim<IxDynImpl>>) -> Result<Array1<f32>> {
    use ndarray::IxDyn;
    // 创建一个动态维度的数组，但这里是一维的
    // let arr_dyn = ArrayD::<f32>::from_shape_vec(Dim(IxDynImpl), vec![1.0, 2.0, 3.0]).unwrap();

    //println!("arr----{:?}", arr.shape());
    let t = arr.len();
    // 将动态维度数组转换为 Array1
    let arr1: Array1<f32> = arr.clone().into_shape(Dim([t as _]))?.into();
    Ok(arr1)
}

impl AiSession {
    pub fn get(&self) -> Result<()> {
        Ok(())
    }

    pub fn get_face_img(param_img: &DynamicImage, boxes: &[nms::BBox]) -> Vec<DynamicImage> {
        let mut faces = vec![];

        for bbox in boxes {
            let t = param_img.crop_imm(bbox.x1 as _,
                                       bbox.y1 as _,
                                       (bbox.x2 - bbox.x1) as _,
                                       (bbox.y2 - bbox.y1) as _);
            faces.push(t);
        }
        faces
    }

    pub fn get_sim(&self, param_img1: &DynamicImage, param_img2: &DynamicImage) -> Result<f32> {
        let im1 = self.get_face_feature(param_img1)?;
        let im2 = self.get_face_feature(param_img2)?;

        let r = cosine_similarity(&(convert_to_array1(&im1)?), &convert_to_array1(&im2)?);

        Ok(r)
    }

    pub fn get_face_feature(&self, param_img: &DynamicImage) -> Result<Array<f32, Dim<IxDynImpl>>> {
        let (img_width, img_height) = (param_img.width(), param_img.height());


        // 输入尺寸
        let input_width = 112;
        let input_height = 112;

        let input_mean = 127.5;
        let input_std = 128.0;
        // 计算原图宽高比和模型宽高比
        let im_ratio = img_height as f32 / img_width as f32;
        let model_ratio = input_height as f32 / input_width as f32;


        // 计算新的宽度和高度
        let mut new_width = input_width;
        let mut new_height = (new_width as f32 * im_ratio) as u32;

        if im_ratio > model_ratio {
            new_height = input_height;
            new_width = (new_height as f32 / im_ratio) as u32;
        }
        let original_img = param_img.resize_exact(new_width, new_height, image::imageops::FilterType::Lanczos3);


        // 创建一个640x640的黑色图像
        let mut new_image = RgbImage::new(input_width, input_height);

        // 将缩放后的图像复制到新图像的左上角
        let roi = image::imageops::overlay(&mut new_image, &original_img.to_rgb8(), 0, 0);

        // 将图像数据转换为ndarray
        let mut input = Array::zeros((1, 3, 112, 112));

        for (x, y, pixel) in new_image.enumerate_pixels() {
            let r = (pixel[0] as f32 - input_mean) / input_std;
            let g = (pixel[1] as f32 - input_mean) / input_std;
            let b = (pixel[2] as f32 - input_mean) / input_std;

            input[[0, 0, y as usize, x as usize]] = r;
            input[[0, 1, y as usize, x as usize]] = g;
            input[[0, 2, y as usize, x as usize]] = b;
        }

        let outputs: SessionOutputs = self.feature_mod.run(inputs!["input.1" => input.view()]?)?;
        let scores_471 = outputs["683"].try_extract_tensor::<f32>()?.to_owned();
        Ok(scores_471)
    }


    pub fn get_face_boxes(&self, param_img: &DynamicImage) -> Result<Vec<nms::BBox>> {
        // let param_img= param_img.resize_exact(1280, 886, image::imageops::FilterType::Lanczos3);
        let (img_width, img_height) = (param_img.width(), param_img.height());


        // 输入尺寸
        let input_width = 640;
        let input_height = 640;

        let input_mean = 127.5;
        let input_std = 128.0;
        // 计算原图宽高比和模型宽高比
        let im_ratio = img_height as f32 / img_width as f32;
        let model_ratio = input_height as f32 / input_width as f32;


        // 计算新的宽度和高度
        let mut new_width = input_width;
        let mut new_height = (new_width as f32 * im_ratio) as u32;

        if im_ratio > model_ratio {
            new_height = input_height;
            new_width = (new_height as f32 / im_ratio) as u32;
        }
        let original_img = param_img.resize_exact(new_width, new_height, image::imageops::FilterType::Lanczos3);


        // 创建一个640x640的黑色图像
        let mut new_image = RgbImage::new(input_width, input_height);

        // 将缩放后的图像复制到新图像的左上角
        let roi = image::imageops::overlay(&mut new_image, &original_img.to_rgb8(), 0, 0);

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


        let outputs = self.det_mod.run(inputs!["input.1" => input.view()]?)?;

        let mut boxes_result = Vec::new();

        for modInfo in ModInfo::get_insight() {


            // ********************---------------*********************************************************************Start End 中
            {
                let stride = modInfo.stride;
                let scores_471 = outputs["471"].try_extract_tensor::<f32>()?.to_owned();
                let bbox_preds_474 = outputs["474"].try_extract_tensor::<f32>()?.to_owned();
                let bbox_preds_474 = bbox_preds_474 * (stride as f32);
                let size: usize = 640 / stride;

                let mut input_boxes = Array::zeros(((size * size) as _, 2));


                for i in 0..size {
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
                    //println!("{}   ----- {:?}", index_i, axis_one);
                    index_i = index_i + 1;
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


                // //println!("bbox_preds_474_temp_box  shape {:?}", bbox_preds_474_temp_box.shape()); //  [3200, 4]
                // for axis_one in bbox_preds_474_temp_box.axis_iter(Axis(0)) {
                //     //println!("bbox_preds_474_temp_box -- {:?}", axis_one);
                // }
                // // OK

                let pos_index: Vec<usize> = scores_471
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &score)| if score >= 0.5 { Some(i) } else { None })
                    .collect::<Vec<_>>();

                let pos_scores = scores_471.select(Axis(0), &pos_index[..]);
                let pos_bboxes = bbox_preds_474_temp_box.select(Axis(0), &pos_index[..]);
                let pos_bboxes = pos_bboxes * 2.0;
                //println!("pos_scores  shape {:?}", pos_scores.shape()); //  [3200, 4]
                //println!("pos_bboxes  shape {:?}", pos_bboxes.shape()); //  [3200, 4]

                for axis_one in pos_scores.axis_iter(Axis(0)) {
                    //println!("pos_scores -- {:?}", axis_one);
                }

                for axis_one in pos_bboxes.axis_iter(Axis(0)) {
                    //println!("pos_bboxes -- {:?}", axis_one);
                }


                pos_scores.axis_iter(Axis(0)).zip(pos_bboxes.axis_iter(Axis(0))).for_each(|(so, bo)| {
                    //println!("bo----{:?}--so----{:?}", bo, so);
                    boxes_result.push(
                        nms::BBox {
                            x1: bo[0],
                            y1: bo[1],
                            x2: bo[2],
                            y2: bo[3],
                            score: so[0],
                        }
                    )
                });
            }


            // ********************---------------*********************************************************************End 中
        }

        let mut boxes = boxes_result.clone();

        boxes.sort_by(|box1, box2| box2.score.total_cmp(&box1.score));
        // -------------


        let mut dt = DrawTarget::new(img_width as _, img_height as _);

        let mut faces = vec![];
        let t_r = nms::nms(boxes_result, 0.5);

        for bbox in &t_r {
            let mut pb = PathBuilder::new();
            //          let x1 = ((bbox.x1 / 640.0) * img_width as f32) as f32;
            //         let y1 = ((bbox.y1 / 640.0) * img_height as f32) as f32;
            //         let x2 = ((bbox.x2 / 640.0) * img_width as f32) as f32;
            //         let y2 = ((bbox.y2 / 640.0) * img_height as f32) as f32;
            //         pb.rect(x1, y1, x2 - x1, y2 - y1);

            // pb.rect(bbox.x1, bbox.y1, bbox.x2 - bbox.x1, bbox.y2 - bbox.y1);
            // println!("result--22---({},{})----({},{})", bbox.x1, bbox.y1,bbox.x2, bbox.y2,);
            // result-BOX---(151.95132,225.53683)----(339.87537,479.1383)
            // result-BOX---(774.05804,149.51135)----(1079.1631,547.2977)
            // result--22---(-104.04868,-126.463165)----(83.875374,127.1383)
            // result--22---(-153.94197,-202.48865)----(151.16313,195.2977)



            // let (img_width, img_height) = (640, 640);
            //
            let x1 = ((bbox.x1 / 640.0) * img_width as f32) as f32;
            let y1 = ((bbox.y1 / 640.0) * img_height as f32) as f32;
            let x2 = ((bbox.x2 / 640.0) * img_width as f32) as f32;
            let y2 = ((bbox.y2 / 640.0) * img_height as f32) as f32;
            pb.rect(x1, y1, x2 - x1, y2 - y1);

            println!("result-BOX---({},{})----({},{})", x1, y1,x2, y2);
            // result-BOX---(302.95294,255.84335)----(677.6265,543.5225)
            // result-BOX---(1543.2783,169.60194)----(2151.5813,620.8409)

            // result-BOX---(151.95132,128.27408)----(339.87537,272.50992)
            // result-BOX---(774.0581,85.034584)----(1079.1631,311.2756)

            let t = original_img.crop_imm(bbox.x1 as _, bbox.y1 as _, (bbox.x2 - bbox.x1) as _, (bbox.y2 - bbox.y1) as _);
            faces.push(t);

            let path = pb.finish();
            let color =  SolidSource { r: 0x00, g: 0x10, b: 0x80, a: 0x80 };
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

        let window = show_image::context()
            .run_function_wait(move |context| -> std::result::Result<_, String> {
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

                let tt = original_img.to_rgb8();
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

        Ok(t_r)
    }
}

impl AiSession {
    fn new() -> Result<Arc<AiSession>> {
        let det = Session::builder()?.commit_from_file(DET_10G_URL)?;
        let feature = Session::builder()?.commit_from_file(DET_R50G_URL)?;
        Ok(Arc::new(AiSession {
            det_mod: det,
            feature_mod: feature,
        }))
    }

    pub fn get_instance() -> Result<Arc<AiSession>> {
        lazy_static! {
             static  ref INSTANCE_AI_SESSION:Arc<AiSession> = AiSession::new().unwrap();
        }
        Ok(INSTANCE_AI_SESSION.clone())
    }
}