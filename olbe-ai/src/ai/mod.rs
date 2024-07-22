use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::sync::{Arc, Mutex, Once};
use ort::{CUDAExecutionProvider, inputs, Session};
use lazy_static::lazy_static;
use anyhow::Result;
use image::{DynamicImage, RgbImage};
use ndarray::Array;

mod nms;

const DET_10G_URL: &str = "D:\\temp\\aaa\\buffalo_l\\det_10g.onnx";
const DET_R50G_URL: &str = "D:\\temp\\aaa\\buffalo_l\\w600k_r50.onnx";

pub fn init() -> Result<()> {
    ort::init()
        .with_execution_providers([CUDAExecutionProvider::default().build()])
        .commit()?;

    Ok(())
}

struct  ModInfo{
    pub stride:i32,
    pub scores:String,
    pub boxes:String,
}
impl ModInfo{
    pub fn new(stride:i32,scores:&str,boxes:&str)->Self{
        Self{
            stride,
            scores: scores.to_string(),
            boxes: boxes.to_string(),
        }
    }

    pub fn get_insight()->Vec<Self>{
        vec![Self::new(8,"448","451"),
             Self::new(16,"471","474"),
             Self::new(16,"494","497")]
    }
}

pub struct AiSession {
    det_mod: Session,
    feature_mod: Session,
}
impl AiSession {
    pub fn get(&self) -> Result<()> {
        Ok(())
    }

    pub fn get_face_boxes(&self,param_img:DynamicImage)->Result<Vec<nms::BBox>>{
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



       let outputs =  self.det_mod.run(inputs!["input.1" => input.view()]?)?;
        // let mut boxes_result = Vec::new();


        for modInfo in ModInfo::get_insight() {
            let stride = modInfo.stride as f32;
            let scores_471 = outputs[modInfo.scores].try_extract_tensor::<f32>()?.to_owned();
            let bbox_preds_474 = outputs[modInfo.boxes].try_extract_tensor::<f32>()?.to_owned();
            let bbox_preds_474 = bbox_preds_474 * stride;

            // let mut input_boxes = Array::zeros((800, 2));
        }

        Ok(vec![])
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