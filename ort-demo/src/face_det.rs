#![allow(clippy::manual_retain)]


use image::{DynamicImage, GenericImageView, RgbImage};
use ndarray::{Array, Array1, Dim, IxDynImpl};
// use ndarray_linalg::Norm;
use ort::{inputs, Session, SessionOutputs};



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

const DET_10G_URL: &str = "D:\\temp\\aaa\\buffalo_l\\w600k_r50.onnx";

// #[show_image::main]
pub fn main222(original_img1: DynamicImage, original_img2: DynamicImage) -> ort::Result<()> {
    // tracing_subscriber::fmt::init();
    //
    // ort::init()
    //     .with_execution_providers([CUDAExecutionProvider::default().build()])
    //     .commit()?;



    let h1 = op_it(original_img1)?;
    convert_to_array1(h1.clone());


    let h2 = op_it(original_img2)?;

    //println!("cosine_similarity  ----  {}", cosine_similarity(&(convert_to_array1(h1.clone()).unwrap()), &convert_to_array1(h2.clone()).unwrap()));

    let r = h1 - h2;
    let r = r.clone() * r;

    let r: f32 = r.iter().map(|o| o.abs()).sum();
    //println!("main222 *---{}", r.sqrt());

    // 1-2 = *---472.96747
    // 1-3 = *---504.29126
    // 2-3 = *---539.7104

    // 2-l1 = *---1024.9376
    // 2-l2 = *---994.59064
    // 1-l2 = *---991.4202
    // s1 -s2 *---285.34122


    // let temp =scores_471.iter();
    //
    // let mut index_i = 0;
    // for axis_one in temp {
    //     //println!("{}   ----- {:?}",index_i, axis_one);
    //     index_i = index_i+1;
    // }
    //
    // let r:f32 = scores_471.iter().map(|o|o.abs()).sum();
    // //println!("*---{}",r);


    Ok(())
}

fn convert_to_array1(arr: Array<f32, Dim<IxDynImpl>>) -> Result<Array1<f32>, &'static str> {
    
    // 创建一个动态维度的数组，但这里是一维的
    // let arr_dyn = ArrayD::<f32>::from_shape_vec(Dim(IxDynImpl), vec![1.0, 2.0, 3.0]).unwrap();


    //println!("arr----{:?}", arr.shape());
    let t = arr.len();
    // 将动态维度数组转换为 Array1
    let arr1: Array1<f32> = arr.into_shape(Dim([t as _])).unwrap().into();
    Ok(arr1)
}


pub fn op_it(original_img: DynamicImage) -> ort::Result<Array<f32, Dim<IxDynImpl>>> {
    // // 打开图片文件
    // // let f = "D:\\Temp\\aaa\\h11_proc.jpg";
    // let original_img = image::open(Path::new(f)).unwrap();
    let original_img = original_img.resize_exact(112, 112, image::imageops::FilterType::Lanczos3);
    let (img_width, img_height) = (original_img.width(), original_img.height());


    // 输入尺寸
    let input_size = (112, 112);
    let input_mean = 127.5;
    let input_std = 128.0;
    // // 缩放图片
    let resized_img = original_img.resize_exact(112, 112, image::imageops::FilterType::Lanczos3);

    // 创建一个640x640的黑色图像
    let mut new_image = RgbImage::new(input_size.0, input_size.1);

    // 将缩放后的图像复制到新图像的左上角
    let roi = image::imageops::overlay(&mut new_image, &resized_img.to_rgb8(), 0, 0);

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

    // 现在 `input` 包含处理后的图像数据
    //  //println!("{:?}", input);

    // //println!(" rgb_img {}",input);


    let model = Session::builder()?.commit_from_file(DET_10G_URL)?;
    for i in model.inputs.iter() {
        //println!("{:?}", i);
    }
    for i in model.outputs.iter() {
        //println!("{:?}", i);
    }
    let outputs: SessionOutputs = model.run(inputs!["input.1" => input.view()]?)?;

    let scores_471 = outputs["683"].try_extract_tensor::<f32>()?.to_owned();

    //let scores_471 = scores_471.clone() * scores_471;

    // let temp =scores_471.iter();
    //
    // let mut index_i = 0;
    // for axis_one in temp {
    //     //println!("{}   ----- {:?}",index_i, axis_one);
    //     index_i = index_i+1;
    // }
    //
    // let r:f32 = scores_471.iter().map(|o|o.abs()).sum();
    // //println!("*---{}",r);


    Ok(scores_471)
}



