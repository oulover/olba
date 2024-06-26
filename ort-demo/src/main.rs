#![allow(clippy::manual_retain)]

use std::path::Path;

use image::{GenericImageView, imageops::FilterType, RgbImage};
use ndarray::{Array, Axis, s};
use ort::{CUDAExecutionProvider, inputs, Session, SessionOutputs};
use raqote::{DrawOptions, DrawTarget, LineJoin, PathBuilder, SolidSource, Source, StrokeStyle};
use show_image::{event, ImageInfo, ImageView, PixelFormat, WindowOptions};

#[derive(Debug, Clone, Copy)]
struct BoundingBox {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
}

const DET_10G_URL: &str = "D:\\temp\\aaa\\buffalo_l\\det_10g.onnx";

#[show_image::main]
fn main() -> ort::Result<()> {
    tracing_subscriber::fmt::init();

    ort::init()
        .with_execution_providers([CUDAExecutionProvider::default().build()])
        .commit()?;

    // 打开图片文件
    let f = "D:\\Temp\\aaa\\t1.jpg";
    let original_img = image::open(Path::new(f)).unwrap();
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
    println!("{:?}", input);

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

    // 448（12800x1=>1x80x80x2 ）,步长为8，高宽640，640/8=80，每行80个预测框，共80行。通过insightface的源码可以看到，num_anchors = 2，
    // 每个位置的目标框是两组,正常来说是黑白图两种，既然是同一个位置，那么可以合并一起，意思就是有2张图 ，每张图大小是80x80，有这么多分值。
    let output_448 = outputs["448"].try_extract_tensor::<f32>()?.to_owned();

    // 451：bboxs: 1x8x80x80 每一个分数对应的四个点(x1,y1,x2,y2)*注意这个点是距离原点的相对值，
    // 还是需要计算的,这里1x8  前面1~4 是一个矩形框的点，后面的4~8是另一张图的矩形框坐标点，就是黑白图。
    let output_451 = outputs["451"].try_extract_tensor::<f32>()?.t().into_owned();

    // ------------------------------------------------------------------------------------------------------------------------
    println!("shape 448 -- {:?}", output_448.shape()); // shape 448 -- [12800, 1]
    let axis_448_0 = output_448.axis_iter(Axis(1));

    for axis_one in axis_448_0 {
        println!("shape 448 -- {:?}", axis_one);
    }


    // --------------------------------------------------------------------------------------------------------------------

    let mut boxes = Vec::new();
    // let output_451 = output_451.slice(s![.., ..,]);


    // let output_451 = output_451.slice(s![0, ..]);  // 选取第一个批次的元素

    // 448（12800x1=>1x80x80x2 ）
    // output_448  shape[12800, 1]
    println!("output_448  shape{:?}", output_448.shape());
    let output_448_reshaped = output_448.into_shape([80, 80, 2]).unwrap(); // 重塑为 (80, 80, 2, 4)
    println!("output_448_reshaped  shape{:?}", output_448_reshaped.shape());

    // 451：  1x8x80x80 每一个分数对应的四个点(x1,y1,x2,y2)*注意这个点是距离原点的相对值，
    // output_451  shape[4, 12800]  == 51,200  ==  1x8x80x80
    println!("output_451  shape{:?}", output_451.shape());
    let output_451_reshaped = output_451.clone().into_shape([8, 80, 80]).unwrap();
    println!("output_451_reshaped  shape{:?}", output_451_reshaped.shape());


    let x_all = output_448_reshaped.axis_iter(Axis(0));
    for row in x_all {
        let row: Vec<_> = row.iter().copied().collect();
        let x1 = row[0] / 640. * (img_width as f32);
        let y1 = row[1] / 640. * (img_height as f32);
        let x2 = row[2] / 640. * (img_width as f32);
        let y2 = row[3] / 640. * (img_height as f32);
        boxes.push(BoundingBox { x1, y1, x2, y2 });
    }

    let mut dt = DrawTarget::new(img_width as _, img_height as _);

    for bbox in boxes {
        let mut pb = PathBuilder::new();
        pb.rect(bbox.x1, bbox.y1, bbox.x2 - bbox.x1, bbox.y2 - bbox.y1);
        let path = pb.finish();
        let color = SolidSource { r: 0x00, g: 0xFF, b: 0x00, a: 0xFF };
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
        .run_function_wait(move |context| -> Result<_, String> {
            let mut window = context
                .create_window(
                    "insightface + det_10g",
                    WindowOptions {
                        size: Some([img_width, img_height]),
                        ..WindowOptions::default()
                    },
                )
                .map_err(|e| e.to_string())?;

            let tt = original_img.to_rgb8();
            let image_view = ImageView::new(ImageInfo::new(PixelFormat::Bgr8, tt.width(), tt.height()), tt.as_raw());

            window.set_image("face", &image_view);
            window.set_overlay("det_10g", &overlay.as_image_view().map_err(|e| e.to_string())?, true);
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