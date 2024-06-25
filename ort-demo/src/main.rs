#![allow(clippy::manual_retain)]

use std::path::Path;

use image::{GenericImageView, imageops::FilterType};
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
    let f = "D:\\Temp\\aaa\\hg11.jpg";
    let original_img = image::open(Path::new(f)).unwrap();
    let (img_width, img_height) = (original_img.width(), original_img.height());
    let img = original_img.resize_exact(640, 640, FilterType::CatmullRom);
    let mut input = Array::zeros((1, 3, 640, 640));
    for pixel in img.pixels() {
        let x = pixel.0 as _;
        let y = pixel.1 as _;
        let [r, g, b, _] = pixel.2.0;
        input[[0, 0, y, x]] = (r as f32) / 255.;
        input[[0, 1, y, x]] = (g as f32) / 255.;
        input[[0, 2, y, x]] = (b as f32) / 255.;
    }

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

    // 448（12800x1=>1x80x80x2 ）
    // output_448  shape[12800, 1]
    let output_448 = outputs["448"].try_extract_tensor::<f32>()?.to_owned();
    println!("output_448  shape{:?}",output_448.shape());
    let output_448_reshaped = output_448.into_shape([80,80,2]).unwrap(); // 重塑为 (80, 80, 2, 4)
    println!("output_448_reshaped  shape{:?}",output_448_reshaped.shape());




    // let output_451 = output_451.slice(s![.., ..,]);
    let output_451 = outputs["451"].try_extract_tensor::<f32>()?.t().into_owned();
    println!("output_451  shape{:?}",output_451.shape());
    let output_451_reshaped = output_451.clone().into_shape([8,80,80]).unwrap();
    println!("output_451_reshaped  shape{:?}",output_451_reshaped.shape());




    // 451：  1x8x80x80 每一个分数对应的四个点(x1,y1,x2,y2)*注意这个点是距离原点的相对值，
    // output_451  shape[4, 12800]  == 51,200  ==  1x8x80x80

    let mut boxes = Vec::new();




    let x_all = output_448_reshaped.axis_iter(Axis(0));
    for row in x_all{
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