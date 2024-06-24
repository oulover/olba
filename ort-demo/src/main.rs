use std::path::Path;

use image::{GenericImageView, imageops::FilterType};
use ndarray::{Array, Axis, s};
use ort::{CUDAExecutionProvider, inputs, Session, SessionOutputs};
use raqote::{DrawTarget, PathBuilder, SolidSource, Source, StrokeStyle};
use show_image::{AsImageView, event, ImageInfo, ImageView, PixelFormat, WindowOptions};

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

    let f = "D:\\Temp\\aaa\\baseball.jpg";
    let original_img = image::open(Path::new(f)).unwrap();
    // let original_img = image::open(Path::new(img_path)).unwrap();
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

    // Run inference
    let outputs: SessionOutputs = model.run(inputs!["input.1" => input.view()]?)?;
    let output_493 = outputs["493"].try_extract_tensor::<f32>()?.t().into_owned();
    let output_496 = outputs["496"].try_extract_tensor::<f32>()?.t().into_owned();

    let mut boxes = Vec::new();
    for (bbox, prob) in output_493.axis_iter(Axis(0)).zip(output_496.axis_iter(Axis(0))) {
        let bbox: Vec<_> = bbox.iter().copied().collect();
        let prob: Vec<_> = prob.iter().copied().collect();
        let xc = bbox[0] / 640. * (img_width as f32);
        let yc = bbox[1] / 640. * (img_height as f32);
        let w = bbox[2] / 640. * (img_width as f32);
        let h = bbox[3] / 640. * (img_height as f32);
        boxes.push((
            BoundingBox {
                x1: xc - w / 2.,
                y1: yc - h / 2.,
                x2: xc + w / 2.,
                y2: yc + h / 2.,
            },
            prob[0],
        ));
    }

    boxes.sort_by(|box1, box2| box2.1.total_cmp(&box1.1));
    let mut result = Vec::new();
    while !boxes.is_empty() {
        result.push(boxes[0]);
        boxes = boxes
            .iter()
            .filter(|box1| {
                let iou = (box1.0.x2.min(result[0].0.x2) - box1.0.x1.max(result[0].0.x1))
                    * (box1.0.y2.min(result[0].0.y2) - box1.0.y1.max(result[0].0.y1));
                iou / (box1.0.x2 - box1.0.x1) * (box1.0.y2 - box1.0.y1) < 0.7
            })
            .copied()
            .collect();
    }

    let mut dt = DrawTarget::new(img_width as _, img_height as _);
    for (bbox, _confidence) in result {
        let mut pb = PathBuilder::new();
        pb.rect(bbox.x1, bbox.y1, bbox.x2 - bbox.x1, bbox.y2 - bbox.y1);
        let path = pb.finish();
        dt.stroke(
            &path,
            &Source::Solid(SolidSource { r: 0x80, g: 0x10, b: 0x40, a: 0x80 }),
            &StrokeStyle { width: 4., ..StrokeStyle::default() },
            &raqote::DrawOptions::new(),
        );
    }

    let overlay: show_image::Image = dt.into();

    let window = show_image::context()
        .run_function_wait(move |context| -> Result<_, String> {
            let mut window = context
                .create_window(
                    "ort + InsightFace",
                    WindowOptions {
                        size: Some([img_width, img_height]),
                        ..WindowOptions::default()
                    },
                )
                .map_err(|e| e.to_string())?;
            let tt = original_img.to_rgb8();
            let image_view = ImageView::new(ImageInfo::new(PixelFormat::Bgr8, tt.width(), tt.height()), tt.as_raw());
            window.set_image("original", &image_view);
            window.set_overlay("detections", &overlay.as_image_view().map_err(|e| e.to_string())?, true);
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