#![allow(clippy::manual_retain)]

// mod build;
// mod yolov8;


use std::path::Path;

use image::{GenericImageView, imageops::FilterType};
use ndarray::{Array, Axis, s};
use ort::{CUDAExecutionProvider,   inputs, Session, SessionOutputs};
use raqote::{DrawOptions, DrawTarget, LineJoin, PathBuilder, SolidSource, Source, StrokeStyle};
use show_image::{AsImageView, event, ImageInfo, ImageView, PixelFormat, WindowOptions};

#[derive(Debug, Clone, Copy)]
struct BoundingBox {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
}

fn intersection(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
    (box1.x2.min(box2.x2) - box1.x1.max(box2.x1)) * (box1.y2.min(box2.y2) - box1.y1.max(box2.y1))
}

fn union(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
    ((box1.x2 - box1.x1) * (box1.y2 - box1.y1)) + ((box2.x2 - box2.x1) * (box2.y2 - box2.y1)) - intersection(box1, box2)
}

// const YOLOV8M_URL: &str = "D:\\temp\\aaa\\yolov8m.onnx";
const YOLOV8M_URL: &str = "D:\\temp\\aaa\\resnet18_imagenet.onnx";
#[rustfmt::skip]
const YOLOV8_CLASS_LABELS: [&str; 80] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
];
use std::error::Error;
use std::time::Instant;

#[show_image::main]
fn main() -> std::result::Result<(),  Box<dyn Error>> {
    tracing_subscriber::fmt::init();

    ort::init()
        .with_execution_providers([CUDAExecutionProvider::default().build()])
        .commit()?;
    let f = "D:\\temp\\aaa\\baseball.jpg";
    let original_img = image::open(Path::new(f)).unwrap();
    let (img_width, img_height) = (original_img.width(), original_img.height());
    let img = original_img.resize_exact(256, 256, FilterType::CatmullRom);
    let mut input = Array::zeros((1, 3, 256, 256));
    for pixel in img.pixels() {
        let x = pixel.0 as _;
        let y = pixel.1 as _;
        let [r, g, b, _] = pixel.2.0;
        input[[0, 0, y, x]] = (r as f32) / 255.;
        input[[0, 1, y, x]] = (g as f32) / 255.;
        input[[0, 2, y, x]] = (b as f32) / 255.;
    }

    let model = Session::builder()?.commit_from_file(YOLOV8M_URL)?;

    // let input_name = model.inputs.first().unwrap().name.clone();

    // Run YOLOv8 inference
    let now = Instant::now();
    let outputs: SessionOutputs = model.run(inputs!["input" => input.view()]?)?;
    let output = outputs["output"].try_extract_tensor::<f32>()?.t().into_owned();

    let elapsed = now.elapsed();
    println!("耗时: {:?}", elapsed.as_secs_f64());

    use std::fs::File;
    use std::io::Read;
    // 打开 CSV 文件
    let mut file = File::open("D:\\temp\\aaa\\imagenet_class_index.csv")?;
    let mut content = String::new();
    file.read_to_string(&mut content)?;

    // 使用 csv 库解析 CSV 内容
    let mut reader = csv::Reader::from_reader(content.as_bytes());
    let mut classes = Vec::new();

    // 遍历 CSV 记录
    for result in reader.records() {
        let record = result?;
        // 提取 class 列的值
        let class_value = record.get(1).unwrap();
        classes.push(class_value.to_string());
    }


    let mut boxes = Vec::new();
    let sss = s![.., .., ];
    // println!("123{}",sss.in_ndim(),);
    // sss.in_ndim();
    let output = output.slice(sss);
    // println!("output---{:?}",output);
    for row in output.axis_iter(Axis(0)) {
        let row: Vec<_> = row.iter().copied().collect();
        let (class_id, prob) = row
            .iter()
            // skip bounding box coordinates
            // .skip(1)
            .enumerate()
            .map(|(index, value)| (index, *value))
            .reduce(|accum, row| if row.1 > accum.1 { row } else { accum })
            .unwrap();
        if prob < 0.5 {
            continue;
        }
        let label = classes.get(class_id).unwrap();
        // let label = YOLOV8_CLASS_LABELS[class_id];
        let xc = row[0] / 256. * (img_width as f32);
        // println!("row---{}",row.len());
        // println!("row[0]---{}",row[0]);
        let yc = row[1] / 256. * (img_height as f32);
        let w = row[2] / 256. * (img_width as f32);
        let h = row[3] / 256. * (img_height as f32);
        boxes.push((
            BoundingBox {
                x1: xc - w / 2.,
                y1: yc - h / 2.,
                x2: xc + w / 2.,
                y2: yc + h / 2.,
            },
            label,
            prob
        ));
    }

    boxes.sort_by(|box1, box2| box2.2.total_cmp(&box1.2));
    let mut result = Vec::new();

    while !boxes.is_empty() {
        result.push(boxes[0]);
        boxes = boxes
            .iter()
            .filter(|box1| intersection(&boxes[0].0, &box1.0) / union(&boxes[0].0, &box1.0) < 0.7)
            .copied()
            .collect();
    }

    let mut dt = DrawTarget::new(img_width as _, img_height as _);

    for (bbox, label, _confidence) in result {
        let mut pb = PathBuilder::new();
        pb.rect(bbox.x1, bbox.y1, bbox.x2 - bbox.x1, bbox.y2 - bbox.y1);
        let path = pb.finish();
        let label = label.as_str();
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

            // let image_view = original_img.view(0, 0, original_img.width(), original_img.height());
            let tt = original_img.to_rgb8();
            let image_view = ImageView::new(ImageInfo::new(PixelFormat::Bgr8,tt.width(),tt.height()),tt.as_raw());


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


// use ort::{GraphOptimizationLevel,Result, Session};
//
// fn main()->Result<()> {
//     let model = Session::builder()?
//         .with_optimization_level(GraphOptimizationLevel::Level3)?
//         .with_intra_threads(4)?
//         .commit_from_file("D:\\temp\\aaa\\resnet18_imagenet.onnx")?;
//     //
//     // let outputs = model.run(ort::inputs!["image" => image]?)?;
//     // let predictions = outputs["output0"].try_extract_tensor::<f32>()?;
//
//     Ok(())
// }