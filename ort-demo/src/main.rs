// 忽略clippy警告，避免手动retain操作
#![allow(clippy::manual_retain)]

use std::path::Path;

// 引入图像处理库
use image::{GenericImageView, imageops::FilterType};
// 引入ndarray库用于数组操作
use ndarray::{Array, Axis, s};
// 引入ONNX Runtime Rust绑定库
use ort::{CUDAExecutionProvider, inputs, Session, SessionOutputs};
// 引入raqote库用于绘图
use raqote::{DrawOptions, DrawTarget, LineJoin, PathBuilder, SolidSource, Source, StrokeStyle};
// 引入show_image库用于显示图像
use show_image::{AsImageView, event, ImageInfo, ImageView, PixelFormat, WindowOptions};

// 定义一个简单的边界框结构体
#[derive(Debug, Clone, Copy)]
struct BoundingBox {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
}

// 指定模型的本地路径
const DET_10G_URL: &str = "D:\\temp\\aaa\\buffalo_l\\det_10g.onnx";

#[show_image::main]
fn main() -> ort::Result<()> {
    // 初始化日志记录器
    tracing_subscriber::fmt::init();

    // 初始化ONNX Runtime环境，使用CUDA执行提供者
    ort::init()
        .with_execution_providers([CUDAExecutionProvider::default().build()])
        .commit()?;

    // 打开图片文件
    let f = "D:\\Temp\\aaa\\hg11.jpg";
    let original_img = image::open(Path::new(f)).unwrap();
    // 获取原始图像的宽度和高度
    let (img_width, img_height) = (original_img.width(), original_img.height());
    // 将图像缩放到模型要求的尺寸
    let img = original_img.resize_exact(640, 640, FilterType::CatmullRom);
    // 创建一个用于模型输入的ndarray
    let mut input = Array::zeros((1, 3, 640, 640));
    // 遍历每个像素，将其归一化后放入输入数组
    for pixel in img.pixels() {
        let x = pixel.0 as _;
        let y = pixel.1 as _;
        let [r, g, b, _] = pixel.2.0;
        input[[0, 0, y, x]] = (r as f32) / 255.;
        input[[0, 1, y, x]] = (g as f32) / 255.;
        input[[0, 2, y, x]] = (b as f32) / 255.;
    }

    // 加载ONNX模型
    let model = Session::builder()?.commit_from_file(DET_10G_URL)?;

    // 运行面部检测推理
    let outputs: SessionOutputs = model.run(inputs!["input.1" => input.view()]?)?;
    // 提取输出张量并转置
    let output = outputs["451"].try_extract_tensor::<f32>()?.t().into_owned();

    // 解析输出结果，获取边界框
    let mut boxes = Vec::new();
    for row in output.axis_iter(Axis(0)) {
        let row: Vec<_> = row.iter().copied().collect();
        // 将边界框坐标从归一化值转换回图像坐标系
        let x1 = row[0] * (img_width as f32);
        let y1 = row[1] * (img_height as f32);
        let x2 = row[2] * (img_width as f32);
        let y2 = row[3] * (img_height as f32);
        // 将边界框添加到向量中
        boxes.push(BoundingBox { x1, y1, x2, y2 });
    }

    // 创建一个用于绘制的DrawTarget
    let mut dt = DrawTarget::new(img_width as _, img_height as _);

    // 遍历所有边界框，绘制矩形框
    for bbox in boxes {
        let mut pb = PathBuilder::new();
        pb.rect(bbox.x1, bbox.y1, bbox.x2 - bbox.x1, bbox.y2 - bbox.y1);
        let path = pb.finish();
        // 设置矩形框的颜色和样式
        let color = SolidSource { r: 0x00, g: 0x10, b: 0x80, a: 0x80 };
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

    // 将DrawTarget转换为show_image图像
    let overlay: show_image::Image = dt.into();

    // 创建一个窗口，设置窗口标题和尺寸
    let window = show_image::context()
        .run_function_wait(move |context| -> Result<_, String> {
            let mut window = context
                .create_window(
                    "Face Detection",
                    WindowOptions {
                        size: Some([img_width, img_height]),
                        ..WindowOptions::default()
                    },
                )
                .map_err(|e| e.to_string())?;
            // 加载图像并设置为窗口背景
            let tt = original_img.to_rgb8();
            let image_view = ImageView::new(ImageInfo::new(PixelFormat::Bgr8, tt.width(), tt.height()), tt.as_raw());
            window.set_image("face", &image_view);
            // 设置叠加层，用于显示边界框
            window.set_overlay("face_detection", &overlay.as_image_view().map_err(|e| e.to_string())?, true);
            Ok(window.proxy())
        })
        .unwrap();

    // 监听窗口事件，如果按下Esc键，则退出程序
    for event in window.event_channel().unwrap() {
        if let event::WindowEvent::KeyboardInput(event) = event {
            if event.input.key_code == Some(event::VirtualKeyCode::Escape) && event.input.state.is_pressed() {
                break;
            }
        }
    }

    Ok(())
}