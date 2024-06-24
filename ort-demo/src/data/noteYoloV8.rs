#![allow(clippy::manual_retain)] // 允许手动处理内存，忽略clippy的警告

use std::path::Path; // 导入Path模块，用于处理文件路径

use image::{GenericImageView, imageops::FilterType}; // 导入image库，用于图像处理
use ndarray::{Array, Axis, s}; // 导入ndarray库，用于多维数组操作
use ort::{CUDAExecutionProvider, inputs, Session, SessionOutputs}; // 导入onnxruntime库，用于执行ONNX模型
use raqote::{DrawOptions, DrawTarget, LineJoin, PathBuilder, SolidSource, Source, StrokeStyle}; // 导入raqote库，用于图像绘制
use show_image::{AsImageView, event, ImageInfo, ImageView, PixelFormat, WindowOptions}; // 导入show_image库，用于显示图像

// 定义一个BoundingBox结构体，用于存储边界框的坐标
struct BoundingBox {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
}

// 计算两个边界框的交集面积
fn intersection(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
    (box1.x2.min(box2.x2) - box1.x1.max(box2.x1)) * (box1.y2.min(box2.y2) - box1.y1.max(box2.y1))
}

// 计算两个边界框的并集面积
fn union(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
    ((box1.x2 - box1.x1) * (box1.y2 - box1.y1)) + ((box2.x2 - box2.x1) * (box2.y2 - box2.y1)) - intersection(box1, box2)
}

// 定义YOLOV8M模型的URL和类别标签
const YOLOV8M_URL: &str = "D:\\temp\\aaa\\yolov8m.onnx";
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

#[show_image::main]
fn main() -> ort::Result<()> {
    tracing_subscriber::fmt::init(); // 初始化日志记录

    ort::init() // 初始化ONNX运行时
        .with_execution_providers([CUDAExecutionProvider::default().build()]) // 使用CUDA执行提供者
        .commit()?;
    let f = "D:\\temp\\aaa\\baseball.jpg"; // 待检测的图像路径
    let original_img = image::open(Path::new(f)).unwrap();
    let (img_width, img_height) = (original_img.width(), original_img.height()); // 获取图像尺寸
    let img = original_img.resize_exact(640, 640, FilterType::CatmullRom); // 将图像缩放到640x640
    let mut input = Array::zeros((1, 3, 640, 640)); // 创建一个全零数组，用于存储模型的输入
    for pixel in img.pixels() { // 遍历图像的每个像素
        let x = pixel.0 as _; // 获取像素的x坐标
        let y = pixel.1 as _; // 获取像素的y坐标
        let [r, g, b, _] = pixel.2.0; // 获取像素的RGB值
        input[[0, 0, y, x]] = (r as f32) / 255.; // 将RGB值归一化并存储到输入数组中
        input[[0, 1, y, x]] = (g as f32) / 255.;
        input[[0, 2, y, x]] = (b as f32) / 255.;
    }

    let model = Session::builder()? // 创建一个ONNX会话构建器
        .commit_from_file(YOLOV8M_URL)?; // 从文件加载模型并创建会话

    // 运行YOLOv8模型进行推理
    let outputs: SessionOutputs = model.run(inputs!["images" => input.view()])?;
    let output = outputs["output0"].try_extract_tensor::<f32>()?.t().into_owned(); // 获取输出张量

    let mut boxes = Vec::new(); // 创建一个向量，用于存储检测到的边界框
    let output = output.slice(s![.., .., 0]); // 切片输出张量
    for row in output.axis_iter(Axis(0)) { // 遍历输出张量的每一行
        println!("row---{:?}", row); // 打印当前行的内容
        let row: Vec<_> = row.iter().copied().collect(); // 将当前行转换为向量
        println!("row-Vec--{:?}", row); // 打印转换后的向量
        let row22: Vec<_> = row.iter().copied().collect(); // 再次将当前行转换为向量
        let row22: Vec<_> = row22
            .iter()
            // skip bounding box coordinates
            .skip(4) // 跳过前四个元素（边界框坐标）
            .enumerate() // 枚举剩余元素
            .map(|(index, value)| (index, *value)).collect(); // 收集类别ID和概率
        println!("row22-Vec--{:?}", row22); // 打印收集后的向量

        let (class_id, prob) = row
            .iter()
            // skip bounding box coordinates
            .skip(4) // 跳过前四个元素（边界框坐标）
            .enumerate() // 枚举剩余元素
            .map(|(index, value)| (index, *value))
            .reduce(|accum, row| if row.1 > accum.1 { row } else { accum }) // 找到概率最高的类别
            .unwrap();
        if prob < 0.5 { // 如果最高概率小于0.5，则忽略当前边界框
            continue;
        }
        let label = YOLOV8_CLASS_LABELS[class_id]; // 获取类别标签
        let xc = row[0] / 640. * (img_width as f32); // 计算边界框中心x坐标
        let yc = row[1] / 640. * (img_height as f32); // 计算边界框中心y坐标
        let w = row[2] / 640. * (img_width as f32); // 计算边界框宽度
        let h = row[3] / 640. * (img_height as f32); // 计算边界框高度
        boxes.push((
            BoundingBox { // 将边界框坐标和标签添加到边界框向量
                x1: xc - w / 2.,
                y1: yc - h / 2.,
                x2: xc + w / 2.,
                y2: yc + h / 2.,
            },
            label,
            prob
        ));
    }

    boxes.sort_by(|box1, box2| box2.2.total_cmp(&box1.2)); // 根据概率对边界框进行排序
    let mut result = Vec::new(); // 创建一个向量，用于存储最终检测结果

    while !boxes.is_empty() { // 当边界框向量不为空时
        result.push(boxes[0]); // 将概率最高的边界框添加到最终检测结果
        boxes = boxes
            .iter()
            .filter(|box1| intersection(&boxes[0].0, &box1.0) / union(&boxes[0].0, &box1.0) < 0.7) // 过滤与当前边界框重叠面积大于0.7的边界框
            .copied()
            .collect(); // 收集过滤后的边界框
    }

    let mut dt = DrawTarget::new(img_width as _, img_height as _); // 创建一个绘制目标

    for (bbox, label, _confidence) in result { // 遍历最终检测结果
        let mut pb = PathBuilder::new(); // 创建一个路径构建器
        pb.rect(bbox.x1, bbox.y1, bbox.x2 - bbox.x1, bbox.y2 - bbox.y1); // 创建一个矩形路径
        let path = pb.finish(); // 完成路径构建
        let color = match label { // 根据类别标签选择颜色
            "baseball bat" => SolidSource { r: 0x00, g: 0x10, b: 0x80, a: 0x80 },
            "baseball glove" => SolidSource { r: 0x20, g: 0x80, b: 0x40, a: 0x80 },
            _ => SolidSource { r: 0x80, g: 0x10, b: 0x40, a: 0x80 }
        };
        dt.stroke( // 在绘制目标上绘制边界框
                   &path,
                   &Source::Solid(color),
                   &StrokeStyle {
                       join: LineJoin::Round,
                       width: 4.,
                       ..StrokeStyle::default()
                   },
                   &DrawOptions::new(), // 使用默认的绘制选项
        );
    }

    let overlay: show_image::Image = dt.into(); // 将绘制目标转换为图像

    let window = show_image::context() // 创建一个显示图像的上下文
        .run_function_wait(move |context| -> Result<_, String> { // 在新线程中运行函数并等待
            let mut window = context // 创建一个新窗口
                .create_window(
                    "ort + YOLOv8", // 窗口标题
                    WindowOptions {
                        size: Some([img_width, img_height]), // 窗口大小与图像尺寸相同
                        ..WindowOptions::default()
                    },
                )
                .map_err(|e| e.to_string())?; // 处理可能出现的错误
            // window.set_image("baseball", &original_img.as_image_view().map_err(|e| e.to_string())?);

            let tt = original_img.to_rgb8(); // 将原始图像转换为RGB8格式
            let image_view = ImageView::new( // 创建一个图像视图
                                             ImageInfo::new(PixelFormat::Bgr8, tt.width(), tt.height()), // 图像信息
                                             tt.as_raw(), // 图像数据
            );

            window.set_image("baseball", &image_view); // 在窗口中设置原始图像
            window.set_overlay("yolo", &overlay.as_image_view().map_err(|e| e.to_string())?, true); // 在窗口中设置叠加层
            Ok(window.proxy()) // 返回窗口代理
        })
        .unwrap();

    for event in window.event_channel().unwrap() { // 循环处理窗口事件
        if let event::WindowEvent::KeyboardInput(event) = event { // 如果是键盘输入事件
            if event.input.key_code == Some(event::VirtualKeyCode::Escape) && event.input.state.is_pressed() { // 如果按下了Escape键
                break; // 退出循环
            }
        }
    }

    Ok(()) // 返回成功
}
