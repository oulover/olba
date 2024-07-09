extern crate opencv;
use opencv::core::{self, Mat, Vector};
use opencv::highgui::{self, imshow, wait_key, destroy_all_windows};
use opencv::imgcodecs::imread;
use opencv::prelude::*;
use opencv::videoio::{self, VideoCapture, CAP_FFMPEG,CAP_ANY};

fn main() {
    let url = "D:\\Temp\\aaa\\ssss22.mp4"; // 替换为你的视频 URL

    let mut cap = VideoCapture::from_file_def(url).unwrap();
    if !cap.is_opened().expect("无法检查视频流是否打开") {
        println!("无法打开2视频流");
        return;
    }

    let mut frame = Mat::default();
    loop {
        if !cap.read(&mut frame).expect("无法读取帧") {
            println!("视频结束或无3法读取帧");
            break;
        }

        imshow("Video", &frame).expect("无法显示帧");
        if wait_key(30).expect("等待按键失败") > 0 {
            break;
        }
    }

    destroy_all_windows().expect("无法关闭窗口");
}