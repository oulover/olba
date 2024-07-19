use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::sync::{Arc, Mutex, Once};
use ort::{CUDAExecutionProvider, Session};
use lazy_static::lazy_static;
use anyhow::Result;
mod nms;

const DET_10G_URL: &str = "D:\\temp\\aaa\\buffalo_l\\det_10g.onnx";
const DET_R50G_URL: &str = "D:\\temp\\aaa\\buffalo_l\\w600k_r50.onnx";

pub fn init() -> Result<()> {
    ort::init()
        .with_execution_providers([CUDAExecutionProvider::default().build()])
        .commit()?;

    Ok(())
}

pub struct AiSession {
    det_mod: Session,
    feature_mod: Session,
}
impl AiSession {
    pub fn get(&self) -> Result<()> {
        Ok(())
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

    pub fn get_instance() -> Result<&'static Arc<AiSession>> {
        lazy_static! {
             static  ref INSTANCE_AI_SESSION:Arc<AiSession> = AiSession::new().unwrap();
        }
        Ok(&INSTANCE_AI_SESSION)
    }
}