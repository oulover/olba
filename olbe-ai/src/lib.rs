#![allow(unused)]
pub mod service;
pub mod config;
pub mod middleware;
pub mod error;
pub mod ai;
pub mod router;
pub mod controller;
mod domain;
pub mod di;

use std::sync::Arc;
use anyhow::Result;
use async_di::Container;
use crate::ai::AiProvider;
use crate::ai::milvus::MilvusClientProvider;

pub struct AppContext {
    container: Container,
}
impl AppContext {
    pub fn new(container: Container) -> Self {
        Self {
            container,
        }
    }
}

pub fn init() -> Result<()> {
    simple_logger::init().unwrap();
    log::info!("----init app---");
    dotenv::dotenv().ok();
    tracing_subscriber::fmt::init();
    // ai::init()?;

    Ok(())
}

pub fn configure_di() -> Result<Container> {
    Ok(Container::new(|b| {
        b.register(AiProvider);
        b.register(MilvusClientProvider);
        service::configure_di(b);
    }))
}


pub async fn start(app_context: Arc<AppContext>) -> Result<()> {
    router::start(app_context).await?;
    Ok(())
}