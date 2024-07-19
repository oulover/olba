pub mod service;
pub mod config;
pub mod middleware;
pub mod error;
pub mod ai;
pub mod router;
pub mod controller;

use anyhow::Result;
pub fn init() -> Result<()> {
    simple_logger::init().unwrap();
    log::info!("----init app---");
    dotenv::dotenv().ok();

    ai::init()?;

    Ok(())
}


pub async fn start() -> Result<()> {
    router::start().await?;
    Ok(())
}