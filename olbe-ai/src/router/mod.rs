use anyhow::Result;
use axum::response::IntoResponse;
use axum::Router;
use axum::routing::get;
use crate::service::service::DemoService;
use crate::service::service_impl::DemoServiceImpl;
use crate::controller;
use crate::controller::RespVO;

pub fn init() -> Result<()> {
    Ok(())
}

pub async fn start() -> Result<()> {
    let app = Router::new()
        .nest("/api", controller::router());

    let addr = std::env::var("WEB_ADDR").unwrap_or("0.0.0.0:9966".to_string());
    log::info!("App start at {}",addr);
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.map_err(|err| err.into())
}

