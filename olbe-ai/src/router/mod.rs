use std::sync::Arc;
use anyhow::Result;
use axum::response::IntoResponse;
use axum::{Extension, Router};
use axum::extract::State;
use axum::routing::get;
use crate::service::service::DemoService;
use crate::service::service_impl::DemoServiceImpl;
use crate::{AppContext, controller};
use crate::controller::RespVO;

pub fn init() -> Result<()> {
    Ok(())
}

pub async fn start(ctx:Arc<AppContext>) -> Result<()> {
    let app = Router::new()
        .nest("/api", controller::router())
        .layer(Extension(ctx));

    let addr = std::env::var("WEB_ADDR").unwrap_or("0.0.0.0:9966".to_string());
    log::info!("App start at {}",addr);
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.map_err(|err| err.into())
}

