use std::sync::Arc;
use anyhow::Result;
use axum::{Extension, Router};
use axum::http::Uri;
use crate::{AppContext, controller};
use crate::controller::R;

pub fn init() -> Result<()> {
    Ok(())
}

pub async fn start(ctx:Arc<AppContext>) -> Result<()> {
    let app = Router::new().fallback(glob_fallback)
        .nest("/api", controller::router())
        .layer(Extension(ctx));

    let addr = std::env::var("WEB_ADDR").unwrap_or("0.0.0.0:9966".to_string());
    log::info!("App start at {}",addr);
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.map_err(|err| err.into())
}

async fn glob_fallback(uri: Uri) ->  R<()> {
    R::<()>::err(format!("No route for {uri}"))
}

