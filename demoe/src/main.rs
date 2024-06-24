mod log_config;
mod route;
mod user_dao;

use axum::{middleware, Router};
use axum::routing::get;
use fast_log::Config;
use tower_http::cors::{Any, CorsLayer};

async fn root() -> &'static str {
    "Hello, World!"
}

#[tokio::main]
async fn main() -> std::io::Result<()> {
    log_config::init().unwrap();
    println!("Hello, world!");
    log::info!("Commencing yak shaving{}", 0);
    let app = route::route_init() ;

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
    Ok(())
}
