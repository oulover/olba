use axum::{
    Router,
    extract::Request,
    middleware::{self, Next},
    response::Response,
    routing::get,
};
use axum::handler::Handler;
use axum::http::{StatusCode, Uri};
use axum::routing::any;
use tower_http::services::{ServeDir, ServeFile};

pub(crate) fn route_init() -> Router {
    let t = Router::new()
        .route("/", any(|| async {}))
        .layer(middleware::from_fn(api_middleware))
        .route("/hello", get(api_hello))
        .route_layer(middleware::from_fn(api_hello_middleware))

        ;

    let r = Router::new()
        .nest_service("/", ServeDir::new("dist/").not_found_service(ServeFile::new("dist/index.html")))

        .route("/static", get(static_info))
        .layer(middleware::from_fn(static_middleware))

        .nest("/api", t)
        .route_layer(middleware::from_fn(after_nest_middleware))
        .fallback(fallback)
        ;
    r
}

async fn fallback(uri: Uri) -> (StatusCode, String) {
    (StatusCode::NOT_FOUND, format!("No route for {uri}"))
}

pub async fn after_nest_middleware(
    request: Request,
    next: Next,
) -> Response {
    println!("------after_nest_middleware--");
    let response = next.run(request).await;
    response
}

pub async fn static_middleware(
    request: Request,
    next: Next,
) -> Response {
    println!("------static_middleware--");
    let response = next.run(request).await;
    response
}

///自定义中间件方法
pub async fn api_middleware(
    request: Request,
    next: Next,
) -> Response {
    println!("------api_middleware");
    let response = next.run(request).await;
    response
}

pub async fn api_hello_middleware(
    request: Request,
    next: Next,
) -> Response {
    println!("------api_hello_middleware");
    let response = next.run(request).await;
    response
}

async fn static_info() -> &'static str {
    "static_info"
}

// 定义处理函数
async fn hello() -> &'static str {
    "Hello"
}

async fn api_hello() -> &'static str {
    "API Hello"
}

async fn api_test() -> &'static str {
    "API Test"
}

async fn demo_one() -> &'static str {
    "Demo One"
}