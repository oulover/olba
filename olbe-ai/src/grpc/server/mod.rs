use std::sync::Arc;
use tonic::transport::Server;
use olbe_grpc::olbe::ai::face::ai_face_server;
use crate::AppContext;
use crate::grpc::server::grpc_face_server::GrpcFaceServer;
use crate::service::ai_service::OlAiService;

pub mod grpc_face_server;

pub async fn start_grpc_server(ctx: Arc<AppContext>) {
    tokio::spawn(async move {
        let addr = std::env::var("GRPC_BIND")
            .unwrap_or_else(|_| "0.0.0.0:9095".into())
            .parse()
            .unwrap();
        log::info!("Grpc start at {:?}",addr);
        Server::builder()
            .add_service(ai_face_server::AiFaceServer::new(GrpcFaceServer::new(ctx.container.resolve().await.unwrap())))
            .serve(addr)
            .await.unwrap();
    });
}