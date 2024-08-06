use std::sync::Arc;
use async_trait::async_trait;
use tonic::{Code, Request, Response, Status};
use olbe_grpc::olbe::ai::face::{FaceResult, RegisterFaceReply, RegisterFaceRequest, SearchFaceReply, SearchFaceRequest};
use crate::service::ai_service::OlAiService;

#[derive(Clone, Debug)]
pub struct GrpcFaceServer {
    ai_face: OlAiService,
}
impl GrpcFaceServer {
    pub fn new(ai_face: OlAiService) -> Self {
        Self { ai_face }
    }
}

#[async_trait]
impl olbe_grpc::olbe::ai::face::ai_face_server::AiFace for GrpcFaceServer {
    async fn search(&self, request: Request<SearchFaceRequest>) -> Result<Response<SearchFaceReply>, Status> {
        let req =  request.get_ref();

        let r = self.ai_face.search_face(req.radius,req.limit,&req.feature)
            .await.map_err(|err| Status::new(Code::Internal, err.to_string()))?;
        let results: Vec<_> = r.into_iter().map(|this| FaceResult {
            id: this.id,
            user_id: this.user_id,
            score: this.score,
        }).collect();

        Ok(Response::new(SearchFaceReply { results }))
    }

    async fn register(&self, request: Request<RegisterFaceRequest>) -> Result<Response<RegisterFaceReply>, Status> {
        let req =  request.get_ref();
        self.ai_face.register_face(req.user_id,&req.feature)
            .await.map_err(|err| Status::new(Code::Internal, err.to_string()))?;
        Ok(Response::new(RegisterFaceReply { result:true }))
    }
}