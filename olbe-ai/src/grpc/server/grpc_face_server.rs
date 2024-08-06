use std::sync::Arc;
use async_trait::async_trait;
use tonic::Code;
use olbe_grpc::olbe::ai::face::{FaceResult, SearchFaceReply, SearchFaceRequest};
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
    async fn search(&self, request: tonic::Request<SearchFaceRequest>) -> Result<tonic::Response<SearchFaceReply>, tonic::Status> {
        let req =  request.get_ref();

        let r = self.ai_face.search_face(req.radius,req.limit,&req.feature)
            .await.map_err(|err| tonic::Status::new(Code::Internal, err.to_string()))?;
        let results: Vec<_> = r.into_iter().map(|this| FaceResult {
            id: this.id,
            user_id: this.user_id,
            score: this.score,
        }).collect();

        Ok(tonic::Response::new(SearchFaceReply { results }))
    }
}