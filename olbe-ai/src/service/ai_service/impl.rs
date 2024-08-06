use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;
use async_di::{Provider, ProvideResult, ResolverRef};
use anyhow::Result;
use image::DynamicImage;
use milvus::index::MetricType;
use milvus::mutate::InsertOptions;
use milvus::query::SearchOptions;
use milvus::value::{Value, ValueVec};
use crate::service::ai_service::ai::AiSession;
use crate::service::ai_service::ai::milvus::{MilvusClient, UserFaceFeature};
use crate::service::ai_service::{AiService, OlAiService, UserFaceFind};

pub struct AiServiceProvider;

#[async_trait::async_trait]
impl Provider for AiServiceProvider {
    type Ref = OlAiService;

    async fn provide(&self, resolver: &ResolverRef) -> ProvideResult<Self::Ref> {
        Ok(Arc::new(AiServiceImpl::new(resolver.resolve().await?, resolver.resolve().await?)?))
    }
}


#[derive(Debug)]
pub(crate) struct AiServiceImpl {
    pub ai_session: Arc<AiSession>,
    pub milvus_client: MilvusClient,
    _unused: PhantomData<()>,
}

impl AiServiceImpl {
    pub fn new(milvus_client: MilvusClient, ai_session: Arc<AiSession>) -> Result<Self> {
        Ok(Self { ai_session, milvus_client, _unused: Default::default() })
    }
}
#[async_trait::async_trait]
impl AiService for AiServiceImpl {
    async fn search_face(&self, radius: Option<f32>, limit: Option<u32>, face_img_byte: &[u8]) -> Result<Vec<UserFaceFind>> {
        let face_img = image::load_from_memory(face_img_byte)?;
        let face_boxes = self.ai_session.get_face_boxes(&face_img)?;

        let mut face_img = self.ai_session.get_face_img(&face_img, &face_boxes);

        let r = face_img.pop().unwrap();

        let feature = self.ai_session.get_face_feature(&r)?;
        let feature: Vec<_> = feature.iter().map(|o| *o).collect();

        let mut search_opt = SearchOptions::default()

            .metric_type(MetricType::IP)
            .output_fields(vec![String::from("user_id"), String::from("id")]);

        if let Some(limit) = limit {
            search_opt = search_opt.limit(limit as _);
        }
        if let Some(radius) = radius {
            search_opt = search_opt.radius(radius);
        }

        let search = self.milvus_client.search(UserFaceFeature::schema()?.name(),
                                               vec![Value::FloatArray(Cow::from(feature))],
                                               "feature", &search_opt).await?;

        let mut users = vec![];
        for s in search {
            let t = s.field.into_iter().fold(HashMap::new(), |mut acc, field| {
                acc.entry(field.name.clone()).or_insert(field);
                acc
            });
            let user_ids = t.get(&String::from("user_id"));
            let ids = t.get(&String::from("id"));
            if let Some(user_ids) = user_ids {
                if let Some(ids) = ids {
                    match &user_ids.value {
                        ValueVec::Long(user_ids) => {
                            match &ids.value {
                                ValueVec::Long(ids) => {
                                    for ((id, uid, ), score) in ids.iter().zip(user_ids.iter()).zip(s.score.iter()) {
                                        users.push(UserFaceFind {
                                            id: *id,
                                            user_id: *uid,
                                            score: *score,
                                        })
                                    }
                                }
                                _ => {}
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        Ok(users)
    }

    async fn register_face(&self, user_id: i64, face_img: DynamicImage) -> Result<bool> {
        let face_boxes = self.ai_session.get_face_boxes(&face_img)?;

        let mut face_img = self.ai_session.get_face_img(&face_img, &face_boxes);

        let r = face_img.pop().unwrap();

        let feature = self.ai_session.get_face_feature(&r).unwrap();
        let feature: Vec<_> = feature.iter().map(|o| *o).collect();

        let t = UserFaceFeature::insert_data(UserFaceFeature {
            id: chrono::Utc::now().timestamp(),
            feature: feature.clone(),
            user_id,
        });

        self.milvus_client.insert(UserFaceFeature::schema()?.name(), t.clone(), Some(InsertOptions::default())).await?;
        self.milvus_client.flush(UserFaceFeature::schema()?.name()).await?;
        Ok(true)
    }
}