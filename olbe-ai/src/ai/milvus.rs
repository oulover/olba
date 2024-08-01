use std::sync::Arc;
use milvus::client::Client;
use milvus::data::FieldColumn;
use milvus::schema::{CollectionSchema, CollectionSchemaBuilder, FieldSchema};
use anyhow::Result;
use async_di::{Provider, ProvideResult, ResolverRef};

pub type MilvusClient = std::sync::Arc<Client>;

pub struct MilvusClientProvider;
impl Provider for MilvusClientProvider{
    type Ref = MilvusClient;

    async fn provide(&self, resolver: &ResolverRef) -> ProvideResult<Self::Ref> {
        get_client().await.map_err(Into::into)
    }
}

pub async fn get_client() -> Result<MilvusClient> {
    Ok(Arc::new(Client::new("http://120.46.194.67:19530").await?))
}


pub struct UserFaceFeature {
    pub id: i64,
    pub feature: Vec<f32>,
    pub user_id: i64,
}
impl UserFaceFeature {
    pub fn insert_data(v: UserFaceFeature) -> Vec<FieldColumn> {
        let mut field: Vec<FieldColumn> = vec![];
        field.push(Self::id(v.id));
        field.push(Self::feature(v.feature));
        field.push(Self::user_id(v.user_id));
        field
    }

    pub fn id(v: i64) -> FieldColumn {
        FieldColumn::new(&FieldSchema::new_primary_int64(
            "id",
            "feature field", false,
        ), vec![v])
    }

    pub fn feature(v: Vec<f32>) -> FieldColumn {
        FieldColumn::new(&FieldSchema::new_float_vector(
            "feature",
            "feature field", 512,
        ), v)
    }

    pub fn user_id(v: i64) -> FieldColumn {
        FieldColumn::new(&FieldSchema::new_int64(
            "user_id",
            "feature field",
        ), vec![v])
    }


    pub fn schema() -> anyhow::Result<CollectionSchema> {
        CollectionSchemaBuilder::new("user_face_feature", "my_user_face")
            .add_field(FieldSchema::new_primary_int64(
                "id",
                "primary key field",
                false,
            ))
            .add_field(FieldSchema::new_float_vector(
                "feature",
                "feature field",
                512,
            ))
            .add_field(FieldSchema::new_int64(
                "user_id",
                "feature field",
            ))
            .build().map_err(Into::into)
    }
}
