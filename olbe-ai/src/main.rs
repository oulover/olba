use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use milvus::client::Client;
use milvus::index::{IndexParams, IndexType};

use olbe_ai::{AppContext, configure_di, init, start};


#[tokio::main]
#[show_image::main]
async fn main() -> Result<()> {
    init()?;


    let client = Client::new("http://120.46.194.67:19530").await?;
    let schema =  olbe_ai::service::ai_service::ai::milvus::UserFaceFeature::schema()?;
    client.create_collection(schema.clone(), None).await?;

    let index_params = IndexParams::new(
        "feature_index".to_owned(),
        IndexType::IvfFlat,
        milvus::index::MetricType::IP,
        HashMap::from([("nlist".to_owned(), "32".to_owned())]),
    );
    client
        .create_index(schema.name(), "feature", index_params)
        .await?;

    let app_ctx = AppContext::new(configure_di()?);

    start(Arc::new(app_ctx)).await?;
    Ok(())
}

