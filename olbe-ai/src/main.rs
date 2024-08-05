use std::collections::HashMap;
use std::sync::Arc;
use anyhow::Result;
use milvus::client::Client;
use milvus::data::FieldColumn;
use milvus::index::{IndexParams, IndexType};
use milvus::options::LoadOptions;
use milvus::query::QueryOptions;
use milvus::schema::{CollectionSchema, CollectionSchemaBuilder, FieldSchema};
use rand::Rng;
// use milvus::client::Client;
// use milvus::schema::{CollectionSchemaBuilder, FieldSchema};
use olbe_ai::{ai, AppContext, configure_di, init, start};
const DEFAULT_VEC_FIELD: &str = "embed";
const DIM: i64 = 256;
#[tokio::main]
#[show_image::main]
async fn main() -> Result<()> {
    init()?;



    let client = Client::new("http://120.46.194.67:19530").await?;
    let schema =ai::milvus::UserFaceFeature::schema()?;
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

    // if let Err(err) = hello_milvus(&client, &schema).await {
    //     println!("failed to run hello milvus: {:?}", err);
    // }
    // // client.drop_collection(schema.name()).await?;

    let app_ctx = AppContext::new(configure_di()?);

    start(Arc::new(app_ctx)).await?;
    Ok(())

}


async fn hello_milvus(client: &Client, collection: &CollectionSchema) -> Result<()> {
    let mut embed_data = Vec::<f32>::new();
    for _ in 1..=DIM * 1000 {
        let mut rng = rand::thread_rng();
        let embed = rng.gen();
        embed_data.push(embed);
    }
    let embed_column =
        FieldColumn::new(collection.get_field(DEFAULT_VEC_FIELD).unwrap(), embed_data);



    client
        .insert(collection.name(), vec![embed_column], None)
        .await?;
    client.flush(collection.name()).await?;
    let index_params = IndexParams::new(
        "feature_index".to_owned(),
        IndexType::IvfFlat,
        milvus::index::MetricType::IP,
        HashMap::from([("nlist".to_owned(), "32".to_owned())]),
    );
    client
        .create_index(collection.name(), DEFAULT_VEC_FIELD, index_params)
        .await?;
    client
        .load_collection(collection.name(), Some(LoadOptions::default()))
        .await?;

    let options = QueryOptions::default();

    let result = client.query(collection.name(), "id > 0", &options).await?;

    println!(
        "result num: {}",
        result.first().map(|c| c.len()).unwrap_or(0),
    );

    Ok(())
}

