use anyhow::Result;

use olbe_ai::{init, start};

#[tokio::main]
#[show_image::main]
async fn main() -> Result<()> {
    init()?;
    start().await?;
    Ok(())
}
