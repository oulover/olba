use anyhow::Result;

use olbe_ai::{init, start};

#[tokio::main]
async fn main() -> Result<()> {
    init()?;
    start().await?;
    Ok(())
}
