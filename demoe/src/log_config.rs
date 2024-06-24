use fast_log::{Config, Logger};
use fast_log::error::LogError;

pub(crate) fn init() -> Result<&'static Logger, LogError> {

    // fast_log::init(Config::new().file("target/test.log").chan_len(Some(100000)))

    fast_log::init(Config::new().console().file("target/test.log").chan_len(Some(100000)))
}