mod web_demo_service;
mod ai_controller;

use anyhow::Error;
use axum::response::{IntoResponse, Response};
use axum::Router;
use axum::routing::get;

pub(crate) fn router() ->Router{
    Router::new().nest("/user",web_demo_service::router()).nest("/face",ai_controller::router())
}




use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

pub const CODE_SUCCESS: &str = "200";
pub const CODE_FAIL: &str = "500";




#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RespVO<T> {
    pub code: Option<String>,
    pub msg: Option<String>,
    pub data: Option<T>,
}

impl<T> RespVO<T>
where
    T: Serialize + DeserializeOwned + Clone,
{
    pub fn from_result(result: anyhow::Result<T>) -> Self {
        match result {
            Ok(data) => Self {
                code: Some(CODE_SUCCESS.to_string()),
                msg: None,
                data: Some(data),
            },
            Err(e) => Self::from_error(e.to_string()),
        }
    }

    pub fn from(data: T) -> Self {
        Self {
            code: Some(CODE_SUCCESS.to_string()),
            msg: None,
            data: Some(data),
        }
    }

    pub fn from_error(error: String) -> Self {

        Self {
            code: Some(CODE_FAIL.to_string()),
            msg: Some(error),
            data: None,
        }
    }

    pub fn json(self) -> axum::Json<RespVO<T>> {
        axum::Json(self)
    }
}

impl<T> ToString for RespVO<T>
where
    T: Serialize + DeserializeOwned + Clone,
{
    fn to_string(&self) -> String {
        serde_json::to_string(self).unwrap()
    }
}
