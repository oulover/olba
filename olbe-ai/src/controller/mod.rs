use std::fmt::Display;
use axum::body::Body;
use axum::response::{IntoResponse, Response};
use axum::{Json, Router};

use serde::{Deserialize, Serialize};
use serde::de::DeserializeOwned;

mod web_demo_service;
mod ai_controller;

pub(crate) fn router() -> Router {
    Router::new().nest("/user", web_demo_service::router()).nest("/face", ai_controller::router())
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum Code {
    Ok200 = 200,
    Err400 = 400,
    Err500 = 500,

}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct R<T> {
    pub code: Code,
    pub msg: Option<String>,
    pub data: Option<T>,
}


impl<T> IntoResponse for R<T>
where
    T: Serialize + DeserializeOwned + Clone,
{
    fn into_response(self) -> Response {
        Json(self).into_response()
    }
}
impl<T> R<T> {
    pub fn new(code: Code, data: Option<T>, msg: Option<String>) -> R<T> {
        Self {
            code,
            msg,
            data,
        }
    }
    pub fn ok_msg(data: Option<T>, msg: Option<String>) -> R<T> {
        Self::new(Code::Ok200, data, msg)
    }
    pub fn ok(data: Option<T>) -> R<T> {
        Self::ok_msg(data, None)
    }

    pub fn err_data_msg(data: Option<T>, msg: Option<String>) -> R<T> {
        Self::new(Code::Err500, data, msg)
    }
    pub fn err_msg(msg: Option<String>) -> R<()> {
        R::<()>::err_data_msg(None::<()>, msg)
    }
    pub fn err(msg: Option<String>) -> R<()> {
        Self::err_msg(msg)
    }

    pub fn result(r:Result<>)-> R<T> {

    }
}

