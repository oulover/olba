use std::fmt::Display;

use axum::{Json, Router};
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde::de::DeserializeOwned;

mod web_demo_service;
mod ai_controller;

pub(crate) fn router() -> Router {
    Router::new().nest("/user", web_demo_service::router()).nest("/face", ai_controller::router())
}

#[derive(Debug, Clone)]
pub enum Code {
    Ok200 = 200,
    Err400 = 400,
    Err500 = 500,

}

impl Serialize for Code {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_i32(self.clone() as i32)
    }
}

impl<'de> Deserialize<'de> for Code {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value: i32 = Deserialize::deserialize(deserializer)?;
        match value {
            200 => Ok(Code::Ok200),
            400 => Ok(Code::Err400),
            500 => Ok(Code::Err500),
            _ => Err(serde::de::Error::custom(format!("Invalid code: {}", value))),
        }
    }
}


#[derive(Debug, Serialize, Clone)]
pub struct R<T> {
    pub code: Code,
    pub msg: Option<String>,
    pub data: Option<T>,
}


impl<T> IntoResponse for R<T>
where
    T: Serialize + Clone,
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
    pub fn ok_opt(data: Option<T>) -> R<T> {
        Self::ok_msg(data, None)
    }
    pub fn ok(data:T) -> R<T> {
        Self::ok_opt(Some(data))
    }

    pub fn err_data_msg(data: Option<T>, msg: Option<String>) -> R<T> {
        Self::new(Code::Err500, data, msg)
    }
    pub fn err_msg(msg: Option<String>) -> R<()> {
        R::<()>::err_data_msg(None::<()>, msg)
    }
    pub fn err_opt(msg: Option<String>) -> R<()> {
        Self::err_msg(msg)
    }
    pub fn err(msg: String) -> R<()> {
        Self::err_opt(Some(msg))
    }

    pub fn result<E: Display>(result: Result<T, E>) -> R<T> {
        match result {
            Ok(data) => { Self::ok(data) }
            Err(err) => { Self::err_data_msg(None::<T>, Some(err.to_string())) }
        }
    }
    pub fn result_opt<E: Display>(result: Result<Option<T>, E>) -> R<T> {
        match result {
            Ok(data) => { Self::ok_opt(data) }
            Err(err) => { Self::err_data_msg(None::<T>, Some(err.to_string())) }
        }
    }
}

