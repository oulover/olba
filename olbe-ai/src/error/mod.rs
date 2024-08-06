use std::fmt::Display;

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use crate::controller::R;

pub type Result<T, E = AppError> = core::result::Result<T, E>;

#[derive(Debug)]
pub enum AppError {
    NotFound,
    InnerError,
    ErrorMsg { msg: String },
}



impl Display for AppError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let str = match self {
            AppError::NotFound => { String::from("NotFound") }
            AppError::InnerError => { String::from("InnerError") }
            AppError::ErrorMsg { msg } => { String::from("ErrorMsg :") + msg }
        };
        write!(f, "{}", str)
    }
}


impl<T> From<T> for AppError
where
    T: Into<anyhow::Error>,
{
    fn from(value: T) -> Self {
        let r: anyhow::Error = value.into();
        Self::ErrorMsg { msg: r.to_string() }
    }
}



