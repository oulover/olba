use std::fmt::Display;

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

pub type Result<T, E = AppError> = core::result::Result<T, E>;


#[derive(Debug )]
pub enum AppError {
    NotFound,
    InnerError,
    ErrorMsg{msg:String},
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        match self {
            AppError::NotFound => (StatusCode::NOT_FOUND,
                                   AppError::NotFound.to_string()).into_response(),
            AppError::InnerError => (StatusCode::INTERNAL_SERVER_ERROR,
                                     AppError::InnerError.to_string()).into_response(),

            AppError::ErrorMsg { .. } => (StatusCode::INTERNAL_SERVER_ERROR,
                                          AppError::InnerError.to_string()).into_response(),
        }
    }
}

impl Display for AppError{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let str = match self {
            AppError::NotFound => { String::from("NotFound") }
            AppError::InnerError => { String::from("InnerError") }
            AppError::ErrorMsg{msg} => { String::from("ErrorMsg :") + msg }
        };
        write!(f, "{}", str)
    }
}


impl <T> From<T> for AppError
where T:Into<anyhow::Error>{
    fn from(value: T) -> Self {
        let r :anyhow::Error=value.into();
        Self::ErrorMsg {msg:r.to_string()}
    }
}

