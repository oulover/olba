use anyhow::Error;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use thiserror::Error;
#[derive(Debug, Error)]
pub enum AppError {
    #[error("Resource not found")]
    NotFound,

    #[error("Resource not found")]
    InnerError,

    #[error("Error msg is {msg}")]
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

impl From<anyhow::Error> for AppError{
    fn from(value: Error) -> Self {
        Self::ErrorMsg {msg:value.to_string()}
    }
}
