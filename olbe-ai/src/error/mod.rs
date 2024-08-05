use std::fmt::Display;
use anyhow::Error;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::de::StdError;
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

// #[derive(Debug)]
// pub enum AppError {
//     NotFound,
//     InnerError,
//     ErrorMsg{msg:String},
// }

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

// impl Display for AppError{
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         let str = match self {
//             AppError::NotFound => { String::from("NotFound") }
//             AppError::InnerError => { String::from("InnerError") }
//             AppError::ErrorMsg{msg} => { String::from("ErrorMsg :") + msg }
//         };
//         write!(f, "{}", str)
//     }
// }

// impl From<anyhow::Error> for AppError{
//     fn from(value: Error) -> Self {
//         Self::ErrorMsg {msg:value.to_string()}
//     }
// }


// impl <E> From<E> for AppError
// where
// E: StdError + Send + Sync + 'static{
//     fn from(value: E) -> Self {
//         Self::ErrorMsg {msg:format!("{:?}",value)}
//     }
// }

