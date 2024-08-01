use async_di::ResolverBuilder;
use crate::service::ai_service::r#impl::AiServiceProvider;

pub(crate) mod service;
pub(crate) mod service_impl;
mod grpc_service;
pub(crate) mod ai_service;

pub fn configure_di(b: &mut ResolverBuilder) {
    b.register(AiServiceProvider);
}



