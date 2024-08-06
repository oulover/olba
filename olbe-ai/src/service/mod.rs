use async_di::ResolverBuilder;
use crate::service::ai_service::r#impl::AiServiceProvider;


pub mod ai_service;

pub fn configure_di(b: &mut ResolverBuilder) {
    b.register(AiServiceProvider);
}



