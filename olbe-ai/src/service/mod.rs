use async_di::ResolverBuilder;
use crate::service::ai_service::r#impl::AiServiceProvider;


pub(crate) mod ai_service;

pub fn configure_di(b: &mut ResolverBuilder) {
    b.register(AiServiceProvider);
}



