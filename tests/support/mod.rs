//! Fixtures shared by the crate's unit tests (through `#[path]` includes)
//! and its integration tests (`mod support;`).

pub mod bound;
pub mod client;
#[cfg(target_os = "macos")]
pub mod daemon;
pub mod embed_server;
pub mod host;
pub mod http;
#[cfg(target_os = "macos")]
pub mod launchd;
pub mod lock;
#[cfg(target_os = "macos")]
pub mod model;
pub mod repo;
pub mod tokens;
