//! Fixtures shared by the crate's unit tests (through `#[path]` includes)
//! and its integration tests (`mod support;`).

pub mod client;
#[cfg(target_os = "macos")]
pub mod daemon;
#[cfg(target_os = "macos")]
pub mod host;
pub mod http;
#[cfg(target_os = "macos")]
pub mod model;
pub mod repo;
#[cfg(target_os = "macos")]
pub mod stub;
