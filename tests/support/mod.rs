//! Fixtures shared by the crate's unit tests (through `#[path]` includes)
//! and its integration tests (`mod support;`).

pub mod http;
#[cfg(target_os = "macos")]
pub mod model;
pub mod repo;
#[cfg(target_os = "macos")]
pub mod stub;
