//! The repo-search embedding server: the model, its Core ML instances and
//! the hardware scheduling that used to live in the daemon, served behind
//! an OpenAI-compatible HTTP API. This crate carries the hardware half;
//! the HTTP service arrives with the loader.
#![cfg(target_os = "macos")]

pub mod ane;
pub mod backend;
pub mod chip;
pub mod coreml;
pub mod gpu;
pub mod scheduler;
#[cfg(feature = "stub")]
pub mod stub;
