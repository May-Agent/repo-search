//! `repo-search` library surface.

/// The `search` client: argument parsing, the agent check, progress and
/// output rendering.
pub mod cli;
/// The `search` daemon.
pub mod daemon;
/// launchd socket activation.
#[cfg(target_os = "macos")]
pub mod launchd;
/// First-use download and verified unpacking of the model archive.
pub mod model_store;
/// Conversions between core's types and the client crate's wire types.
pub mod wire;
