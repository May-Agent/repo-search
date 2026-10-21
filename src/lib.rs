//! `repo-search` library surface.

/// The library client of the `search` daemon.
pub mod client;
/// The `search` daemon.
#[cfg(target_os = "macos")]
pub mod daemon;
/// Event-driven waits on kqueue for the install flow.
#[cfg(target_os = "macos")]
pub mod kqueue;
/// launchd socket activation and the LaunchAgent document.
#[cfg(target_os = "macos")]
pub mod launchd;
/// First-use download and verified unpacking of the model archive.
pub mod model_store;
/// The daemon's wire types and JSON-lines framing.
pub mod protocol;
