//! `repo-search` library surface.

/// The `search` client: argument parsing, the agent check, progress and
/// output rendering.
pub mod cli;
/// The `search` daemon.
pub mod daemon;
/// Installing and replacing the per-user LaunchAgent.
#[cfg(target_os = "macos")]
pub mod install;
/// Event-driven waits on kqueue for the install flow.
#[cfg(target_os = "macos")]
pub mod kqueue;
/// launchd socket activation and the LaunchAgent document.
#[cfg(target_os = "macos")]
pub mod launchd;
/// First-use download and verified unpacking of the model archive.
pub mod model_store;
/// The per-user paths: `~/Library` on macOS, XDG on Linux.
pub mod paths;
/// Spawn-on-demand: the client's launcher without launchd.
pub mod spawn;
/// Conversions between core's types and the client crate's wire types.
pub mod wire;
