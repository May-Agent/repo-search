//! The client of the `search` daemon: its wire protocol and the connection
//! that speaks it, with no dependency on the indexing and retrieval core.
//! The `launcher` feature adds starting the daemon from a program path.

/// The library client of the `search` daemon.
pub mod client;
/// Starting the daemon: where it lives, how it is installed or spawned and
/// the sequence that connects to it.
#[cfg(feature = "launcher")]
pub mod launcher;
/// The daemon's wire types and JSON-lines framing.
pub mod protocol;
