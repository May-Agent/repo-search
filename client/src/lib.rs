//! The client of the `search` daemon: its wire protocol and the connection
//! that speaks it, with no dependency on the indexing and retrieval core.

/// The library client of the `search` daemon.
pub mod client;
/// The daemon's wire types and JSON-lines framing.
pub mod protocol;
