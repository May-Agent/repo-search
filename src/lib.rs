//! `repo-search` library surface.

/// The library client of the `search` daemon.
pub mod client;
/// First-use download and verified unpacking of the model archive.
pub mod model_store;
/// The daemon's wire types and JSON-lines framing.
pub mod protocol;
