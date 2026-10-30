# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.7](https://github.com/May-Agent/repo-search/compare/v0.0.6...v0.0.7) - 2026-10-30

### Fixed

- build the x86_64 linux dist on ubuntu-24.04

## [0.0.6](https://github.com/May-Agent/repo-search/compare/v0.0.5...v0.0.6) - 2026-10-30

### Added

- packaged ONNX model, auto-download, and EP selection on non-macOS
- ONNX embedding server for Linux

## [0.0.5](https://github.com/May-Agent/repo-search/compare/v0.0.4...v0.0.5) - 2026-10-29

### Added

- *(client)* move the daemon launcher into the client crate

## [0.0.4](https://github.com/May-Agent/repo-search/compare/v0.0.3...v0.0.4) - 2026-10-29

### Added

- client crate without the core dependency
- document collections indexed for caller-supplied documents

### Other

- publish repo-search-client with the other crates

## [0.0.3](https://github.com/May-Agent/repo-search/compare/v0.0.2...v0.0.3) - 2026-10-29

### Added

- *(daemon)* run the Linux daemon on demand
- build the library on Linux
- *(daemon)* bind its own socket and reach a server over TCP
- *(daemon)* seed a new worktree's index from a sibling
- *(daemon)* embed through the server over HTTP
- *(daemon)* index scheduler that embeds through the server
- *(install)* run the embedding server as its own launch agent
- *(daemon)* tokenizer store and prompt-aware token counter
- *(server)* standard embeddings API and search --server
- *(server)* embedding server crate with the hardware scheduler
- *(core)* model card, prompt template and embeddings wire types
- *(daemon)* keep the index in the per-user state dir

### Fixed

- *(search)* break score ties by content hash, not load order
- *(daemon)* key a worktree by its on-disk spelling
- *(daemon)* never drop the watcher's Gone behind a pending change
- *(daemon)* install the SIGTERM handler before reporting ready
- *(daemon)* run a waited or freshly seeded worktree's work at stage-1 urgency
- *(deps)* move tokio and pulldown-cmark to their patched releases
- *(daemon)* answer a connection a stop catches before its frame
- *(daemon)* answer queued clients when startup fails

### Other

- Merge remote-tracking branch 'origin/dev' into fix/knn-ties
- *(publish)* publish repo-search-server with the other crates
- *(spawn)* reap a leftover daemon before its temp dir goes
- Merge remote-tracking branch 'origin/dev' into feat/linux-daemon
- *(dist)* ship x86_64 and aarch64 Linux binaries
- Merge remote-tracking branch 'origin/dev' into feat/linux-daemon
- Merge remote-tracking branch 'origin/dev' into feat/linux-daemon
- *(cli)* put daemon launching behind a Launcher trait
- size thread pools from the CPU count
- drop the in-process embedder API
- Merge remote-tracking branch 'origin/dev' into feat/embedding-server

## [0.0.2](https://github.com/May-Agent/repo-search/compare/v0.0.1...v0.0.2) - 2026-10-21

### Other

- *(release)* compute versions from tags again

## [0.0.1](https://github.com/May-Agent/repo-search/compare/v0.0.0...v0.0.1) - 2026-10-21

### Added

- the search CLI client
- install the LaunchAgent into user/<uid> on demand
- identify the build in the handshake and add kqueue waits
- *(convert)* three-bucket model (128, 256, 512)
- *(core)* retrieval, fusion and output
- *(core)* documentation and configuration chunkers behind a file-level chunker trait
- *(convert)* Core ML conversion pipeline for EmbeddingGemma 2
- *(core)* incremental worktree reconcile with freshness report
- *(core)* int8 KNN with fp16 re-rank and flat-file vector cache
- *(core)* tree-sitter based Rust chunker
- *(core)* per-worktree index store and shared vector cache
- workspace scaffold, CI and release tooling

### Fixed

- *(daemon)* keep the GPU off stage 1 on base-class chips
- *(daemon)* bound the model load and log where it is
- *(daemon)* keep a search's on-the-spot chunks away from the indexers
- never queue a chunk that got its vector after the barrier's read
- make a no-change barrier cost nothing per pending chunk

### Other

- *(release)* skip semver checks before 1.0
- *(release)* compute versions against crates.io
- Merge remote-tracking branch 'origin/dev' into fix/model-load-bound
- pin scheduler count snapshots to gate order
- *(daemon)* take a resync's missing set from the reconciler's books
- *(daemon)* count index --wait progress from the owner's books
- *(daemon)* answer vector search and status from the reconciler's books
- *(reconcile)* keep coverage bookkeeping between barriers
- *(daemon)* keep per-worktree queue counts instead of scanning
- Keep index --wait from blocking its worktree's other requests
- Keep Background-QoS daemon tests from starving on a loaded host
- Count committed chunk rows in the model test helper, as coverage does
- Flush the writer before killing the host in kill_and_resume_on_the_model
- End test daemon hosts when their parent test process dies
- Run the daemon under launchd socket activation
- Assemble the daemon: model loader, registry, worktree owners and watcher
- Schedule all embedding work through one actor
- Add the daemon protocol, its client and model download progress
- Batch Core ML predictions and watch GPU headroom
- Re-chunk an index built by a different chunker
- Cap chunks at 512 model tokens with signature-prefixed split parts
- size the release-gate timeout to the measured cold run
- run the coremltools build inside its venv
- install setuptools for the coremltools build
- seed the coremltools build venv
- run the embedder's model tests on PRs into main
- Merge origin/dev
- Add Core ML embedder with ANE and GPU instances
- Add verified first-use model download and model workflow
- run clippy once per feature with default features off
- Merge origin/dev
- Add Swift, TypeScript, JavaScript, Python, C, C++ and Go chunkers
- *(core)* drop the unused half dependency
- Merge origin/dev into feat/knn
- Merge origin/dev into feat/knn
- Merge origin/dev into feat/chunker
- run CI on the release PR and auto-merge it
- initial commit with MIT OR Apache-2.0 licenses
