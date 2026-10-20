# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.0](https://github.com/May-Agent/repo-search/releases/tag/v0.0.0) - 2026-10-20

### Added

- *(convert)* three-bucket model (128, 256, 512)
- *(core)* retrieval, fusion and output
- *(core)* documentation and configuration chunkers behind a file-level chunker trait
- *(convert)* Core ML conversion pipeline for EmbeddingGemma 2
- *(core)* incremental worktree reconcile with freshness report
- *(core)* int8 KNN with fp16 re-rank and flat-file vector cache
- *(core)* tree-sitter based Rust chunker
- *(core)* per-worktree index store and shared vector cache
- workspace scaffold, CI and release tooling

### Other

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
