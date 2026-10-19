# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.0](https://github.com/May-Agent/repo-search/releases/tag/core/v0.0.0) - 2026-10-19

### Added

- *(core)* incremental worktree reconcile with freshness report
- *(core)* int8 KNN with fp16 re-rank and flat-file vector cache
- *(core)* tree-sitter based Rust chunker
- *(core)* per-worktree index store and shared vector cache
- workspace scaffold, CI and release tooling

### Fixed

- *(core)* satisfy clippy 1.99 in the vector cache

### Other

- *(core)* answer reconcile coverage from the index without reading vectors
- *(core)* drop the unused half dependency
- Merge origin/dev into feat/knn
- Merge origin/dev into feat/knn
- Merge origin/dev into feat/chunker
