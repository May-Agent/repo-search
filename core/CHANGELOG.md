# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.0](https://github.com/May-Agent/repo-search/releases/tag/core/v0.0.0) - 2026-10-21

### Added

- the search CLI client
- *(core)* retrieval, fusion and output
- *(core)* documentation and configuration chunkers behind a file-level chunker trait
- *(core)* incremental worktree reconcile with freshness report
- *(core)* int8 KNN with fp16 re-rank and flat-file vector cache
- *(core)* tree-sitter based Rust chunker
- *(core)* per-worktree index store and shared vector cache
- workspace scaffold, CI and release tooling

### Fixed

- make a no-change barrier cost nothing per pending chunk
- *(core)* budget selection takes one chunk per file per pass
- *(core)* satisfy clippy 1.99 in the vector cache

### Other

- Merge pull request #57 from May-Agent/fix/stale-multi-member
- Rebuild chunk text the way it was written when checking for staleness
- Keep Background-QoS daemon tests from starving on a loaded host
- Assemble the daemon: model loader, registry, worktree owners and watcher
- Schedule all embedding work through one actor
- Merge remote-tracking branch 'origin/dev' into feat/daemon
- Keep ignored files indexed across searches with and without --include-ignored
- Share one embedding service between search and indexing
- Merge origin/dev into feat/record-unchunkable
- Record files that fail to chunk instead of re-parsing them every pass
- Re-chunk an index built by a different chunker
- Gate the cap test file on the JavaScript chunker
- Estimate the cap's cut from one encode before probing it
- Cap chunks at 512 model tokens with signature-prefixed split parts
- Merge origin/dev
- Add Swift, TypeScript, JavaScript, Python, C, C++ and Go chunkers
- Merge origin/dev into feat/doc-config-chunkers
- *(core)* answer reconcile coverage from the index without reading vectors
- *(core)* drop the unused half dependency
- Merge origin/dev into feat/knn
- Merge origin/dev into feat/knn
- Merge origin/dev into feat/chunker
