# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.3](https://github.com/May-Agent/repo-search/compare/server/v0.0.2...server/v0.0.3) - 2026-10-29

### Other

- updated the following local packages: repo-search-core, repo-search-coreml

## [0.0.2](https://github.com/May-Agent/repo-search/releases/tag/server/v0.0.2) - 2026-10-29

### Added

- *(daemon)* embed through the server over HTTP
- *(server)* standard embeddings API and search --server
- *(server)* embedding server crate with the hardware scheduler

### Fixed

- *(deps)* move tokio and pulldown-cmark to their patched releases
- *(server)* keep the headroom monitor for the server's lifetime

### Other

- Merge remote-tracking branch 'origin/dev' into feat/linux-daemon
- Merge remote-tracking branch 'origin/dev' into feat/linux-daemon
- size thread pools from the CPU count
- *(server)* wait for the urgent item to queue before opening the gate
- drop the in-process embedder API
