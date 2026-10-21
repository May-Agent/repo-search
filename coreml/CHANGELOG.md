# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.2](https://github.com/May-Agent/repo-search/compare/coreml/v0.0.1...coreml/v0.0.2) - 2026-10-21

### Other

- updated the following local packages: repo-search-core

## [0.0.1](https://github.com/May-Agent/repo-search/compare/coreml/v0.0.0...coreml/v0.0.1) - 2026-10-21

### Added

- workspace scaffold, CI and release tooling

### Other

- Schedule all embedding work through one actor
- Batch Core ML predictions and watch GPU headroom
- Share one embedding service between search and indexing
- Estimate the cap's cut from one encode before probing it
- Cap chunks at 512 model tokens with signature-prefixed split parts
- Add Core ML embedder with ANE and GPU instances
