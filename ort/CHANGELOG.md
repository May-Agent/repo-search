# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.7](https://github.com/May-Agent/repo-search/compare/repo-search-ort-v0.0.6...repo-search-ort-v0.0.7) - 2026-11-04

### Fixed

- *(ort)* share one tokenizer across requests ([#194](https://github.com/May-Agent/repo-search/pull/194))
- *(ort)* batch same-bucket inputs across requests ([#194](https://github.com/May-Agent/repo-search/pull/194))
- *(ort)* fail loudly when an execution provider cannot register and serve every device

## [0.0.6](https://github.com/May-Agent/repo-search/compare/repo-search-ort-v0.0.5...repo-search-ort-v0.0.6) - 2026-11-01

### Other

- updated the following local packages: repo-search-core

## [0.0.5](https://github.com/May-Agent/repo-search/compare/repo-search-ort-v0.0.4...repo-search-ort-v0.0.5) - 2026-10-31

### Added

- CUDA and TensorRT execution providers for the ONNX server

## [0.0.4](https://github.com/May-Agent/repo-search/releases/tag/repo-search-ort-v0.0.4) - 2026-10-30

### Added

- CPU never shares an execution-provider chain with accelerators
- packaged ONNX model, auto-download, and EP selection on non-macOS
- ONNX embedding server for Linux
