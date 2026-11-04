# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.7](https://github.com/May-Agent/repo-search/compare/client/v0.0.6...client/v0.0.7) - 2026-11-04

### Fixed

- do not install production launchd labels under a relocated HOME

## [0.0.6](https://github.com/May-Agent/repo-search/compare/client/v0.0.5...client/v0.0.6) - 2026-11-02

### Fixed

- *(server)* run the agent under a per-build name for its own e5rt cache
- handshake compares PROTOCOL only

### Other

- Merge pull request #185 from May-Agent/fix/181-per-build-ane-cache

## [0.0.5](https://github.com/May-Agent/repo-search/compare/client/v0.0.4...client/v0.0.5) - 2026-11-01

### Added

- *(reconcile)* index only git-tracked files

## [0.0.4](https://github.com/May-Agent/repo-search/compare/client/v0.0.3...client/v0.0.4) - 2026-10-29

### Added

- *(client)* move the daemon launcher into the client crate

## [0.0.3](https://github.com/May-Agent/repo-search/releases/tag/client/v0.0.3) - 2026-10-29

### Added

- client crate without the core dependency
