# repo-search

**Search a repository for code you can describe but cannot name.**

`repo-search` is a local, hardware-aware hybrid code search engine and `search` CLI designed for both human developers and coding agents (such as [May](https://github.com/May-Agent/may)). It combines AST-aware Tree-sitter chunking, SQLite FTS5 lexical BM25, and on-device dense vector retrieval merged via Reciprocal Rank Fusion (RRF)—returning token-budgeted results without sending source code off the machine.

## Highlights

- **Apple Neural Engine (ANE) & GPU Co-Scheduling (`repo-search-coreml`)** — On Apple Silicon, a compiled multi-function `.mlmodelc` bundle shares weights across sequence-length buckets (`len_128`, `len_256`, `len_512`) so short code chunks never pay for 512-token padding:
  - **`CpuAndNeuralEngine` (ANE)** serves interactive queries and low-power background indexing.
  - **`CpuAndGpu`** accelerates bulk stage-1 indexing only when the host's `ChipClass` (Pro/Max/Ultra vs. base M-series) and real-time thermal/power `gpu_headroom` permit.
- **Cross-Platform ONNX Runtime Backend (`repo-search-ort`)** — Portable local embedding execution on Linux (`x86_64` and `aarch64`).
- **AST-Aware Tree-Sitter Chunking (`repo-search-core`)** — Parses source files into syntactic items (`fn`, `struct`, `trait`, `impl`, `class`, etc.) stamped with a deterministic 32-hex `CHUNKER_FINGERPRINT` derived from the chunker sources and grammar versions.
- **Hybrid Retrieval + Token Budgeting** — Merges FTS5 BM25 lexical matches and vector KNN via Reciprocal Rank Fusion (`--mode hybrid`), packing ranked snippets until the caller's token budget (`--budget`, default `4000`) is filled.
- **Zero-Config Socket-Activated Daemon** — The `search` binary acts as a thin, current-thread client that auto-spawns or socket-activates the background `launchd` (macOS) / on-demand (Linux) daemon, reconciles worktrees live via `FSEvents` / `inotify`, automatically replaces the daemon when its `BuildId` changes, and exits cleanly after idle timeout.

## Usage

```bash
# Hybrid semantic + lexical search (default budget: 4000 tokens)
search "where are capability handles narrowed on derive"

# Ranked file paths only
search -l "terminal IPC command registration"

# Filter by language, AST node kind, symbol pattern, or path prefix
search "connection pool backoff" --lang rust --kind fn,struct --path src/
search "retry policy" --symbol "Client::*" --budget 2000

# Choose retrieval channel explicitly (hybrid | lexical | semantic)
search "zero-copy texture import" --mode semantic

# Check index coverage, pending chunks, model, GPU headroom, and daemon state
search status

# Force an immediate worktree reconciliation and wait until every chunk is embedded
search index --wait
```

Exit codes follow `grep`: `0` when results are returned, `1` when none match, and `2` on usage or runtime error.

## Workspace Layout

| Crate | Role |
| --- | --- |
| `repo-search` (`src/`) | `search` CLI client, `launchd` socket activation, first-use model store, and daemon entrypoint |
| `core/` (`repo-search-core`) | Tree-sitter AST chunker, SQLite FTS5 + vector store, worktree reconciler, and RRF hybrid search |
| `coreml/` (`repo-search-coreml`) | Objective-C / Core ML bindings (`objc2-core-ml`) for multi-bucket ANE & GPU embedding on Apple Silicon |
| `ort/` (`repo-search-ort`) | ONNX Runtime embedding backend for Linux |
| `server/` (`repo-search-server`) | Local OpenAI-compatible embedding server with `ChipClass` detection, ANE/GPU scheduling, and idle exit |
| `client/` (`repo-search-client`) | Wire protocol, `BuildId` handshake, and daemon launcher traits |

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT) at your option.
