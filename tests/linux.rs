//! What only Linux does: inotify watching, and the real `search` binary
//! spawning its own daemon from the XDG directories.
//!
//! The inotify cases drive `daemon::watch` and a `Bound` daemon against
//! real filesystem events. The binary cases run `CARGO_BIN_EXE_search`
//! with every `XDG_*` variable pointed at a tempdir and a `daemon.toml`
//! naming the fake standard-API server; `search` and `search index` need
//! the daemon's real tokenizer, which the daemon downloads from Hugging
//! Face into the tempdir's cache, so those two need network access.

#![cfg(target_os = "linux")]

#[allow(dead_code)] // each test file uses its own slice of the fixtures
mod support;

use std::fs;
use std::os::unix::fs::PermissionsExt as _;
use std::path::{Path, PathBuf};
use std::process::Output;
use std::time::Duration;

use repo_search::daemon::server::Endpoint;
use repo_search::daemon::watch::{self, WatchMsg, Watcher};
use repo_search_client::protocol::{Final, Request};
use repo_search_core::StateDir;
use repo_search_core::model::MODEL_ID;
use support::bound::spawn;
use support::client::{final_of, hello, request};
use support::embed_server::{FakeConfig, FakeServer};
use support::lock;
use support::repo::{self, TestRepo};
use support::tokens::StubTokens;

/// A bound on every wait for an event or a process.
const TIMEOUT: Duration = Duration::from_secs(120);

/// Long enough for the watcher's debounce (`QUIET`, half a second) to
/// have flushed anything the filesystem events signalled.
const SETTLE: Duration = Duration::from_millis(2500);

/// The marker an edit lands in the index.
const MARKER: &str = "xylophonemarker";

/// A watched worktree root in a tempdir, with a few files.
fn tree() -> (tempfile::TempDir, PathBuf) {
    let tmp = tempfile::tempdir().expect("tmp");
    let root = tmp.path().join("worktree");
    fs::create_dir_all(root.join("src")).expect("dirs");
    fs::create_dir(root.join(".git")).expect("git dir");
    fs::write(root.join("src/a.rs"), "fn a() {}\n").expect("file");
    fs::write(root.join("b.rs"), "fn b() {}\n").expect("file");
    (tmp, root)
}

fn watch_root(root: &Path) -> Watcher {
    watch::start(root, &root.join(".git")).expect("the watcher starts")
}

/// The next mark within `TIMEOUT`.
async fn next(watcher: &mut Watcher) -> WatchMsg {
    tokio::time::timeout(TIMEOUT, watcher.changed.recv())
        .await
        .expect("a mark arrives")
        .expect("the watcher lives")
}

/// Marks until a `Gone`, which may follow a `Changed` for the removed
/// files' own events.
async fn until_gone(watcher: &mut Watcher) {
    while next(watcher).await != WatchMsg::Gone {}
}

/// An edit is a debounced `Changed`.
#[tokio::test]
async fn an_edit_signals_changed() {
    let (_tmp, root) = tree();
    let mut watcher = watch_root(&root);
    fs::write(root.join("src/a.rs"), "fn a() { 1 }\n").expect("edit");
    assert_eq!(next(&mut watcher).await, WatchMsg::Changed);
}

/// `rm -rf` of the root is `Gone`.
#[tokio::test]
async fn removing_the_root_signals_gone() {
    let (_tmp, root) = tree();
    let mut watcher = watch_root(&root);
    fs::remove_dir_all(&root).expect("remove");
    until_gone(&mut watcher).await;
}

/// Renaming the root away is `Gone` too: inotify's `MOVE_SELF` names the
/// root, whose path no longer exists.
#[tokio::test]
async fn renaming_the_root_signals_gone() {
    let (tmp, root) = tree();
    let mut watcher = watch_root(&root);
    fs::rename(&root, tmp.path().join("moved")).expect("rename");
    until_gone(&mut watcher).await;
}

/// The reconciler's reads — opens, reads, closes, directory listings —
/// are inotify events too, and none of them may signal: that would be a
/// reindex loop. A write after the storm proves the watcher still sees
/// changes.
#[tokio::test]
async fn reads_signal_nothing() {
    let (_tmp, root) = tree();
    let mut watcher = watch_root(&root);
    for _ in 0..20 {
        for entry in walk(&root) {
            let _ = fs::read(&entry);
        }
    }
    assert!(
        tokio::time::timeout(SETTLE, watcher.changed.recv())
            .await
            .is_err(),
        "reads produced a mark"
    );
    fs::write(root.join("b.rs"), "fn b() { 1 }\n").expect("edit");
    assert_eq!(next(&mut watcher).await, WatchMsg::Changed);
}

/// Every path under `dir`, listing directories as it goes.
fn walk(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for entry in fs::read_dir(dir).expect("read_dir") {
        let path = entry.expect("entry").path();
        if path.is_dir() {
            files.extend(walk(&path));
        } else {
            files.push(path);
        }
    }
    files
}

/// An edit is embedded in the background: after `index --wait`, an
/// appended function reaches the embedding server with no query asking
/// for it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_edit_is_embedded_in_the_background() {
    let repo = repo::repo(&[("src/f.rs".to_owned(), repo::rust_fn("fn_0", 500))]);
    let tmp = tempfile::tempdir().expect("tmp");
    let fake = FakeServer::unix(
        &tmp.path().join("server.sock"),
        FakeConfig {
            model_id: MODEL_ID.to_owned(),
            token: None,
        },
    );
    let socket = tmp.path().join("daemon.sock");
    let _daemon = spawn(
        &socket,
        StateDir::new(tmp.path().join("state")),
        Endpoint::Unix(fake.path().to_path_buf()),
        StubTokens::default(),
        Duration::from_secs(600),
    )
    .await;
    let events = request(
        &socket,
        &hello(),
        Request::Index {
            worktree: repo.root.clone(),
            wait: true,
        },
    )
    .await;
    assert!(matches!(final_of(&events), Final::Indexed { .. }));
    drop(fake.recorded());

    repo::append(
        &repo.root,
        "src/edit.rs",
        &repo::rust_fn(&format!("{MARKER}_fn"), 500),
    );
    loop {
        let recorded = fake.next();
        if recorded.path == "/v1/embeddings" && recorded.body.to_string().contains(MARKER) {
            break;
        }
    }
}

/// The installed `search` of a user whose every XDG directory is a
/// tempdir and whose `daemon.toml` names the fake server.
struct Installed {
    tmp: tempfile::TempDir,
    repo: TestRepo,
    _fake: FakeServer,
    /// Kills the daemon a failing test leaves running.
    _reaper: lock::Reaper,
}

impl Installed {
    fn new() -> Self {
        let repo = repo::repo(&[("src/f.rs".to_owned(), repo::rust_fn("fn_0", 500))]);
        let tmp = tempfile::tempdir().expect("tmp");
        for dir in ["home", "state", "cache", "config/repo-search", "runtime"] {
            fs::create_dir_all(tmp.path().join(dir)).expect("dirs");
        }
        fs::set_permissions(
            tmp.path().join("runtime"),
            fs::Permissions::from_mode(0o700),
        )
        .expect("runtime dir mode");
        let fake = FakeServer::unix(
            &tmp.path().join("server.sock"),
            FakeConfig {
                model_id: MODEL_ID.to_owned(),
                token: None,
            },
        );
        fs::write(
            tmp.path().join("config/repo-search/daemon.toml"),
            format!("server_socket = {:?}\n", fake.path().display().to_string()),
        )
        .expect("daemon.toml");
        Self {
            _reaper: lock::Reaper(tmp.path().join("runtime/repo-search")),
            tmp,
            repo,
            _fake: fake,
        }
    }

    /// Runs `search args...` in the repository.
    async fn search(&self, args: &[&str]) -> Output {
        let home = |dir: &str| self.tmp.path().join(dir);
        let mut command = tokio::process::Command::new(env!("CARGO_BIN_EXE_search"));
        command
            .args(args)
            .current_dir(&self.repo.root)
            .env_clear()
            .env("PATH", std::env::var_os("PATH").expect("PATH"))
            .env("HOME", home("home"))
            .env("XDG_STATE_HOME", home("state"))
            .env("XDG_CACHE_HOME", home("cache"))
            .env("XDG_CONFIG_HOME", home("config"))
            .env("XDG_RUNTIME_DIR", home("runtime"))
            .kill_on_drop(true);
        tokio::time::timeout(TIMEOUT, command.output())
            .await
            .expect("search finishes")
            .expect("search runs")
    }

    fn run_dir(&self) -> PathBuf {
        self.tmp.path().join("runtime/repo-search")
    }
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// `search status` spawns the real daemon through `main.rs` and is
/// answered by it; the daemon outlives the client, and its log is in the
/// XDG state dir.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn search_status_spawns_the_real_daemon() {
    let installed = Installed::new();
    let output = installed.search(&["status", "--json"]).await;
    assert!(output.status.success(), "{}", text(&output.stderr));
    let status: serde_json::Value = serde_json::from_slice(&output.stdout).expect("status is JSON");
    let pid = lock::alive(&installed.run_dir()).expect("the daemon outlives the client");
    assert_eq!(status["daemon"]["pid"].as_u64(), Some(u64::from(pid)));
    assert!(
        installed
            .tmp
            .path()
            .join("state/repo-search/daemon.log")
            .exists()
    );
    assert!(installed.run_dir().join("daemon.sock").exists());
    assert!(installed.run_dir().join("daemon.cmd").exists());

    // The warm second search is answered by the same daemon.
    let again = installed.search(&["status", "--json"]).await;
    let again: serde_json::Value = serde_json::from_slice(&again.stdout).expect("JSON");
    assert_eq!(again["daemon"]["pid"], status["daemon"]["pid"]);
}

/// `search index` and `search <query>` through the real binary, daemon
/// and tokenizer: the index covers the repository and the query finds an
/// edit made after it. Downloads the pinned tokenizer.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn search_and_index_through_the_real_binary() {
    let installed = Installed::new();
    let indexed = installed.search(&["index", "--wait"]).await;
    assert!(indexed.status.success(), "{}", text(&indexed.stderr));
    assert!(
        text(&indexed.stdout).starts_with("indexed "),
        "{}",
        text(&indexed.stdout)
    );

    repo::append(
        &installed.repo.root,
        "src/edit.rs",
        &repo::rust_fn(&format!("{MARKER}_fn"), 500),
    );
    let found = installed.search(&[MARKER]).await;
    assert!(found.status.success(), "{}", text(&found.stderr));
    assert!(
        text(&found.stdout).contains(MARKER),
        "{}",
        text(&found.stdout)
    );
}
