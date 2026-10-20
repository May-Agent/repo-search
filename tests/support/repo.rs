//! A temporary git repository for the daemon tests.

use std::path::{Path, PathBuf};
use std::process::Command;

/// A git repository in a temp dir.
pub struct TestRepo {
    /// The worktree root.
    pub root: PathBuf,
    /// The git common dir (`<root>/.git`).
    pub common_dir: PathBuf,
    _tmp: tempfile::TempDir,
}

/// Creates a repository with `files` as `(path, contents)` and commits it.
pub fn repo(files: &[(String, String)]) -> TestRepo {
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path().join("repo");
    std::fs::create_dir_all(&root).expect("repo root");
    git(&root, &["init", "-q", "-b", "main"]);
    for (path, contents) in files {
        let path = root.join(path);
        std::fs::create_dir_all(path.parent().expect("file has a parent")).expect("dirs");
        std::fs::write(path, contents).expect("file write");
    }
    git(&root, &["add", "-A"]);
    git(
        &root,
        &[
            "-c",
            "user.email=tests@repo-search",
            "-c",
            "user.name=repo-search tests",
            "commit",
            "-q",
            "-m",
            "initial",
        ],
    );
    TestRepo {
        common_dir: root.join(".git"),
        root,
        _tmp: tmp,
    }
}

/// Appends `contents` to `root/path`, creating the file when missing.
pub fn append(root: &Path, path: &str, contents: &str) {
    use std::io::Write as _;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(root.join(path))
        .expect("append open");
    file.write_all(contents.as_bytes()).expect("append");
}

/// A Rust function whose chunk lands near `text_bytes` bytes of text under
/// `StubCounter`. `name` must be a unique identifier: it lands in the
/// chunk's title and text.
///
/// A function this size never packs with neighbours (the pack threshold is
/// 400 bytes), so each becomes a chunk of its own whose token count is
/// `(title + text) / 4 + 1`.
pub fn rust_fn(name: &str, text_bytes: usize) -> String {
    let head = format!("pub fn {name}() -> usize {{\n    \"");
    let tail = "\".len()\n}\n";
    let body = "x".repeat(text_bytes.saturating_sub(head.len() + tail.len()));
    format!("{head}{body}{tail}")
}

/// Adds a linked worktree named `name` next to `repo` on a new `branch`,
/// returning its path.
pub fn add_worktree(repo: &TestRepo, name: &str, branch: &str) -> PathBuf {
    let path = repo
        .root
        .parent()
        .expect("the repo root has a parent")
        .join(name);
    git(
        &repo.root,
        &[
            "worktree",
            "add",
            "-q",
            "-b",
            branch,
            path.to_str().expect("utf-8 path"),
        ],
    );
    path
}

/// Removes the linked worktree at `path`, directory included.
pub fn remove_worktree(repo: &TestRepo, path: &Path) {
    git(
        &repo.root,
        &[
            "worktree",
            "remove",
            "--force",
            path.to_str().expect("utf-8 path"),
        ],
    );
}

fn git(root: &Path, args: &[&str]) {
    let status = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .status()
        .expect("git runs");
    assert!(status.success(), "git {args:?} failed: {status}");
}
