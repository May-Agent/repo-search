//! The daemon's per-user state directory: every file the daemon persists
//! lives under it — the per-worktree index and the per-repository vector
//! cache — keyed by a fingerprint of the canonical path each belongs to,
//! so the daemon never writes under a worktree.
//!
//! The root is passed explicitly from the daemon's configuration down to
//! every component that opens or stats a stored file; nothing below the
//! daemon's own startup looks it up.

#[cfg(target_os = "macos")]
use std::io;
use std::path::{Path, PathBuf};

/// The daemon's per-user state root.
///
/// `worktrees/<fingerprint>/index.db` holds each worktree's index and
/// `repos/<fingerprint>/` each repository's shared vector cache, where
/// `fingerprint` is the blake3 hash — hex — of the canonical worktree
/// root or git common dir.
#[derive(Clone, Debug)]
pub struct StateDir(PathBuf);

impl StateDir {
    /// The per-user state root on macOS:
    /// `~/Library/Application Support/repo-search`.
    ///
    /// # Errors
    ///
    /// When the home directory is unknown.
    #[cfg(target_os = "macos")]
    pub fn user() -> io::Result<Self> {
        let home = std::env::home_dir()
            .ok_or_else(|| io::Error::other("the home directory is unknown"))?;
        Ok(Self(home.join("Library/Application Support/repo-search")))
    }

    /// A state dir rooted at `root`; tests pass a tempdir.
    pub fn new(root: PathBuf) -> Self {
        Self(root)
    }

    /// The state root itself.
    pub fn root(&self) -> &Path {
        &self.0
    }

    /// `worktree`'s index database:
    /// `<root>/worktrees/<fingerprint>/index.db`.
    ///
    /// `worktree` is its canonical root — the daemon resolves it — so two
    /// worktrees at different paths get separate indexes and the key is
    /// stable across daemon restarts.
    pub fn worktree_index(&self, worktree: &Path) -> PathBuf {
        self.0
            .join("worktrees")
            .join(fingerprint(worktree))
            .join("index.db")
    }

    /// The repository's shared cache directory: `<root>/repos/
    /// <fingerprint>`.
    ///
    /// `common_dir` is the canonical git common dir, so every worktree of
    /// one repository shares the directory and two repositories never
    /// touch the same cache.
    pub fn repo_cache(&self, common_dir: &Path) -> PathBuf {
        self.0.join("repos").join(fingerprint(common_dir))
    }
}

/// `path`'s key under the state dir: the blake3 of its bytes, hex. The
/// hash is the whole directory name, so the path itself is never encoded
/// in the layout and collisions are not a concern.
fn fingerprint(path: &Path) -> String {
    blake3::hash(path.as_os_str().as_encoded_bytes())
        .to_hex()
        .to_string()
}
