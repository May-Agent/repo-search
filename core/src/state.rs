//! The daemon's per-user state directory: every file the daemon persists
//! lives under it — the per-worktree index and the per-repository vector
//! cache — keyed by a fingerprint of the canonical path each belongs to,
//! so the daemon never writes under a worktree.
//!
//! The root is passed explicitly from the daemon's configuration down to
//! every component that opens or stats a stored file; nothing below the
//! daemon's own startup looks it up.

use std::path::{Path, PathBuf};

/// The daemon's per-user state root.
///
/// `worktrees/<fingerprint>/index.db` holds each worktree's index and
/// `repos/<fingerprint>/` each repository's shared vector cache, where
/// `fingerprint` is the blake3 hash — hex — of the canonical worktree
/// root or git common dir. A document collection lives whole under
/// `collections/<fingerprint>/`, the fingerprint being that of its name.
#[derive(Clone, Debug)]
pub struct StateDir(PathBuf);

impl StateDir {
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

    /// The directory of the document collection `name`:
    /// `<root>/collections/<fingerprint>`. Any string names a collection;
    /// the hash keeps the name out of the layout.
    pub fn collection(&self, name: &str) -> PathBuf {
        self.0
            .join("collections")
            .join(fingerprint(Path::new(name)))
    }

    /// The directory holding the collection's documents as files, the
    /// root its index and retriever treat as a worktree.
    pub fn collection_docs(&self, name: &str) -> PathBuf {
        self.collection(name).join("docs")
    }

    /// The collection's index database.
    pub fn collection_index(&self, name: &str) -> PathBuf {
        self.collection(name).join("index.db")
    }

    /// The collection's own vector cache directory; collections never
    /// share vectors with each other or with a repository.
    pub fn collection_cache(&self, name: &str) -> PathBuf {
        self.collection(name).join("vectors")
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
