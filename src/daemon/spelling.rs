//! A path's on-disk spelling: the name each component's parent directory
//! stores for it.
//!
//! The daemon keys a worktree's owner by its canonicalized root and the
//! repository's shared vector cache and writer by the canonicalized git
//! common dir. On Linux `fs::canonicalize` resolves symlinks but keeps
//! the spelling each component was reached with, so on a case-insensitive
//! mount — a virtiofs share of a case-insensitive APFS volume — one
//! directory has a canonical path for every case spelling it is requested
//! under, hence an owner, an index and a cache per spelling. macOS's
//! `realpath(3)` already returns the stored spelling, so [`canonical`] is
//! the identity there.

use std::ffi::{OsStr, OsString};
use std::fs;
use std::io;
use std::os::unix::fs::{DirEntryExt, MetadataExt};
use std::path::{Component, Path, PathBuf};

/// `path` canonicalized, then respelled to its on-disk spelling.
pub fn canonical(path: &Path) -> io::Result<PathBuf> {
    on_disk(&fs::canonicalize(path)?)
}

/// `path` with every component spelled as its parent directory stores it:
/// the parent's `read_dir` entry whose `(dev, ino)` equals the
/// component's — a directory has exactly one. The input is canonicalized,
/// so no component needs resolving; a symlink's own name is respelled
/// like any other entry's.
pub fn on_disk(path: &Path) -> io::Result<PathBuf> {
    let mut spelled = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(name) => spelled.push(stored(&spelled, name)?),
            _ => spelled.push(component.as_os_str()),
        }
    }
    Ok(spelled)
}

/// The name `parent` stores for the entry `name` resolves to. An entry
/// already named `name` is the stored name by definition — the fast path
/// that needs no inode scan; otherwise the one entry whose `(dev, ino)`
/// equals the component's supplies it, with the dirent's own inode the
/// filter before the metadata read that confirms the device too.
fn stored(parent: &Path, name: &OsStr) -> io::Result<OsString> {
    let dir = if parent.as_os_str().is_empty() {
        Path::new(".")
    } else {
        parent
    };
    let meta = fs::symlink_metadata(dir.join(name))?;
    let id = (meta.dev(), meta.ino());
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        if entry.file_name() == name {
            return Ok(entry.file_name());
        }
        if entry.ino() == id.1 {
            let meta = entry.metadata()?;
            if (meta.dev(), meta.ino()) == id {
                return Ok(entry.file_name());
            }
        }
    }
    Err(io::Error::new(
        io::ErrorKind::NotFound,
        format!(
            "{} lists no entry for {}",
            dir.display(),
            name.to_string_lossy()
        ),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A wrong-case spelling resolves to the stored one. Whether the
    /// volume is case-insensitive is detected on the tempdir itself: only
    /// there does the wrong spelling resolve at all, and two spellings of
    /// one directory exist only where it does.
    #[test]
    fn a_wrong_case_spelling_resolves_to_the_stored_one() {
        let tmp = tempfile::tempdir().expect("the tempdir is created");
        let stored = tmp.path().join("MixedCase").join("SubDir");
        fs::create_dir_all(&stored).expect("the directory is created");
        let wrong = tmp.path().join("mixedcase").join("subdir");
        if !wrong.exists() {
            // The volume is case-sensitive: the wrong spelling names
            // nothing, and the aliasing this fixes cannot arise.
            eprintln!("skipping: the tempdir's volume is case-sensitive");
            return;
        }
        assert_eq!(on_disk(&wrong).expect("the path respells"), stored);
    }
}
