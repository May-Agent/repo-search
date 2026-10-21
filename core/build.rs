//! Computes the chunker fingerprint: a hash of the chunker sources and of the
//! tree-sitter grammar versions, so an index can tell it was chunked by a
//! different chunker. The grammars are pinned exactly in this crate's manifest,
//! which is the one file that travels with the crate wherever it is built.

use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use toml_edit::DocumentMut;

/// Bytes of the digest kept; the hex form is twice this long.
const FINGERPRINT_BYTES: usize = 16;

fn main() {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("cargo sets it"));
    let cargo_toml = manifest.join("Cargo.toml");
    println!("cargo:rerun-if-changed=src/chunk");
    println!("cargo:rerun-if-changed=Cargo.toml");

    let mut hasher = Sha256::new();
    let mut sources = Vec::new();
    collect_files(&manifest.join("src/chunk"), &mut sources);
    sources.sort();
    for path in &sources {
        let rel = path
            .strip_prefix(&manifest)
            .expect("collected under the manifest dir");
        let bytes = fs::read(path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
        hasher.update(rel.to_string_lossy().as_bytes());
        hasher.update([0]);
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(&bytes);
    }

    let text = fs::read_to_string(&cargo_toml)
        .unwrap_or_else(|e| panic!("reading {}: {e}", cargo_toml.display()));
    let doc: DocumentMut = text
        .parse()
        .unwrap_or_else(|e| panic!("{} is not valid TOML: {e}", cargo_toml.display()));
    let dependencies = doc["dependencies"]
        .as_table()
        .expect("the manifest has a [dependencies] table");
    let mut grammars: Vec<(&str, &str)> = dependencies
        .iter()
        .filter(|(name, _)| *name == "tree-sitter" || name.starts_with("tree-sitter-"))
        .map(|(name, item)| {
            let requirement = item["version"]
                .as_str()
                .unwrap_or_else(|| panic!("{name} has no version requirement"));
            let version = requirement.strip_prefix('=').unwrap_or_else(|| {
                panic!("{name} must be pinned exactly (`={requirement}`) for the fingerprint")
            });
            (name, version)
        })
        .collect();
    grammars.sort_unstable();
    for (name, version) in grammars {
        hasher.update(name.as_bytes());
        hasher.update([0]);
        hasher.update(version.as_bytes());
        hasher.update([0]);
    }

    let digest = hasher.finalize();
    let mut hex = String::with_capacity(FINGERPRINT_BYTES * 2);
    for byte in &digest[..FINGERPRINT_BYTES] {
        write!(hex, "{byte:02x}").expect("writing to a String");
    }
    println!("cargo:rustc-env=REPO_SEARCH_CHUNKER_FINGERPRINT={hex}");
}

/// Appends every file under `dir`, recursively.
fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(dir).unwrap_or_else(|e| panic!("reading {}: {e}", dir.display()));
    for entry in entries {
        let path = entry.expect("directory entry").path();
        if path.is_dir() {
            collect_files(&path, out);
        } else {
            out.push(path);
        }
    }
}
