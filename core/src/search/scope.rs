//! Turning the user's filters into one [`ChunkFilter`] per result list.

use globset::{GlobBuilder, GlobSet, GlobSetBuilder};

use crate::chunk::Language;
use crate::store::{ChunkFilter, FileState, Side};

use super::{FilterError, Filters};

/// The syntactic node kinds a short `--kind` name stands for; a name not
/// listed is a node kind itself.
const KIND_ALIASES: &[(&str, &[&str])] = &[
    ("fn", &["function_item", "function_signature_item"]),
    ("struct", &["struct_item"]),
    ("enum", &["enum_item"]),
    ("trait", &["trait_item"]),
    ("impl", &["impl_item"]),
    ("mod", &["mod_item"]),
    ("macro", &["macro_definition", "macro_invocation"]),
    ("const", &["const_item"]),
    ("static", &["static_item"]),
    ("type", &["type_item", "associated_type"]),
    ("union", &["union_item"]),
];

/// The filter of each list that the user's filters leave in scope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Scope {
    /// The code list's filter; `None` when the filters exclude code.
    pub code: Option<ChunkFilter>,
    /// The docs list's filter; `None` when the filters exclude documentation.
    pub docs: Option<ChunkFilter>,
}

impl Scope {
    /// Resolves `filters` against the indexed `files` and the node `kinds`
    /// the index holds (`Index::kinds`).
    ///
    /// `--kind doc` selects documentation and a syntactic kind selects code;
    /// likewise a documentation language versus a code language. A list stays
    /// when every filter that speaks to the choice admits it, so `--kind fn
    /// --lang markdown` leaves neither. `--symbol` matches code symbols only.
    ///
    /// # Errors
    ///
    /// When a language is unknown, a kind is neither `doc`, a short name nor
    /// one of `kinds`, or a path glob does not compile.
    pub fn resolve(
        filters: &Filters,
        files: &[FileState],
        kinds: &[String],
    ) -> Result<Self, FilterError> {
        for kind in &filters.kinds {
            let known = is_doc_kind(kind)
                || KIND_ALIASES.iter().any(|(alias, _)| alias == kind)
                || kinds.contains(kind);
            if !known {
                return Err(FilterError::UnknownKind(kind.clone()));
            }
        }
        let languages = filters
            .langs
            .iter()
            .map(|name| {
                Language::of_name(name).ok_or_else(|| FilterError::UnknownLanguage(name.clone()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let globs = path_globs(&filters.paths)?;

        let wants_docs_kind = filters.kinds.iter().any(|k| is_doc_kind(k));
        let wants_code_kind = filters.kinds.iter().any(|k| !is_doc_kind(k));

        let admits = |side: Side| {
            let docs = side == Side::Docs;
            let by_kind = filters.kinds.is_empty()
                || if docs {
                    wants_docs_kind
                } else {
                    wants_code_kind
                };
            let by_lang = languages.is_empty() || languages.iter().any(|l| l.is_doc() == docs);
            let by_symbol = !docs || filters.symbol.is_none();
            by_kind && by_lang && by_symbol
        };

        let side_filter = |side: Side| {
            admits(side).then(|| {
                let docs = side == Side::Docs;
                let extensions = languages
                    .iter()
                    .filter(|l| l.is_doc() == docs)
                    .flat_map(|l| l.extensions().iter().copied())
                    .collect();
                ChunkFilter {
                    side,
                    paths: globs.as_ref().map(|globs| {
                        files
                            .iter()
                            .map(|f| f.path.as_str())
                            .filter(|p| globs.is_match(p))
                            .map(str::to_owned)
                            .collect()
                    }),
                    extensions,
                    kinds: if docs {
                        Vec::new()
                    } else {
                        code_kinds(&filters.kinds)
                    },
                    symbol: filters.symbol.clone(),
                }
            })
        };
        Ok(Self {
            code: side_filter(Side::Code),
            docs: side_filter(Side::Docs),
        })
    }
}

/// Whether `--kind` names the documentation list.
fn is_doc_kind(kind: &str) -> bool {
    Side::of_kind(kind) == Side::Docs
}

/// A glob also selects everything below it, so `core/src` means `core/src/**`.
/// `*` and `?` stay within one path component; only `**` crosses `/`.
fn path_globs(patterns: &[String]) -> Result<Option<GlobSet>, FilterError> {
    if patterns.is_empty() {
        return Ok(None);
    }
    let mut set = GlobSetBuilder::new();
    for pattern in patterns {
        let pattern = pattern.trim_end_matches('/');
        for glob in [pattern.to_owned(), format!("{pattern}/**")] {
            let glob = GlobBuilder::new(&glob)
                .literal_separator(true)
                .build()
                .map_err(|source| FilterError::BadGlob {
                    pattern: pattern.to_owned(),
                    source,
                })?;
            set.add(glob);
        }
    }
    Ok(Some(set.build().map_err(|source| {
        FilterError::BadGlob {
            pattern: patterns.join(", "),
            source,
        }
    })?))
}

/// The node kinds the code list's `--kind` names expand to.
fn code_kinds(kinds: &[String]) -> Vec<String> {
    kinds
        .iter()
        .filter(|k| !is_doc_kind(k))
        .flat_map(|k| {
            KIND_ALIASES
                .iter()
                .find(|(alias, _)| alias == k)
                .map_or_else(
                    || vec![k.clone()],
                    |(_, kinds)| kinds.iter().map(|s| (*s).to_owned()).collect(),
                )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::store::{ContentHash, FileState, Side};

    use super::{FilterError, Filters, Scope};

    fn files(paths: &[&str]) -> Vec<FileState> {
        paths
            .iter()
            .map(|p| FileState {
                path: (*p).to_owned(),
                size: 0,
                mtime_ns: 0,
                inode: 0,
                content_hash: ContentHash::of_file(b""),
            })
            .collect()
    }

    fn strings(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| (*s).to_owned()).collect()
    }

    /// The kinds the index of these tests holds.
    fn held() -> Vec<String> {
        strings(&["custom_kind", "function_item"])
    }

    #[test]
    fn no_filters_keep_both_lists_unrestricted() {
        let scope = Scope::resolve(&Filters::default(), &files(&["a.rs"]), &[]).unwrap();
        let code = scope.code.unwrap();
        assert_eq!(code.side, Side::Code);
        assert_eq!(code.paths, None);
        assert!(code.kinds.is_empty());
        assert_eq!(scope.docs.unwrap().side, Side::Docs);
    }

    #[test]
    fn a_syntactic_kind_drops_docs_and_expands_aliases() {
        let filters = Filters {
            kinds: strings(&["fn", "custom_kind"]),
            ..Filters::default()
        };
        let scope = Scope::resolve(&filters, &[], &held()).unwrap();
        assert!(scope.docs.is_none());
        assert_eq!(
            scope.code.unwrap().kinds,
            ["function_item", "function_signature_item", "custom_kind"]
        );
    }

    #[test]
    fn kind_doc_selects_docs_only() {
        let filters = Filters {
            kinds: strings(&["doc"]),
            ..Filters::default()
        };
        let scope = Scope::resolve(&filters, &[], &held()).unwrap();
        assert!(scope.code.is_none());
        assert!(scope.docs.unwrap().kinds.is_empty());
    }

    #[test]
    fn doc_and_syntactic_kinds_keep_both_lists() {
        let filters = Filters {
            kinds: strings(&["doc", "struct"]),
            ..Filters::default()
        };
        let scope = Scope::resolve(&filters, &[], &held()).unwrap();
        assert_eq!(scope.code.unwrap().kinds, ["struct_item"]);
        assert!(scope.docs.is_some());
    }

    #[test]
    fn a_documentation_language_selects_docs_and_its_extensions() {
        let filters = Filters {
            langs: strings(&["Markdown"]),
            ..Filters::default()
        };
        let scope = Scope::resolve(&filters, &[], &[]).unwrap();
        assert!(scope.code.is_none());
        let docs = scope.docs.unwrap();
        assert_eq!(docs.extensions, ["md", "markdown", "mdx"]);
        assert_eq!(docs.paths, None);
    }

    #[test]
    fn a_code_language_selects_code_and_its_extensions() {
        let filters = Filters {
            langs: strings(&["rust", "toml"]),
            ..Filters::default()
        };
        let scope = Scope::resolve(&filters, &[], &[]).unwrap();
        assert!(scope.docs.is_none());
        let code = scope.code.unwrap();
        assert_eq!(code.extensions, ["rs", "toml"]);
        assert_eq!(code.paths, None);
    }

    #[test]
    fn a_language_and_a_glob_restrict_both_ways() {
        let filters = Filters {
            langs: strings(&["rust"]),
            paths: strings(&["core/src"]),
            ..Filters::default()
        };
        let scope =
            Scope::resolve(&filters, &files(&["core/src/a.rs", "core/src/b.md"]), &[]).unwrap();
        let code = scope.code.unwrap();
        assert_eq!(code.extensions, ["rs"]);
        assert_eq!(code.paths.unwrap(), ["core/src/a.rs", "core/src/b.md"]);
    }

    #[test]
    fn conflicting_kind_and_language_leave_nothing() {
        let filters = Filters {
            kinds: strings(&["fn"]),
            langs: strings(&["markdown"]),
            ..Filters::default()
        };
        let scope = Scope::resolve(&filters, &[], &held()).unwrap();
        assert!(scope.code.is_none() && scope.docs.is_none());
    }

    #[test]
    fn symbol_filter_drops_docs() {
        let filters = Filters {
            symbol: Some("Environment::*".into()),
            ..Filters::default()
        };
        let scope = Scope::resolve(&filters, &[], &held()).unwrap();
        assert!(scope.docs.is_none());
        assert_eq!(
            scope.code.unwrap().symbol.as_deref(),
            Some("Environment::*")
        );
    }

    #[test]
    fn path_globs_match_files_and_directories() {
        let filters = Filters {
            paths: strings(&["core/src", "**/*.md"]),
            ..Filters::default()
        };
        let all = files(&[
            "core/src/a.rs",
            "core/tests/b.rs",
            "docs/c.md",
            "core/src/x/d.rs",
        ]);
        let scope = Scope::resolve(&filters, &all, &[]).unwrap();
        assert_eq!(
            scope.code.unwrap().paths.unwrap(),
            ["core/src/a.rs", "docs/c.md", "core/src/x/d.rs"]
        );
    }

    #[test]
    fn a_star_stays_within_one_path_component() {
        let filters = Filters {
            paths: strings(&["core/*.rs", "docs/**/*.md"]),
            ..Filters::default()
        };
        let all = files(&[
            "core/a.rs",
            "core/src/b.rs",
            "docs/c.md",
            "docs/x/y/d.md",
            "other/e.rs",
        ]);
        let scope = Scope::resolve(&filters, &all, &[]).unwrap();
        assert_eq!(
            scope.code.unwrap().paths.unwrap(),
            ["core/a.rs", "docs/c.md", "docs/x/y/d.md"]
        );
    }

    #[test]
    fn an_unknown_kind_is_an_error_and_a_known_one_without_chunks_is_not() {
        let unknown = Filters {
            kinds: strings(&["fn", "nonsense_kind"]),
            ..Filters::default()
        };
        let err = Scope::resolve(&unknown, &[], &held()).unwrap_err();
        assert!(
            matches!(&err, FilterError::UnknownKind(k) if k == "nonsense_kind"),
            "{err}"
        );

        // `union` is a short name and `custom_kind` is held: both resolve.
        let known = Filters {
            kinds: strings(&["union", "custom_kind", "doc"]),
            ..Filters::default()
        };
        assert!(Scope::resolve(&known, &[], &held()).is_ok());
    }

    #[test]
    fn unknown_language_and_bad_glob_are_errors() {
        let bad_lang = Filters {
            langs: strings(&["cobol"]),
            ..Filters::default()
        };
        assert!(Scope::resolve(&bad_lang, &[], &[]).is_err());
        let bad_glob = Filters {
            paths: strings(&["[a"]),
            ..Filters::default()
        };
        assert!(Scope::resolve(&bad_glob, &[], &[]).is_err());
    }
}
