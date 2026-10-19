//! Language detection by file extension.

use std::path::Path;

/// A documentation markup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocFormat {
    /// Markdown and MDX.
    Markdown,
    /// reStructuredText.
    Rst,
    /// AsciiDoc.
    AsciiDoc,
    /// Plain text: one section, split on paragraphs.
    Text,
}

/// A configuration syntax.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigFormat {
    /// TOML.
    Toml,
    /// YAML.
    Yaml,
    /// JSON, including comments and trailing commas.
    Json,
}

/// What a file is, decided by its extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    /// Rust source.
    Rust,
    /// Documentation.
    Doc(DocFormat),
    /// Configuration.
    Config(ConfigFormat),
}

impl Language {
    /// The language of the file at `path`, or `None` for an extension the
    /// chunkers do not know; such files are never read.
    #[must_use]
    pub fn of_path(path: &Path) -> Option<Self> {
        let extension = path.extension()?.to_str()?;
        const EXTENSIONS: &[(&[&str], Language)] = &[
            (&["rs"], Language::Rust),
            (
                &["md", "markdown", "mdx"],
                Language::Doc(DocFormat::Markdown),
            ),
            (&["rst"], Language::Doc(DocFormat::Rst)),
            (&["adoc", "asciidoc"], Language::Doc(DocFormat::AsciiDoc)),
            (&["txt"], Language::Doc(DocFormat::Text)),
            (&["toml"], Language::Config(ConfigFormat::Toml)),
            (&["yaml", "yml"], Language::Config(ConfigFormat::Yaml)),
            (&["json", "jsonc"], Language::Config(ConfigFormat::Json)),
        ];
        EXTENSIONS
            .iter()
            .find(|(names, _)| {
                names
                    .iter()
                    .any(|name| extension.eq_ignore_ascii_case(name))
            })
            .map(|&(_, language)| language)
    }

    /// The name `--lang` filters on.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Rust => "rust",
            Self::Doc(DocFormat::Markdown) => "markdown",
            Self::Doc(DocFormat::Rst) => "rst",
            Self::Doc(DocFormat::AsciiDoc) => "asciidoc",
            Self::Doc(DocFormat::Text) => "text",
            Self::Config(ConfigFormat::Toml) => "toml",
            Self::Config(ConfigFormat::Yaml) => "yaml",
            Self::Config(ConfigFormat::Json) => "json",
        }
    }

    /// `true` for documentation languages.
    #[must_use]
    pub fn is_doc(self) -> bool {
        matches!(self, Self::Doc(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn languages_are_detected_by_extension() {
        let of = |path: &str| Language::of_path(Path::new(path));
        assert_eq!(of("src/lib.rs"), Some(Language::Rust));
        assert_eq!(of("a/b.MD"), Some(Language::Doc(DocFormat::Markdown)));
        assert_eq!(of("x.mdx"), Some(Language::Doc(DocFormat::Markdown)));
        assert_eq!(of("x.rst"), Some(Language::Doc(DocFormat::Rst)));
        assert_eq!(of("x.adoc"), Some(Language::Doc(DocFormat::AsciiDoc)));
        assert_eq!(of("notes.txt"), Some(Language::Doc(DocFormat::Text)));
        assert_eq!(of("Cargo.toml"), Some(Language::Config(ConfigFormat::Toml)));
        assert_eq!(of("ci.yml"), Some(Language::Config(ConfigFormat::Yaml)));
        assert_eq!(
            of("tsconfig.json"),
            Some(Language::Config(ConfigFormat::Json))
        );
        assert_eq!(of("logo.png"), None);
        assert_eq!(of("Makefile"), None);
        assert!(Language::Doc(DocFormat::Rst).is_doc());
        assert_eq!(Language::Config(ConfigFormat::Yaml).name(), "yaml");
    }
}
