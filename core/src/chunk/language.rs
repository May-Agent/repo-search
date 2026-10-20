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
    #[cfg(feature = "lang-rust")]
    Rust,
    /// Swift source.
    #[cfg(feature = "lang-swift")]
    Swift,
    /// TypeScript source (`.ts` only; `.tsx` is [`Language::Tsx`]).
    #[cfg(feature = "lang-typescript")]
    TypeScript,
    /// TypeScript with JSX.
    #[cfg(feature = "lang-typescript")]
    Tsx,
    /// JavaScript source, including JSX, `.mjs` and `.cjs`.
    #[cfg(feature = "lang-javascript")]
    JavaScript,
    /// Python source.
    #[cfg(feature = "lang-python")]
    Python,
    /// C source and headers.
    #[cfg(feature = "lang-c")]
    C,
    /// C++ source and headers.
    #[cfg(feature = "lang-cpp")]
    Cpp,
    /// Go source.
    #[cfg(feature = "lang-go")]
    Go,
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
        // `.h` is a C++ header when that language is built and C
        // otherwise; `.C` and `.H` always name C++ sources. These match
        // case-sensitively and never reach the table below.
        #[cfg(feature = "lang-cpp")]
        if matches!(extension, "h" | "C" | "H") {
            return Some(Language::Cpp);
        }
        #[cfg(all(feature = "lang-c", not(feature = "lang-cpp")))]
        if extension == "h" {
            return Some(Language::C);
        }
        #[cfg(not(feature = "lang-cpp"))]
        if matches!(extension, "C" | "H") {
            return None;
        }
        const EXTENSIONS: &[(&[&str], Language)] = &[
            #[cfg(feature = "lang-rust")]
            (&["rs"], Language::Rust),
            #[cfg(feature = "lang-swift")]
            (&["swift"], Language::Swift),
            #[cfg(feature = "lang-typescript")]
            (&["ts", "mts", "cts"], Language::TypeScript),
            #[cfg(feature = "lang-typescript")]
            (&["tsx"], Language::Tsx),
            #[cfg(feature = "lang-javascript")]
            (&["js", "jsx", "mjs", "cjs"], Language::JavaScript),
            #[cfg(feature = "lang-python")]
            (&["py", "pyi"], Language::Python),
            #[cfg(feature = "lang-c")]
            (&["c"], Language::C),
            #[cfg(feature = "lang-cpp")]
            (&["cc", "cpp", "cxx", "hpp", "hh"], Language::Cpp),
            #[cfg(feature = "lang-go")]
            (&["go"], Language::Go),
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
            #[cfg(feature = "lang-rust")]
            Self::Rust => "rust",
            #[cfg(feature = "lang-swift")]
            Self::Swift => "swift",
            #[cfg(feature = "lang-typescript")]
            Self::TypeScript => "typescript",
            #[cfg(feature = "lang-typescript")]
            Self::Tsx => "tsx",
            #[cfg(feature = "lang-javascript")]
            Self::JavaScript => "javascript",
            #[cfg(feature = "lang-python")]
            Self::Python => "python",
            #[cfg(feature = "lang-c")]
            Self::C => "c",
            #[cfg(feature = "lang-cpp")]
            Self::Cpp => "cpp",
            #[cfg(feature = "lang-go")]
            Self::Go => "go",
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
        #[cfg(feature = "lang-rust")]
        assert_eq!(of("src/lib.rs"), Some(Language::Rust));
        #[cfg(feature = "lang-swift")]
        {
            assert_eq!(of("main.swift"), Some(Language::Swift));
            assert_eq!(Language::Swift.name(), "swift");
        }
        #[cfg(feature = "lang-typescript")]
        {
            assert_eq!(of("a.ts"), Some(Language::TypeScript));
            assert_eq!(of("b.TSX"), Some(Language::Tsx));
            assert_eq!(of("a.mts"), Some(Language::TypeScript));
            assert_eq!(of("a.cts"), Some(Language::TypeScript));
        }
        #[cfg(feature = "lang-javascript")]
        {
            assert_eq!(of("c.jsx"), Some(Language::JavaScript));
            assert_eq!(of("d.mjs"), Some(Language::JavaScript));
            assert_eq!(of("e.cjs"), Some(Language::JavaScript));
            assert_eq!(Language::JavaScript.name(), "javascript");
        }
        #[cfg(feature = "lang-python")]
        {
            assert_eq!(of("mod.py"), Some(Language::Python));
            assert_eq!(of("mod.pyi"), Some(Language::Python));
        }
        #[cfg(feature = "lang-c")]
        assert_eq!(of("x.c"), Some(Language::C));
        #[cfg(all(feature = "lang-c", not(feature = "lang-cpp")))]
        {
            assert_eq!(of("x.h"), Some(Language::C));
            assert_eq!(of("x.C"), None);
            assert_eq!(of("x.H"), None);
        }
        #[cfg(feature = "lang-cpp")]
        {
            assert_eq!(of("x.cc"), Some(Language::Cpp));
            assert_eq!(of("x.CPP"), Some(Language::Cpp));
            assert_eq!(of("x.hpp"), Some(Language::Cpp));
            assert_eq!(of("x.h"), Some(Language::Cpp));
            assert_eq!(of("x.C"), Some(Language::Cpp));
            assert_eq!(of("x.H"), Some(Language::Cpp));
            assert_eq!(Language::Cpp.name(), "cpp");
        }
        #[cfg(feature = "lang-go")]
        assert_eq!(of("main.go"), Some(Language::Go));
    }
}
