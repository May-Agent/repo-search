//! Query analysis: the lexical terms of a free-text query and the symbol
//! names it mentions.

use std::borrow::Cow;

use unicode_general_category::{GeneralCategory, get_general_category};

use crate::store::split_identifiers;

/// The evaluation harness's stop words (`STOP` in `eval_retrieval.py`), one
/// per line, verbatim: the retrieval quality measured there holds only for
/// this exact list.
const STOP_WORDS: &str = include_str!("stopwords.txt");

/// A symbol path the query names, as `::`-separated segments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolName(Vec<String>);

impl SymbolName {
    /// Whether `symbol` (a `::`-joined symbol path) is this name or ends with
    /// it on a segment boundary: `Environment::insert` is named by `insert`
    /// and by `Environment::insert`, not by `Environment`.
    #[must_use]
    pub fn names(&self, symbol: &str) -> bool {
        let mut have = symbol.rsplit("::");
        self.0.iter().rev().all(|want| have.next() == Some(want))
    }
}

impl SymbolName {
    /// The FTS5 phrase matching text that spells this name, in any case and
    /// with any separator (`env_insert`, `Environment::insert`), the way the
    /// index tokenizes it; `None` when the name has no ASCII word.
    #[must_use]
    pub fn mention_expr(&self) -> Option<String> {
        let lowered: Vec<String> = self.0.iter().map(|s| s.to_lowercase()).collect();
        let words: Vec<Cow<'_, str>> = lowered.iter().flat_map(|s| self::words(s)).collect();
        (!words.is_empty()).then(|| format!("\"{}\"", words.join(" ")))
    }
}

/// What a query means to the lexical channel and to definition ranking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryTerms {
    terms: Vec<String>,
    symbols: Vec<SymbolName>,
}

impl QueryTerms {
    /// Analyzes `query`.
    ///
    /// Terms are built as the evaluation harness builds them: the query and
    /// its camel-split identifiers (`ComputedSignal` gives `Computed Signal`,
    /// matching how the index stores identifiers) are lowercased and cut the
    /// way `unicode61` tokenizes the index — runs of alphanumeric characters
    /// plus the combining marks it strips, so `Größe` is the one term
    /// `größe`, not `gr` and `e`, and `İstanbul` and NFD `cafés` are the
    /// index's `istanbul` and `cafes`. Words of one character and stop words
    /// are dropped; the rest are distinct, in order of first appearance.
    ///
    /// A symbol name is an identifier-like token: it contains `_`, a camel
    /// boundary, or is a `::` path. Plain words never name a symbol, so a
    /// natural-language query does not promote every function called like one
    /// of its words.
    #[must_use]
    pub fn parse(query: &str) -> Self {
        let mut terms: Vec<String> = Vec::new();
        let split = split_identifiers(query);
        for text in [query.to_lowercase(), split.to_lowercase()] {
            for word in words(&text) {
                if word.chars().nth(1).is_some()
                    && !is_stop_word(&word)
                    && !terms.iter().any(|t| t.as_str() == word)
                {
                    terms.push(word.into_owned());
                }
            }
        }

        let mut symbols: Vec<SymbolName> = Vec::new();
        for token in query.split(|c: char| !(c.is_alphanumeric() || c == '_' || c == ':')) {
            let token = token.trim_matches(':');
            if token.is_empty() {
                continue;
            }
            let segments: Vec<&str> = token.split("::").filter(|s| !s.is_empty()).collect();
            let identifier_like =
                segments.len() > 1 || token.contains('_') || !split_identifiers(token).is_empty();
            if identifier_like {
                let name = SymbolName(segments.iter().map(|s| (*s).to_owned()).collect());
                if !symbols.contains(&name) {
                    symbols.push(name);
                }
            }
        }
        Self { terms, symbols }
    }

    /// The lexical terms: lowercased, distinct, stop words removed.
    #[must_use]
    pub fn terms(&self) -> &[String] {
        &self.terms
    }

    /// The symbol names the query mentions.
    #[must_use]
    pub fn symbols(&self) -> &[SymbolName] {
        &self.symbols
    }

    /// The FTS5 expression matching any term, each quoted as a string; `None`
    /// when no term survives, which means the lexical channel has nothing to
    /// ask.
    #[must_use]
    pub fn fts_match(&self) -> Option<String> {
        if self.terms.is_empty() {
            return None;
        }
        let quoted: Vec<String> = self.terms.iter().map(|t| format!("\"{t}\"")).collect();
        Some(quoted.join(" OR "))
    }
}

/// The combining marks `unicode61` keeps inside a token and strips from it:
/// `sqlite3Fts5UnicodeIsdiacritic`'s set, the U+0300–U+0331 Latin-script
/// marks its `remove_diacritics` drops. Other marks are separators to
/// `unicode61` (its token categories are `L* N* Co`), so they cut here too.
const DIACRITICS: u64 = 0x0003_61F8_0802_9FDF;

fn is_diacritic(c: char) -> bool {
    let c = c as u32;
    (0x300..0x340).contains(&c) && (DIACRITICS >> (c - 0x300)) & 1 == 1
}

/// Whether `unicode61` keeps `c` inside a token: its token categories are
/// `L* N* Co` — letters, numbers and private use — and a diacritic continues
/// the token it follows.
fn is_token_char(c: char) -> bool {
    matches!(
        get_general_category(c),
        GeneralCategory::LowercaseLetter
            | GeneralCategory::UppercaseLetter
            | GeneralCategory::TitlecaseLetter
            | GeneralCategory::ModifierLetter
            | GeneralCategory::OtherLetter
            | GeneralCategory::DecimalNumber
            | GeneralCategory::LetterNumber
            | GeneralCategory::OtherNumber
            | GeneralCategory::PrivateUse
    ) || is_diacritic(c)
}

/// The tokens the index's `unicode61` tokenizer cuts `text` into: runs of
/// token characters, emitted without the combining marks (a run of marks
/// alone is a separator to `unicode61`, so a word stripped to nothing does
/// not survive). On already-lowercased ASCII text this is exactly
/// `[a-z0-9]+`, so ASCII terms are unchanged.
fn words(text: &str) -> impl Iterator<Item = Cow<'_, str>> {
    text.split(|c: char| !is_token_char(c))
        .filter(|w| !w.is_empty())
        .map(|w| {
            if w.contains(is_diacritic) {
                Cow::Owned(w.chars().filter(|&c| !is_diacritic(c)).collect())
            } else {
                Cow::Borrowed(w)
            }
        })
        .filter(|w| !w.is_empty())
}

fn is_stop_word(word: &str) -> bool {
    STOP_WORDS.lines().any(|stop| stop == word)
}

#[cfg(test)]
mod tests {
    use super::QueryTerms;

    fn terms(query: &str) -> Vec<String> {
        QueryTerms::parse(query).terms().to_vec()
    }

    #[test]
    fn lowercases_and_drops_stop_words() {
        assert_eq!(
            terms("How does the Environment store a value?"),
            ["how", "environment", "store", "value"]
        );
    }

    #[test]
    fn splits_camel_case_keeping_the_whole_identifier() {
        assert_eq!(
            terms("ComputedSignal"),
            ["computedsignal", "computed", "signal"]
        );
        assert_eq!(
            terms("parse HTTPServer"),
            ["parse", "httpserver", "http", "server"]
        );
    }

    #[test]
    fn the_whole_query_comes_before_the_split_parts() {
        // The evaluation lists every word of the query, then the parts.
        assert_eq!(
            terms("ComputedSignal cache"),
            ["computedsignal", "cache", "computed", "signal"]
        );
    }

    #[test]
    fn snake_case_and_paths_split_on_separators() {
        assert_eq!(
            terms("env_insert Environment::insert"),
            ["env", "insert", "environment"]
        );
    }

    #[test]
    fn stop_word_parts_of_identifiers_are_dropped() {
        assert_eq!(terms("isEmpty"), ["isempty", "empty"]);
    }

    #[test]
    fn identifier_like_words_are_not_stop_words() {
        // Only the evaluation's list is dropped: `new`, `get`, `type` stay.
        assert_eq!(terms("get new type"), ["get", "new", "type"]);
    }

    #[test]
    fn terms_are_distinct_and_single_characters_dropped() {
        assert_eq!(terms("x signal Signal SIGNAL"), ["signal"]);
    }

    #[test]
    fn an_all_stop_word_query_has_no_terms() {
        let q = QueryTerms::parse("is the of");
        assert!(q.terms().is_empty());
        assert_eq!(q.fts_match(), None);
    }

    #[test]
    fn the_stop_list_is_the_evaluations() {
        let words: Vec<&str> = super::STOP_WORDS.lines().collect();
        assert_eq!(words.len(), 63);
        let mut distinct = words.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(distinct.len(), words.len());
        assert_eq!(words.first(), Some(&"a"));
        assert_eq!(words.last(), Some(&"just"));
    }

    /// Queries from the evaluation (and edge cases) with the MATCH expression
    /// `eval_retrieval.terms` builds for them, generated by that script.
    #[test]
    fn the_lexical_query_matches_the_evaluations() {
        #[derive(serde::Deserialize)]
        struct Case {
            query: String,
            #[serde(rename = "match")]
            expected: Option<String>,
        }
        let cases: Vec<Case> =
            serde_json::from_str(include_str!("query_parity.json")).expect("fixture parses");
        assert!(cases.len() > 50);
        for case in cases {
            assert_eq!(
                QueryTerms::parse(&case.query).fts_match(),
                case.expected,
                "{:?}",
                case.query
            );
        }
    }

    #[test]
    fn builds_a_quoted_or_expression() {
        let q = QueryTerms::parse("ComputedSignal cache");
        assert_eq!(
            q.fts_match().unwrap(),
            r#""computedsignal" OR "cache" OR "computed" OR "signal""#
        );
    }

    #[test]
    fn non_ascii_letters_keep_their_words() {
        // `unicode61` tokenizes these as single runs, so the query does too.
        assert_eq!(terms("Größe"), ["größe"]);
        assert_eq!(terms("naïve café"), ["naïve", "café"]);
        assert_eq!(terms("日本語"), ["日本語"]);
    }

    #[test]
    fn combining_marks_stay_inside_the_word_and_are_stripped() {
        // `unicode61` keeps the Latin combining marks (U+0300–U+0331) inside a
        // token and strips them, so the index holds `istanbul` and `cafes`.
        assert_eq!(terms("İstanbul"), ["istanbul"]);
        assert_eq!(terms("cafe\u{301}s"), ["cafes"]);
        assert_eq!(terms("Que\u{301}bec"), ["quebec"]);
    }

    #[test]
    fn marks_unicode61_does_not_strip_cut_the_word() {
        // A combining mark outside the diacritic set (U+0305, Mn) is a
        // separator to `unicode61`, so it is one here too.
        assert_eq!(terms("ab\u{305}cd"), ["ab", "cd"]);
        // A spacing mark (U+093E, Mc) splits the same way.
        assert_eq!(terms("xy\u{93e}zw"), ["xy", "zw"]);
        // Marks alone carry no token.
        assert_eq!(terms("\u{301} \u{301}"), Vec::<String>::new());
    }

    #[test]
    fn a_single_character_is_dropped_even_non_ascii() {
        assert_eq!(terms("日"), Vec::<String>::new());
        assert_eq!(terms("é"), Vec::<String>::new());
        // And so is a word that is one character once its marks are stripped.
        assert_eq!(terms("e\u{301}"), Vec::<String>::new());
        assert_eq!(terms("日 のクエリ"), ["のクエリ"]);
    }

    #[test]
    fn only_identifier_like_tokens_name_symbols() {
        let q =
            QueryTerms::parse("layout of ComputedSignal and env_insert plus Environment::insert");
        assert_eq!(q.symbols().len(), 3);
        assert!(q.symbols().iter().any(|s| s.names("ComputedSignal")));
        assert!(!q.symbols().iter().any(|s| s.names("layout")));
        assert!(q.symbols().iter().any(|s| s.names("env_insert")));
        assert!(q.symbols().iter().any(|s| s.names("Environment::insert")));
    }

    #[test]
    fn a_mention_is_a_phrase_of_the_names_words() {
        let q = QueryTerms::parse("ComputedSignal env_insert Environment::insert");
        let exprs: Vec<Option<String>> = q.symbols().iter().map(|s| s.mention_expr()).collect();
        assert_eq!(
            exprs,
            [
                Some(r#""computedsignal""#.to_owned()),
                Some(r#""env insert""#.to_owned()),
                Some(r#""environment insert""#.to_owned()),
            ]
        );
    }

    #[test]
    fn a_symbol_name_matches_on_segment_boundaries() {
        let q = QueryTerms::parse("Environment::insert");
        let name = &q.symbols()[0];
        assert!(name.names("Environment::insert"));
        assert!(name.names("crate::Environment::insert"));
        assert!(!name.names("Environment"));
        assert!(!name.names("MyEnvironment::insert"));
    }
}
