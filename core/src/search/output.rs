//! The result lists, their token-budgeted selection, and the text and JSON
//! renderings.

use std::collections::HashMap;
use std::fmt;
use std::io;

use askama::Template;
use serde::Serialize;

use super::fuse::Channel;
use super::{Error, TokenCounter};
use crate::reconcile::Report;
use crate::store::{ChunkRow, Side};

impl fmt::Display for Channel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Bm25 => "bm25",
            Self::Vector => "vector",
        })
    }
}

/// What the caller's reconcile barrier found, the input of the freshness line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reconciliation {
    /// The commit `HEAD` pointed at.
    pub commit: String,
    /// Files added, modified or removed by the barrier.
    pub changed_files: u64,
    /// Chunk rows in the index.
    pub chunks_total: u64,
    /// Chunk rows whose vector is cached.
    pub chunks_with_vector: u64,
}

impl Reconciliation {
    /// The freshness input for a barrier that produced `report` at `commit`.
    #[must_use]
    pub fn new(commit: impl Into<String>, report: &Report) -> Self {
        Self {
            commit: commit.into(),
            changed_files: report.files_changed(),
            chunks_total: report.chunks_total,
            chunks_with_vector: report.chunks_with_vector,
        }
    }
}

/// How many chunks have a vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Coverage {
    with_vector: u64,
    total: u64,
}

impl Coverage {
    /// `with_vector` of `total` chunks, clamped to the total.
    #[must_use]
    pub fn new(with_vector: u64, total: u64) -> Self {
        Self {
            with_vector: with_vector.min(total),
            total,
        }
    }

    /// The fraction of chunks with a vector; `1.0` for an empty index.
    #[must_use]
    pub fn fraction(self) -> f64 {
        if self.total == 0 {
            1.0
        } else {
            self.with_vector as f64 / self.total as f64
        }
    }

    /// The coverage in whole percent, rounded down so that anything short of
    /// full coverage never reads `100`; computed on integers, since
    /// `0.29 * 100.0` is not `29.0`.
    #[must_use]
    pub fn percent(self) -> u32 {
        if self.total == 0 {
            return 100;
        }
        u32::try_from(u128::from(self.with_vector) * 100 / u128::from(self.total))
            .expect("a share of at most 100")
    }
}

impl Serialize for Coverage {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_f64(self.fraction())
    }
}

/// How fresh the answer is: the `-- fresh` line and the JSON `freshness`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Freshness {
    /// The commit `HEAD` pointed at.
    pub commit: String,
    /// Files the barrier reconciled.
    pub changed_files: u64,
    /// Chunks with a vector after this query's on-the-spot embedding;
    /// serialized as a fraction.
    pub vectors: Coverage,
}

impl Freshness {
    pub(super) fn new(reconciliation: Reconciliation, embedded_on_the_spot: u64) -> Self {
        Self {
            commit: reconciliation.commit,
            changed_files: reconciliation.changed_files,
            vectors: Coverage::new(
                reconciliation.chunks_with_vector + embedded_on_the_spot,
                reconciliation.chunks_total,
            ),
        }
    }

    /// The coverage in whole percent, rounded down.
    #[must_use]
    pub fn percent(&self) -> u32 {
        self.vectors.percent()
    }
}

/// One chunk of the code list.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CodeChunk {
    /// First and last line, 1-based inclusive.
    pub lines: [u32; 2],
    /// The symbol paths of the chunk's items, `, `-joined; `None` when it
    /// defines none.
    pub symbol: Option<String>,
    /// The channels that found it.
    pub channels: Vec<Channel>,
    /// The text, read from the working tree.
    pub text: String,
}

/// One chunk of the docs list.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DocChunk {
    /// First and last line, 1-based inclusive.
    pub lines: [u32; 2],
    /// The heading path, e.g. `Layout › Stacks › Alignment`.
    pub heading: String,
    /// The channels that found it.
    pub channels: Vec<Channel>,
    /// The text, read from the working tree.
    pub text: String,
}

/// A file with its emitted chunks.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FileHit<C> {
    /// Worktree-relative path.
    pub path: String,
    /// The fused score of the file's best chunk.
    pub score: f64,
    /// The emitted chunks in rank order; empty (and omitted from JSON) for a
    /// files-only answer.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub chunks: Vec<C>,
}

/// The answer to a query. A list is `None` when it is empty.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Results {
    /// The code list.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<Vec<FileHit<CodeChunk>>>,
    /// The docs list.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub docs: Option<Vec<FileHit<DocChunk>>>,
    /// How fresh the index was when the query ran.
    pub freshness: Freshness,
}

impl Results {
    /// Whether both lists are empty (the CLI's exit code 1).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.code.is_none() && self.docs.is_none()
    }

    /// The `--json` rendering.
    #[must_use]
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("results contain only JSON-representable values")
    }

    /// The text rendering: a `Code:` and a `Docs:` section, then the
    /// freshness line.
    #[must_use]
    pub fn to_text(&self) -> String {
        let mut sections = Vec::new();
        if let Some(code) = &self.code {
            sections.push(SectionView {
                title: "Code",
                files: code
                    .iter()
                    .map(|file| FileView {
                        path: &file.path,
                        chunks: file
                            .chunks
                            .iter()
                            .map(|c| ChunkView {
                                lines: c.lines,
                                label: c.symbol.as_deref().unwrap_or(""),
                                channels: &c.channels,
                                text: &c.text,
                            })
                            .collect(),
                    })
                    .collect(),
            });
        }
        if let Some(docs) = &self.docs {
            sections.push(SectionView {
                title: "Docs",
                files: docs
                    .iter()
                    .map(|file| FileView {
                        path: &file.path,
                        chunks: file
                            .chunks
                            .iter()
                            .map(|c| ChunkView {
                                lines: c.lines,
                                label: &c.heading,
                                channels: &c.channels,
                                text: &c.text,
                            })
                            .collect(),
                    })
                    .collect(),
            });
        }
        TextReport {
            sections,
            freshness: &self.freshness,
        }
        .render()
        .expect("the template renders its own typed context")
    }
}

#[derive(Template)]
#[template(path = "search.txt")]
struct TextReport<'a> {
    sections: Vec<SectionView<'a>>,
    freshness: &'a Freshness,
}

struct SectionView<'a> {
    title: &'static str,
    files: Vec<FileView<'a>>,
}

struct FileView<'a> {
    path: &'a str,
    chunks: Vec<ChunkView<'a>>,
}

struct ChunkView<'a> {
    lines: [u32; 2],
    label: &'a str,
    channels: &'a [Channel],
    text: &'a str,
}

/// Why a chunk's text could not be read from the working tree.
#[derive(Debug, Clone)]
pub(super) enum ReadError {
    /// The file is gone or no longer holds the indexed chunk.
    Stale { path: String, start: u32, end: u32 },
    /// The file could not be read.
    Io {
        path: String,
        kind: io::ErrorKind,
        message: String,
    },
}

impl<E: std::error::Error + 'static> From<ReadError> for Error<E> {
    fn from(e: ReadError) -> Self {
        match e {
            ReadError::Stale { path, start, end } => Error::Stale { path, start, end },
            ReadError::Io {
                path,
                kind,
                message,
            } => Error::Io {
                path,
                source: io::Error::new(kind, message),
            },
        }
    }
}

/// A chunk of one list, ranked. Its text is read only once the selection
/// reaches it.
#[derive(Debug)]
pub(super) struct RankedChunk {
    pub row: ChunkRow,
    pub score: f64,
    pub channels: Vec<Channel>,
    pub text: Option<Result<String, ReadError>>,
}

impl RankedChunk {
    /// The `, `-joined distinct non-empty symbols of the chunk's members.
    pub fn symbols(&self) -> Option<String> {
        let mut seen: Vec<&str> = Vec::new();
        for member in &self.row.members {
            if !member.symbol.is_empty() && !seen.contains(&member.symbol.as_str()) {
                seen.push(&member.symbol);
            }
        }
        (!seen.is_empty()).then(|| seen.join(", "))
    }

    /// The heading path of a doc chunk: its title without the leading path.
    pub fn heading(&self) -> &str {
        self.row
            .title
            .split_once(" › ")
            .map_or("", |(_, heading)| heading)
    }

    /// What the text rendering prints next to the line range: the symbols of
    /// a code chunk (nothing when it has none), the heading of a docs chunk.
    fn label(&self, side: Side) -> String {
        match side {
            Side::Code => self.symbols().unwrap_or_default(),
            Side::Docs => self.heading().to_owned(),
        }
    }

    /// The text of a chunk the selection emitted.
    fn text(&self) -> &str {
        match &self.text {
            Some(Ok(text)) => text,
            _ => unreachable!("an emitted chunk has its text read"),
        }
    }

    fn lines(&self) -> [u32; 2] {
        [*self.row.lines.start(), *self.row.lines.end()]
    }
}

/// The chunks (indices into a ranked list) one file contributes.
#[derive(Debug, Default)]
pub(super) struct Group {
    pub path: String,
    pub score: f64,
    pub chunks: Vec<usize>,
}

/// Where the selection stands.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Step {
    /// The unit at `from` of the list of `side` is next and its text is unread.
    Needs { side: Side, from: usize },
    /// The budget is filled or the lists are exhausted.
    Done,
}

/// Fills a budget of tokens from the two ranked lists, taking them in turn
/// — code first, one file per turn in rank order — and adding the file's
/// next-best chunk on each of its turns, so a file's second chunk comes only
/// after every ranked file of the list has shown its best one. A pass gives
/// every ranked file of a list one turn and repeats while budget and chunks
/// remain. A list ends at the first chunk that does not fit — what was taken
/// stays — or when every file's chunks are taken, and the other list then
/// continues alone until the budget is used. The very first unit is always
/// taken, so a budget smaller than the best chunk still answers. Files are
/// compared by rank within their own list: scores never cross between the
/// two lists.
///
/// A chunk's cost is what the text rendering prints for it: its text and
/// label plus, for the first chunk of a file, the path (files-only: the
/// path alone). A chunk's text is read only when the selection reaches it,
/// so [`advance`](Self::advance) stops at the first chunk whose text is
/// unread and the caller supplies it. The emitted chunks are grouped by
/// file in the order the files were first chosen, each file's chunks in
/// rank order.
pub(super) struct Selector {
    budget: usize,
    files_only: bool,
    groups: [Vec<Group>; 2],
    /// Each side's files in rank order; built on the first advance.
    files: [Option<Vec<FilePass>>; 2],
    /// The file whose turn of the current pass is next; wraps to the first
    /// file of the list.
    cursor: [usize; 2],
    open: [bool; 2],
    used: usize,
    /// Whether any unit has been emitted: the very first is always taken
    /// whatever its cost (even zero, which `used` alone cannot detect), so a
    /// budget smaller than the best chunk still answers.
    emitted: bool,
    /// The list whose turn it is: 0 code, 1 docs.
    turn: usize,
}

const SIDES: [Side; 2] = [Side::Code, Side::Docs];

/// One file of a list's pass order: its ranked-chunk indices and how far the
/// selection's passes over it have gone.
#[derive(Debug)]
struct FilePass {
    /// The file's ranked-chunk indices in rank order; a file's chunks need
    /// not be consecutive in the ranked list.
    chunks: Vec<usize>,
    /// Chunks already taken: the next turn of the file offers
    /// `chunks[taken]`.
    taken: usize,
    /// The file's entry in `groups`, once its first chunk is taken.
    group: Option<usize>,
}

/// `list` grouped by path: files in order of first (best-ranked) appearance,
/// each with its ranked-chunk indices in rank order.
fn file_order(list: &[RankedChunk]) -> Vec<FilePass> {
    let mut files: Vec<FilePass> = Vec::new();
    let mut by_path: HashMap<&str, usize> = HashMap::new();
    for (i, chunk) in list.iter().enumerate() {
        match by_path.get(chunk.row.path.as_str()) {
            Some(&f) => files[f].chunks.push(i),
            None => {
                by_path.insert(chunk.row.path.as_str(), files.len());
                files.push(FilePass {
                    chunks: vec![i],
                    taken: 0,
                    group: None,
                });
            }
        }
    }
    files
}

impl Selector {
    pub fn new(budget: usize, files_only: bool) -> Self {
        Self {
            budget,
            files_only,
            groups: [Vec::new(), Vec::new()],
            files: [None, None],
            cursor: [0; 2],
            open: [true; 2],
            used: 0,
            emitted: false,
            turn: 0,
        }
    }

    /// Takes chunks until the next one needs its text, the budget is filled
    /// or the lists end.
    ///
    /// # Errors
    ///
    /// When the text of a chunk the selection reaches could not be read.
    pub fn advance(
        &mut self,
        lists: [&[RankedChunk]; 2],
        counter: &impl TokenCounter,
    ) -> Result<Step, ReadError> {
        for (side, list) in lists.iter().enumerate() {
            if self.files[side].is_none() {
                self.files[side] = Some(file_order(list));
            }
        }
        while self.open.iter().any(|o| *o) {
            let side = self.turn;
            if !self.open[side] {
                self.turn = 1 - side;
                continue;
            }
            let files = self.files[side].as_mut().expect("built above");
            if self.files_only {
                // Files-only is a single pass: a file's unit is its path and
                // no text is read.
                let Some(file) = files.get(self.cursor[side]) else {
                    self.open[side] = false;
                    continue;
                };
                let best = &lists[side][file.chunks[0]];
                let cost = counter.count(&best.row.path);
                if self.emitted && self.used + cost > self.budget {
                    self.open[side] = false;
                    continue;
                }
                self.used += cost;
                self.emitted = true;
                self.groups[side].push(Group {
                    path: best.row.path.clone(),
                    score: best.score,
                    chunks: Vec::new(),
                });
                self.cursor[side] += 1;
                self.turn = 1 - side;
                continue;
            }
            // The next file of the pass with a chunk left; a file whose
            // chunks are all taken is skipped, and a full cycle ends the
            // side.
            if files.is_empty() {
                self.open[side] = false;
                continue;
            }
            let mut g = self.cursor[side];
            for _ in 0..files.len() {
                if files[g].taken < files[g].chunks.len() {
                    break;
                }
                g = if g + 1 == files.len() { 0 } else { g + 1 };
            }
            if files[g].taken == files[g].chunks.len() {
                self.open[side] = false;
                continue;
            }
            self.cursor[side] = g;
            let file = &mut files[g];
            let idx = file.chunks[file.taken];
            let chunk = &lists[side][idx];
            // A file's first chunk is also charged the path; a chunk that
            // does not fit ends the list, the chunks already taken stay.
            let mut cost = usize::from(file.taken == 0) * counter.count(&chunk.row.path);
            match &chunk.text {
                None => {
                    return Ok(Step::Needs {
                        side: SIDES[side],
                        from: idx,
                    });
                }
                Some(Err(e)) => return Err(e.clone()),
                Some(Ok(text)) => {
                    cost += counter.count(&chunk.label(SIDES[side])) + counter.count(text);
                }
            }
            if self.emitted && self.used + cost > self.budget {
                self.open[side] = false;
                continue;
            }
            self.used += cost;
            self.emitted = true;
            match file.group {
                Some(group) => self.groups[side][group].chunks.push(idx),
                None => {
                    file.group = Some(self.groups[side].len());
                    self.groups[side].push(Group {
                        path: chunk.row.path.clone(),
                        score: chunk.score,
                        chunks: vec![idx],
                    });
                }
            }
            file.taken += 1;
            self.cursor[side] = if g + 1 == files.len() { 0 } else { g + 1 };
            self.turn = 1 - side;
        }
        Ok(Step::Done)
    }

    /// The selected files of the code list, then of the docs list.
    pub fn finish(self) -> [Vec<Group>; 2] {
        self.groups
    }
}

/// The code list for the selected groups; `None` when empty.
pub(super) fn code_hits(
    ranked: &[RankedChunk],
    groups: Vec<Group>,
) -> Option<Vec<FileHit<CodeChunk>>> {
    hits(groups, |i| {
        let c = &ranked[i];
        CodeChunk {
            lines: c.lines(),
            symbol: c.symbols(),
            channels: c.channels.clone(),
            text: c.text().to_owned(),
        }
    })
}

/// The docs list for the selected groups; `None` when empty.
pub(super) fn doc_hits(
    ranked: &[RankedChunk],
    groups: Vec<Group>,
) -> Option<Vec<FileHit<DocChunk>>> {
    hits(groups, |i| {
        let c = &ranked[i];
        DocChunk {
            lines: c.lines(),
            heading: c.heading().to_owned(),
            channels: c.channels.clone(),
            text: c.text().to_owned(),
        }
    })
}

fn hits<C>(groups: Vec<Group>, chunk: impl Fn(usize) -> C) -> Option<Vec<FileHit<C>>> {
    let hits: Vec<FileHit<C>> = groups
        .into_iter()
        .map(|g| FileHit {
            path: g.path,
            score: g.score,
            chunks: g.chunks.into_iter().map(&chunk).collect(),
        })
        .collect();
    (!hits.is_empty()).then_some(hits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{ChunkId, ContentHash, MemberRow};

    /// Runs the whole selection; every text is already read.
    fn select(
        lists: [&[RankedChunk]; 2],
        budget: usize,
        files_only: bool,
        counter: &impl TokenCounter,
    ) -> [Vec<Group>; 2] {
        let mut selector = Selector::new(budget, files_only);
        assert_eq!(selector.advance(lists, counter).unwrap(), Step::Done);
        selector.finish()
    }

    struct Words;

    impl TokenCounter for Words {
        fn count(&self, text: &str) -> usize {
            text.split_whitespace().count()
        }
    }

    fn ranked(path: &str, symbol: &str, text: &str, score: f64) -> RankedChunk {
        RankedChunk {
            row: ChunkRow {
                id: ChunkId(1),
                path: path.to_owned(),
                title: format!("{path} › fn {symbol}"),
                kind: "function_item".to_owned(),
                lines: 3..=9,
                content_hash: ContentHash::of_chunk("t", text),
                members: vec![MemberRow {
                    bytes: 0..1,
                    kind: "function_item".to_owned(),
                    symbol: symbol.to_owned(),
                }],
            },
            score,
            channels: vec![Channel::Bm25, Channel::Vector],
            text: Some(Ok(text.to_owned())),
        }
    }

    fn doc(path: &str, heading: &str, text: &str, score: f64) -> RankedChunk {
        let mut chunk = ranked(path, "", text, score);
        chunk.row.kind = "doc".to_owned();
        chunk.row.title = format!("{path} › {heading}");
        chunk.row.members[0].symbol.clear();
        chunk.channels = vec![Channel::Vector];
        chunk
    }

    fn freshness() -> Freshness {
        Freshness::new(
            Reconciliation {
                commit: "abc123".to_owned(),
                changed_files: 2,
                chunks_total: 10,
                chunks_with_vector: 9,
            },
            0,
        )
    }

    #[test]
    fn chunks_of_one_file_group_under_its_first_rank() {
        let code = [
            ranked("a.rs", "one", "x y", 0.9),
            ranked("b.rs", "two", "x y", 0.8),
            ranked("a.rs", "three", "x y", 0.7),
        ];
        let [groups, _] = select([&code, &[]], 1000, false, &Words);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].path, "a.rs");
        assert_eq!(groups[0].chunks, [0, 2]);
        assert_eq!(groups[0].score, 0.9);
        assert_eq!(groups[1].path, "b.rs");
    }

    #[test]
    fn a_chunk_that_does_not_fit_ends_its_list_not_just_its_file() {
        // First chunks cost 3 (path, symbol, one word), later chunks of a
        // file 2. Pass one takes a, b, c for 9; b's second chunk (11) does
        // not fit the budget of 11 and ends the whole list — c's second
        // chunk, which would have fit, is never reached.
        let code = [
            ranked("a.rs", "one", "x", 0.9),
            ranked("b.rs", "two", "x", 0.8),
            ranked("b.rs", "big", "w w w w w w w w w w", 0.7),
            ranked("c.rs", "three", "x", 0.6),
            ranked("c.rs", "four", "x", 0.5),
        ];
        let [groups, _] = select([&code, &[]], 11, false, &Words);
        assert_eq!(group_paths(&groups), ["a.rs", "b.rs", "c.rs"]);
        assert_eq!(groups[0].chunks, [0]);
        assert_eq!(groups[1].chunks, [1]);
        assert_eq!(groups[2].chunks, [3]);
    }

    #[test]
    fn the_best_chunk_is_emitted_even_over_budget() {
        let code = [ranked("a.rs", "one", "x y z", 0.9)];
        let [groups, _] = select([&code, &[]], 1, false, &Words);
        assert_eq!(groups[0].chunks, [0]);
    }

    #[test]
    fn a_zero_cost_first_unit_still_closes_the_free_take() {
        // The first unit is free whatever it costs — including nothing: an
        // empty path, label and text price it at 0, and the second unit must
        // still fit the budget on its own.
        let mut first = ranked("", "", "", 0.9);
        first.row.members[0].symbol.clear();
        let code = [first, ranked("b.rs", "two", "x y z", 0.8)];
        let [groups, _] = select([&code, &[]], 0, false, &Words);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].path, "");
        assert_eq!(groups[0].chunks, [0]);
    }

    #[test]
    fn code_and_docs_share_the_budget_alternately() {
        let code = [
            ranked("a.rs", "one", "x y", 0.9),
            ranked("b.rs", "two", "x y", 0.8),
        ];
        let docs = [doc("d.md", "H", "p q", 0.5), doc("e.md", "I", "p q", 0.4)];
        // Each unit costs 4 (path, heading/symbol, two words): room for three.
        let [c, d] = select([&code, &docs], 12, false, &Words);
        assert_eq!((c.len(), d.len()), (2, 1));
    }

    fn group_paths(groups: &[Group]) -> Vec<&str> {
        groups.iter().map(|g| g.path.as_str()).collect()
    }

    #[test]
    fn the_lists_take_turns_one_chunk_per_file_per_pass() {
        // Code ranks a.rs, b.rs, a.rs; docs d.md, e.md, d.md. Leaving every
        // text unread makes `advance` report the selection order: pass one
        // takes a1, d1, b1, e1 — one chunk per file, code first — and pass
        // two a2, d2.
        let mut code = [
            ranked("a.rs", "one", "x", 0.9),
            ranked("b.rs", "two", "x", 0.8),
            ranked("a.rs", "three", "x", 0.7),
        ];
        let mut docs = [
            doc("d.md", "H", "p", 0.5),
            doc("e.md", "I", "p", 0.4),
            doc("d.md", "J", "p", 0.3),
        ];
        for chunk in code.iter_mut().chain(&mut docs) {
            chunk.text = None;
        }
        let mut selector = Selector::new(1000, false);
        let mut taken = Vec::new();
        loop {
            match selector.advance([&code, &docs], &Words).unwrap() {
                Step::Done => break,
                Step::Needs { side, from } => {
                    taken.push((side, from));
                    let chunk = match side {
                        Side::Code => &mut code[from],
                        Side::Docs => &mut docs[from],
                    };
                    chunk.text = Some(Ok("w".to_owned()));
                }
            }
        }
        assert_eq!(
            taken,
            [
                (Side::Code, 0),
                (Side::Docs, 0),
                (Side::Code, 1),
                (Side::Docs, 1),
                (Side::Code, 2),
                (Side::Docs, 2),
            ]
        );
        let [c, d] = selector.finish();
        assert_eq!(group_paths(&c), ["a.rs", "b.rs"]);
        assert_eq!(c[0].chunks, [0, 2]);
        assert_eq!(group_paths(&d), ["d.md", "e.md"]);
        assert_eq!(d[0].chunks, [0, 2]);
    }

    #[test]
    fn a_files_second_chunk_comes_only_after_every_files_first() {
        // a.rs ranks first and third: its second chunk is reached only on the
        // second pass, after b's first. a1 + d1 + b1 = 9 fills the budget, so
        // a2 is never emitted — a whole-file take of a would have emitted it.
        let code = [
            ranked("a.rs", "one", "x", 0.9),
            ranked("b.rs", "two", "x", 0.8),
            ranked("a.rs", "three", "x", 0.7),
        ];
        let docs = [doc("d.md", "H", "p", 0.5), doc("e.md", "I", "p", 0.4)];
        let [c, d] = select([&code, &docs], 9, false, &Words);
        assert_eq!(group_paths(&c), ["a.rs", "b.rs"]);
        assert_eq!(c[0].chunks, [0]);
        assert_eq!(c[1].chunks, [1]);
        assert_eq!(group_paths(&d), ["d.md"]);
    }

    #[test]
    fn output_groups_files_in_first_chosen_order_and_chunks_in_rank_order() {
        // The passes emit a1, b1, c1, then a2, b2; the output groups each
        // file's chunks in rank order and lists the files in the order they
        // were first chosen.
        let code = [
            ranked("a.rs", "one", "x", 0.9),
            ranked("b.rs", "two", "x", 0.8),
            ranked("a.rs", "three", "x", 0.7),
            ranked("b.rs", "four", "x", 0.6),
            ranked("c.rs", "five", "x", 0.5),
        ];
        let [groups, _] = select([&code, &[]], 1000, false, &Words);
        assert_eq!(group_paths(&groups), ["a.rs", "b.rs", "c.rs"]);
        assert_eq!(groups[0].chunks, [0, 2]);
        assert_eq!(groups[1].chunks, [1, 3]);
        assert_eq!(groups[2].chunks, [4]);
    }

    #[test]
    fn an_exhausted_list_leaves_the_turns_to_the_others_next_pass() {
        // Code ends after a's only chunk; docs takes every remaining turn and
        // reaches d's second chunk on its own second pass.
        let code = [ranked("a.rs", "one", "x", 0.9)];
        let docs = [
            doc("d.md", "H", "p", 0.5),
            doc("e.md", "I", "p", 0.4),
            doc("d.md", "J", "p", 0.3),
        ];
        let [c, d] = select([&code, &docs], 1000, false, &Words);
        assert_eq!(group_paths(&c), ["a.rs"]);
        assert_eq!(group_paths(&d), ["d.md", "e.md"]);
        assert_eq!(d[0].chunks, [0, 2]);
        assert_eq!(d[1].chunks, [1]);
    }

    #[test]
    fn the_budget_can_run_out_in_the_middle_of_a_pass() {
        // a1 + d1 + b1 = 9 fills the budget inside pass one: e.md is never
        // taken and a's second chunk is never reached.
        let code = [
            ranked("a.rs", "one", "x", 0.9),
            ranked("b.rs", "two", "x", 0.8),
            ranked("a.rs", "three", "x", 0.7),
        ];
        let docs = [doc("d.md", "H", "p", 0.5), doc("e.md", "I", "p", 0.4)];
        let [c, d] = select([&code, &docs], 9, false, &Words);
        assert_eq!(group_paths(&c), ["a.rs", "b.rs"]);
        assert_eq!(c[0].chunks, [0]);
        assert_eq!(c[1].chunks, [1]);
        assert_eq!(group_paths(&d), ["d.md"]);
    }

    #[test]
    fn the_selection_uses_ranks_never_scores() {
        // The same ranks under different scores — including a second-pass
        // chunk — give the same selection.
        let code_a = [
            ranked("a.rs", "one", "x", 0.9),
            ranked("b.rs", "two", "x", 0.1),
            ranked("a.rs", "three", "x", 0.05),
        ];
        let docs_a = [doc("d.md", "H", "p", 0.5), doc("e.md", "I", "p", 0.4)];
        let code_b = [
            ranked("a.rs", "one", "x", 0.01),
            ranked("b.rs", "two", "x", 0.99),
            ranked("a.rs", "three", "x", 0.98),
        ];
        let docs_b = [doc("d.md", "H", "p", 0.001), doc("e.md", "I", "p", 1.0)];
        let shape = |[c, d]: [Vec<Group>; 2]| {
            (
                c.iter()
                    .map(|g| (g.path.clone(), g.chunks.clone()))
                    .collect::<Vec<_>>(),
                d.iter()
                    .map(|g| (g.path.clone(), g.chunks.clone()))
                    .collect::<Vec<_>>(),
            )
        };
        assert_eq!(
            shape(select([&code_a, &docs_a], 11, false, &Words)),
            shape(select([&code_b, &docs_b], 11, false, &Words))
        );
    }

    #[test]
    fn files_only_lists_each_file_once_by_path_cost() {
        let code = [
            ranked("a.rs", "one", "lots of words here", 0.9),
            ranked("a.rs", "two", "more", 0.8),
            ranked("b.rs", "three", "x", 0.7),
            ranked("c.rs", "four", "x", 0.6),
        ];
        let [groups, _] = select([&code, &[]], 2, true, &Words);
        assert_eq!(
            groups.iter().map(|g| g.path.as_str()).collect::<Vec<_>>(),
            ["a.rs", "b.rs"]
        );
        assert!(groups.iter().all(|g| g.chunks.is_empty()));
    }

    fn results() -> Results {
        let code = [ranked(
            "src/env.rs",
            "Environment::insert",
            "fn insert() {\n    1\n}",
            0.0328,
        )];
        let docs = [doc(
            "docs/guide.md",
            "Guide › Stacks",
            "Stacks align items.",
            0.0164,
        )];
        let [cg, dg] = select([&code, &docs], 1000, false, &Words);
        Results {
            code: code_hits(&code, cg),
            docs: doc_hits(&docs, dg),
            freshness: freshness(),
        }
    }

    #[test]
    fn json_has_the_specified_shape() {
        let json: serde_json::Value = serde_json::from_str(&results().to_json()).unwrap();
        let file = &json["code"][0];
        assert_eq!(file["path"], "src/env.rs");
        assert_eq!(file["score"], 0.0328);
        let chunk = &file["chunks"][0];
        assert_eq!(chunk["lines"], serde_json::json!([3, 9]));
        assert_eq!(chunk["symbol"], "Environment::insert");
        assert_eq!(chunk["channels"], serde_json::json!(["bm25", "vector"]));
        assert_eq!(chunk["text"], "fn insert() {\n    1\n}");
        assert!(chunk.get("heading").is_none());

        let doc_file = &json["docs"][0];
        assert_eq!(doc_file["path"], "docs/guide.md");
        assert_eq!(doc_file["score"], 0.0164);
        let doc_chunk = &doc_file["chunks"][0];
        assert_eq!(doc_chunk["lines"], serde_json::json!([3, 9]));
        assert_eq!(doc_chunk["heading"], "Guide › Stacks");
        assert_eq!(doc_chunk["channels"], serde_json::json!(["vector"]));
        assert_eq!(doc_chunk["text"], "Stacks align items.");
        assert!(doc_chunk.get("symbol").is_none());

        assert_eq!(json["freshness"]["commit"], "abc123");
        assert_eq!(json["freshness"]["changed_files"], 2);
        assert_eq!(json["freshness"]["vectors"], 0.9);
    }

    #[test]
    fn an_empty_list_is_omitted_from_json() {
        let mut r = results();
        r.docs = None;
        let json: serde_json::Value = serde_json::from_str(&r.to_json()).unwrap();
        assert!(json.get("docs").is_none());
        assert!(json.get("code").is_some());
        r.code = None;
        assert!(r.is_empty());
    }

    #[test]
    fn files_only_json_omits_chunks() {
        let code = [ranked("a.rs", "one", "x", 0.9)];
        let [cg, _] = select([&code, &[]], 100, true, &Words);
        let r = Results {
            code: code_hits(&code, cg),
            docs: None,
            freshness: freshness(),
        };
        let json: serde_json::Value = serde_json::from_str(&r.to_json()).unwrap();
        assert_eq!(json["code"][0]["path"], "a.rs");
        assert!(json["code"][0].get("chunks").is_none());
    }

    #[test]
    fn text_has_sections_chunks_and_the_freshness_line() {
        let expected = "\
Code:
src/env.rs
  3-9  Environment::insert  [bm25+vector]
    fn insert() {
        1
    }

Docs:
docs/guide.md
  3-9  Guide › Stacks  [vector]
    Stacks align items.

-- fresh @ abc123 +2 changed files reconciled; vectors 90%
";
        assert_eq!(results().to_text(), expected);
    }

    #[test]
    fn text_omits_an_empty_section_and_lists_files_alone() {
        let code = [
            ranked("a.rs", "one", "x", 0.9),
            ranked("b.rs", "two", "x", 0.8),
        ];
        let [cg, _] = select([&code, &[]], 100, true, &Words);
        let r = Results {
            code: code_hits(&code, cg),
            docs: None,
            freshness: freshness(),
        };
        assert_eq!(
            r.to_text(),
            "Code:\na.rs\nb.rs\n\n-- fresh @ abc123 +2 changed files reconciled; vectors 90%\n"
        );
    }

    #[test]
    fn coverage_rounds_down_and_counts_on_the_spot_vectors() {
        let r = |with_vector, embedded| {
            Freshness::new(
                Reconciliation {
                    commit: String::new(),
                    changed_files: 0,
                    chunks_total: 1000,
                    chunks_with_vector: with_vector,
                },
                embedded,
            )
            .percent()
        };
        assert_eq!(r(999, 0), 99);
        assert_eq!(r(900, 100), 100);
        // More vectors than chunks cannot read above full.
        assert_eq!(r(1000, 5), 100);
        let empty = Freshness::new(
            Reconciliation {
                commit: String::new(),
                changed_files: 0,
                chunks_total: 0,
                chunks_with_vector: 0,
            },
            0,
        );
        assert_eq!(empty.percent(), 100);
    }

    #[test]
    fn percent_is_integer_arithmetic_rounded_down() {
        // 0.29 * 100.0 and 0.57 * 100.0 are 28.999... and 56.999... in floats.
        assert_eq!(Coverage::new(29, 100).percent(), 29);
        assert_eq!(Coverage::new(57, 100).percent(), 57);
        assert_eq!(Coverage::new(1, 3).percent(), 33);
        assert_eq!(Coverage::new(2, 3).percent(), 66);
        assert_eq!(Coverage::new(99_999, 100_000).percent(), 99);
        assert_eq!(Coverage::new(0, 0).percent(), 100);
        assert_eq!(Coverage::new(u64::MAX - 1, u64::MAX).percent(), 99);
    }

    #[test]
    fn a_code_chunk_without_symbols_is_not_charged_a_heading() {
        // The text rendering prints no label for it, so the budget does not
        // count its title (`a.rs › fn `).
        let mut chunk = ranked("a.rs", "", "x y", 0.9);
        chunk.row.members[0].symbol.clear();
        assert_eq!(chunk.label(Side::Code), "");
        assert_eq!(chunk.label(Side::Docs), "fn ");
        let mut second = ranked("a.rs", "", "z", 0.8);
        second.row.members[0].symbol.clear();
        let code = [chunk, second];
        // path (1) + text (2), then text (1) of the same file: 4 tokens. A
        // charged heading would add 1 to each and overflow the budget.
        let [groups, _] = select([&code, &[]], 4, false, &Words);
        assert_eq!(groups[0].chunks, [0, 1]);
    }

    #[test]
    fn the_selection_asks_for_text_only_of_units_it_reaches() {
        let mut code = [
            ranked("a.rs", "one", "x y", 0.9),
            ranked("b.rs", "two", "x y", 0.8),
            ranked("c.rs", "three", "x y", 0.7),
            ranked("d.rs", "four", "x y", 0.6),
        ];
        for chunk in &mut code {
            chunk.text = None;
        }
        let mut selector = Selector::new(8, false);
        let mut reads = Vec::new();
        loop {
            match selector.advance([&code, &[]], &Words).unwrap() {
                Step::Done => break,
                Step::Needs { side, from } => {
                    assert_eq!(side, Side::Code);
                    reads.push(from);
                    code[from].text = Some(Ok("x y".to_owned()));
                }
            }
        }
        // Two units of four tokens fill the budget; the third is read, since
        // its cost decides that it does not fit, and the fourth never is.
        assert_eq!(reads, [0, 1, 2]);
        let [groups, _] = selector.finish();
        assert_eq!(groups.len(), 2);
    }

    #[test]
    fn files_only_never_asks_for_text() {
        let mut code = [ranked("a.rs", "one", "x y", 0.9)];
        code[0].text = None;
        let mut selector = Selector::new(100, true);
        assert_eq!(selector.advance([&code, &[]], &Words).unwrap(), Step::Done);
        assert_eq!(selector.finish()[0].len(), 1);
    }

    #[test]
    fn an_unreadable_unit_is_an_error_only_when_the_selection_reaches_it() {
        let mut code = [
            ranked("a.rs", "one", "x y", 0.9),
            ranked("b.rs", "big", "w w w w w w w w w w", 0.8),
            ranked("c.rs", "gone", "x", 0.7),
        ];
        code[2].text = Some(Err(ReadError::Stale {
            path: "c.rs".to_owned(),
            start: 3,
            end: 9,
        }));
        // The second unit does not fit and ends the list: the third is never
        // reached.
        let mut selector = Selector::new(8, false);
        assert_eq!(selector.advance([&code, &[]], &Words).unwrap(), Step::Done);
        // With room for the second, the selection reaches the third.
        let mut selector = Selector::new(100, false);
        assert!(matches!(
            selector.advance([&code, &[]], &Words),
            Err(ReadError::Stale { .. })
        ));
    }

    #[test]
    fn symbols_join_distinct_member_symbols_and_headings_drop_the_path() {
        let mut chunk = ranked("a.rs", "one", "x", 0.1);
        chunk.row.members.push(MemberRow {
            bytes: 1..2,
            kind: "function_item".to_owned(),
            symbol: "two".to_owned(),
        });
        chunk.row.members.push(chunk.row.members[0].clone());
        assert_eq!(chunk.symbols().as_deref(), Some("one, two"));
        let d = doc("d.md", "A › B", "t", 0.1);
        assert_eq!(d.heading(), "A › B");
        assert_eq!(d.symbols(), None);
    }
}
