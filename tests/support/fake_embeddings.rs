//! `FakeEmbeddings`: a gated, recording [`Embeddings`] impl for the
//! daemon-scheduler tests.
//!
//! Every call parses its inputs back into documents or a query and sends
//! a `Started` [`Record`] per input on the `std::sync::mpsc` the test
//! holds. It then waits out the gates installed for it, and records each
//! input's embedding just before it answers deterministic vectors; a call
//! dropped unanswered records `Cancelled` instead. Gates are async
//! latches keyed by what a call carries, never by call order, which is
//! not deterministic across a multi-thread runtime. An aborted request is
//! cancelled at its gate and records no embedding — once it parked there:
//! a task aborted while it runs stops only at its next yield, so a test
//! waits for `Cancelled` before opening that gate. Tests wait on records,
//! gates and scheduler state, never on sleeps.

use std::collections::HashMap;
use std::sync::mpsc;

use repo_search::daemon::scheduler_http::QueryInput;
use repo_search::daemon::server::{Answered, Embeddings};
use repo_search_core::UnitVector;
use repo_search_core::embed_api::{ServerTiming, Urgency};
use repo_search_core::model::{DOCUMENT, QueryTask};
use repo_search_core::store::ContentHash;
use tokio::sync::watch;

/// The width of the fake vectors.
const DIM: usize = 64;

/// What a [`Record`] describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// A query embedding (`u=0`, `task:` input).
    Query,
    /// A document embedded as query work (`u=0`, `title:` input).
    QueryDocument,
    /// A document embedded as indexing work (`u>0`).
    Index,
    /// An embed call began, recorded before its gates.
    Started,
    /// An embed call was dropped unanswered — its request task was
    /// aborted — recorded as the call's future is dropped.
    Cancelled,
}

/// One input of one embed call.
#[derive(Debug)]
pub struct Record {
    /// What the input was.
    pub kind: Kind,
    /// The call's `Priority` urgency.
    pub urgency: Urgency,
    /// The document's content hash; `None` for queries.
    pub hash: Option<ContentHash>,
    /// The inputs in the call: 1 for flights and queries, more for a
    /// `Documents` job.
    pub batch_len: usize,
}

/// The fake's failure; scheduler-level tests inject it with `fail`.
#[derive(Debug, thiserror::Error)]
#[error("fake embeddings failure: {0}")]
pub struct FakeError(String);

/// A [`QueryInput`] that never truncates: the stub counts fit every test
/// query under the model's cap.
pub struct StubQueries;

impl QueryInput for StubQueries {
    fn input(&self, task: QueryTask, query: &str) -> String {
        task.prompt(query)
    }
}

/// A closed latch the test opens; every call waiting on it then runs.
pub struct Gate(watch::Sender<bool>);

impl Gate {
    fn new() -> (Self, watch::Receiver<bool>) {
        let (tx, rx) = watch::channel(false);
        (Self(tx), rx)
    }

    /// Opens the latch, for every waiting and later call.
    pub fn open(&self) {
        self.0.send_replace(true);
    }

    /// Closes the latch again for later calls. A call already past it is
    /// unaffected; one still waiting keeps waiting, so the test closes a
    /// gate only once the call it opened for has recorded its embedding.
    pub fn close(&self) {
        self.0.send_replace(false);
    }
}

/// Waits until `latch` opens. A test that ended dropped its gates; the
/// call then runs on.
async fn pass(latch: Option<&watch::Receiver<bool>>) {
    if let Some(latch) = latch {
        let _ = latch.clone().wait_for(|open| *open).await;
    }
}

/// The test-side [`Embeddings`]. An indexing (`u>0`) call waits on the
/// gate of its document's hash, then on the indexing gate; an interactive
/// (`u=0`) call waits on the interactive gate. Gates are installed before
/// the scheduler takes the fake.
pub struct FakeEmbeddings {
    records: mpsc::Sender<Record>,
    hash_gates: HashMap<ContentHash, watch::Receiver<bool>>,
    index_gate: Option<watch::Receiver<bool>>,
    interactive_gate: Option<watch::Receiver<bool>>,
    failures: HashMap<ContentHash, String>,
}

impl FakeEmbeddings {
    /// A fake whose calls answer at once.
    pub fn new(records: mpsc::Sender<Record>) -> Self {
        Self {
            records,
            hash_gates: HashMap::new(),
            index_gate: None,
            interactive_gate: None,
            failures: HashMap::new(),
        }
    }

    /// Gates every indexing call of the document `hash`.
    pub fn gate(&mut self, hash: ContentHash) -> Gate {
        let (gate, rx) = Gate::new();
        self.hash_gates.insert(hash, rx);
        gate
    }

    /// Gates every indexing call — urgency above 0.
    pub fn gate_index(&mut self) -> Gate {
        let (gate, rx) = Gate::new();
        self.index_gate = Some(rx);
        gate
    }

    /// Gates every interactive call — urgency 0.
    pub fn gate_interactive(&mut self) -> Gate {
        let (gate, rx) = Gate::new();
        self.interactive_gate = Some(rx);
        gate
    }

    /// Makes every indexing call of the document `hash` fail with
    /// `message` once its gates open.
    pub fn fail(&mut self, hash: ContentHash, message: &str) {
        self.failures.insert(hash, message.to_owned());
    }

    /// `input`'s document title and text, parsed back out of
    /// [`DOCUMENT::format`], or `None` for a query prompt.
    fn document(input: &str) -> Option<(&str, &str)> {
        let rest = input.strip_prefix(DOCUMENT.before)?;
        let (title, text) = rest.split_once(DOCUMENT.between)?;
        Some((title, &text[..text.len() - DOCUMENT.after.len()]))
    }

    /// A deterministic unit vector, as `tests/support/stub.rs` builds.
    fn vector(text: &str) -> UnitVector {
        let mut v = vec![0.0f32; DIM];
        for word in text.split_whitespace() {
            let mut hash = 0xcbf2_9ce4_8422_2325u64;
            for byte in word.as_bytes() {
                hash ^= u64::from(*byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
            v[hash as usize % DIM] += 1.0;
        }
        UnitVector::normalize(v)
    }

    fn record(&self, record: Record) {
        let _ = self.records.send(record);
    }
}

/// Records `Cancelled` for each input of a call dropped before it
/// answered; disarmed once the call has its answer.
struct Unanswered<'a> {
    fake: &'a FakeEmbeddings,
    urgency: Urgency,
    hashes: Vec<Option<ContentHash>>,
    armed: bool,
}

impl Drop for Unanswered<'_> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        for &hash in &self.hashes {
            self.fake.record(Record {
                kind: Kind::Cancelled,
                urgency: self.urgency,
                hash,
                batch_len: self.hashes.len(),
            });
        }
    }
}

impl Embeddings for FakeEmbeddings {
    type Error = FakeError;

    async fn embed(&self, inputs: Vec<String>, urgency: Urgency) -> Result<Answered, FakeError> {
        let interactive = urgency.get() == 0;
        let parsed: Vec<(Kind, Option<ContentHash>)> = inputs
            .iter()
            .map(|input| match Self::document(input) {
                Some((title, text)) => {
                    let kind = if interactive {
                        Kind::QueryDocument
                    } else {
                        Kind::Index
                    };
                    (kind, Some(ContentHash::of_chunk(title, text)))
                }
                None => (Kind::Query, None),
            })
            .collect();
        for &(_, hash) in &parsed {
            self.record(Record {
                kind: Kind::Started,
                urgency,
                hash,
                batch_len: inputs.len(),
            });
        }
        let mut unanswered = Unanswered {
            fake: self,
            urgency,
            hashes: parsed.iter().map(|&(_, hash)| hash).collect(),
            armed: true,
        };
        if interactive {
            pass(self.interactive_gate.as_ref()).await;
        } else {
            for &(_, hash) in &parsed {
                if let Some(hash) = hash {
                    pass(self.hash_gates.get(&hash)).await;
                }
            }
            pass(self.index_gate.as_ref()).await;
            unanswered.armed = false;
            for &(_, hash) in &parsed {
                if let Some(message) = hash.and_then(|hash| self.failures.get(&hash)) {
                    return Err(FakeError(message.clone()));
                }
            }
        }
        unanswered.armed = false;
        for &(kind, hash) in &parsed {
            self.record(Record {
                kind,
                urgency,
                hash,
                batch_len: inputs.len(),
            });
        }
        Ok(Answered {
            vectors: inputs.iter().map(|input| Self::vector(input)).collect(),
            timing: ServerTiming::default(),
        })
    }
}
