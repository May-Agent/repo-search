//! The `search` command-line client.

use clap::{Args, Parser, Subcommand, ValueEnum};
use std::process::ExitCode;

/// Search a repository for code you can describe but cannot name.
///
/// Exit codes follow grep: 0 results, 1 none, 2 error.
#[derive(Parser)]
#[command(
    name = "search",
    version,
    subcommand_precedence_over_arg = true,
    subcommand_negates_reqs = true
)]
struct Cli {
    /// Free-text query.
    #[arg(required = true)]
    query: Option<String>,

    #[command(flatten)]
    options: SearchOptions,

    #[command(subcommand)]
    command: Option<Command>,
}

/// Options for `search <query>`.
#[derive(Args)]
struct SearchOptions {
    /// Return results until the token budget is used.
    #[arg(long, default_value_t = 4000)]
    budget: u32,

    /// Ranked file list only.
    #[arg(short = 'l', long)]
    files: bool,

    /// Retrieval channels.
    #[arg(long, value_enum, default_value_t = Mode::Hybrid)]
    mode: Mode,

    /// Restrict paths (repeatable).
    #[arg(long)]
    path: Vec<String>,

    /// Restrict language.
    #[arg(long)]
    lang: Option<String>,

    /// Restrict syntactic kinds (fn, struct, trait, ...).
    #[arg(long, value_delimiter = ',')]
    kind: Vec<String>,

    /// Restrict symbol path, e.g. `Environment::*`.
    #[arg(long)]
    symbol: Option<String>,

    /// Include files `.gitignore` excludes.
    #[arg(long)]
    include_ignored: bool,

    /// Structured output.
    #[arg(long)]
    json: bool,
}

#[derive(Clone, ValueEnum)]
enum Mode {
    /// FTS5 BM25 and vector KNN merged by reciprocal-rank fusion.
    Hybrid,
    /// FTS5 over chunk title and text.
    Lexical,
    /// Vector KNN only.
    Semantic,
}

#[derive(Subcommand)]
enum Command {
    /// Coverage, pending chunks, daemon state.
    Status,
    /// Reconcile now.
    Index {
        /// Block until every chunk has a vector.
        #[arg(long)]
        wait: bool,
    },
}

fn main() -> ExitCode {
    let Cli {
        query: _,
        options: _,
        command: _,
    } = Cli::parse();
    eprintln!("error: the search daemon is not built yet");
    ExitCode::from(2)
}
