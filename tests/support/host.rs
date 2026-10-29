//! The re-executed test hosts' stdin lifeline: a host exits the moment
//! its parent is gone. The pid-exit wait a test applies to a host is
//! `repo_search_client::launcher::kqueue::ExitWatch` on macOS.

use std::io::Read as _;

/// Ends the process the moment stdin reaches EOF. The parent holds the
/// write end and never writes, so EOF means the parent is gone — SIGKILL,
/// a `timeout` wrapper, an abort — and the host ends itself at once, the
/// same end the tests' `kill()` gives it. A read error is a broken
/// lifeline and ends the process the same way.
pub fn watch_stdin() {
    std::thread::spawn(|| {
        let mut stdin = std::io::stdin().lock();
        let mut buffer = [0u8; 512];
        while stdin.read(&mut buffer).unwrap_or(0) > 0 {}
        std::process::exit(0);
    });
}
