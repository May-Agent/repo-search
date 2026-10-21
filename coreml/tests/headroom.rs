//! `HeadroomMonitor` needs no model: it reports on start and drops cleanly.

use std::sync::mpsc;
use std::time::{Duration, Instant};

use repo_search_coreml::HeadroomMonitor;

#[test]
fn the_monitor_reports_on_start_and_drops_cleanly() {
    let (tx, rx) = mpsc::channel();
    let monitor = HeadroomMonitor::start(Box::new(move |headroom| {
        // The receiver is gone once the test has finished; a late event is
        // not an error.
        let _ = tx.send(headroom);
    }))
    .expect("the monitor starts");
    let _headroom = rx
        .recv_timeout(Duration::from_secs(1))
        .expect("on_change within 1 s of start");
    let dropped = Instant::now();
    drop(monitor);
    assert!(
        dropped.elapsed() < Duration::from_secs(1),
        "drop stops the run loop and joins promptly"
    );
}
