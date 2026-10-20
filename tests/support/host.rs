//! The re-executed test hosts' lifecycle helpers: the stdin lifeline a
//! host exits on and the pid-exit wait a test applies to a host.

use std::io::Read as _;
use std::time::Duration;

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

/// Waits for `pid` to exit through a kqueue `EVFILT_PROC`/`NOTE_EXIT`
/// registration — event-driven, not polled. Returns `false` on timeout.
pub fn wait_for_exit(pid: u32, timeout: Duration) -> bool {
    // SAFETY: kqueue() has no preconditions.
    let kq = unsafe { libc::kqueue() };
    assert!(
        kq >= 0,
        "kqueue failed: {}",
        std::io::Error::last_os_error()
    );
    let change = libc::kevent {
        ident: pid as libc::uintptr_t,
        filter: libc::EVFILT_PROC,
        flags: libc::EV_ADD | libc::EV_ONESHOT,
        fflags: libc::NOTE_EXIT,
        data: 0,
        udata: std::ptr::null_mut(),
    };
    let mut event = libc::kevent {
        ident: 0,
        filter: 0,
        flags: 0,
        fflags: 0,
        data: 0,
        udata: std::ptr::null_mut(),
    };
    let deadline = libc::timespec {
        tv_sec: timeout.as_secs() as libc::time_t,
        tv_nsec: libc::c_long::from(timeout.subsec_nanos()),
    };
    // SAFETY: `kq` is a live kqueue fd; `change` and `event` name valid
    // kevent structs of the counts given.
    let n = unsafe { libc::kevent(kq, &change, 1, &mut event, 1, &deadline) };
    // SAFETY: `kq` is owned by this call.
    unsafe { libc::close(kq) };
    match n {
        1 => {
            if event.flags & libc::EV_ERROR != 0 {
                // Registration failed after the process went away.
                return event.data == libc::ESRCH as libc::intptr_t;
            }
            true
        }
        0 => false,
        _ => match std::io::Error::last_os_error().raw_os_error() {
            // The process exited before the registration: that is the event.
            Some(libc::ESRCH) => true,
            other => panic!("kevent failed: {other:?}"),
        },
    }
}
