//! Thread scheduling: quality-of-service classes and the default thread count.

#[cfg(target_os = "macos")]
use std::io;

/// A thread's quality-of-service class, from the most to the least urgent.
/// The scheduler keeps the first on performance cores and may confine the last
/// to efficiency cores.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QosClass {
    /// Work the user is waiting on.
    UserInteractive,
    /// Work the user started and expects soon.
    UserInitiated,
    /// Long-running work with a visible progress indicator.
    Utility,
    /// Work the user is not aware of.
    Background,
}

/// The `qos_class_t` naming `class`.
#[cfg(target_os = "macos")]
fn raw_class(class: QosClass) -> libc::qos_class_t {
    use libc::qos_class_t as Raw;
    match class {
        QosClass::UserInteractive => Raw::QOS_CLASS_USER_INTERACTIVE,
        QosClass::UserInitiated => Raw::QOS_CLASS_USER_INITIATED,
        QosClass::Utility => Raw::QOS_CLASS_UTILITY,
        QosClass::Background => Raw::QOS_CLASS_BACKGROUND,
    }
}

/// Marks the calling thread with `class`.
///
/// # Panics
///
/// When the system rejects the class.
#[cfg(target_os = "macos")]
pub fn set_current_thread(class: QosClass) {
    // SAFETY: the call only changes the calling thread's QoS class.
    let rc = unsafe { libc::pthread_set_qos_class_self_np(raw_class(class), 0) };
    assert_eq!(rc, 0, "pthread_set_qos_class_self_np failed: {rc}");
}

/// A thread's identifier, so another thread can start a QoS
/// [`Override`] on it.
#[cfg(target_os = "macos")]
#[derive(Debug, Clone, Copy)]
pub struct ThreadId(libc::pthread_t);

#[cfg(target_os = "macos")]
impl ThreadId {
    /// The calling thread's identifier.
    #[must_use]
    pub fn current() -> Self {
        // SAFETY: `pthread_self` always succeeds.
        Self(unsafe { libc::pthread_self() })
    }
}

// SAFETY: the identifier is opaque; the pthread calls that take it accept it
// from any thread while the thread it names is alive.
#[cfg(target_os = "macos")]
unsafe impl Send for ThreadId {}

// SAFETY: sharing the identifier is sharing a read-only pthread_t; the same
// calls take it from any thread.
#[cfg(target_os = "macos")]
unsafe impl Sync for ThreadId {}

/// The opaque token `pthread_override_qos_class_end_np` takes.
#[cfg(target_os = "macos")]
type RawOverride = *mut std::ffi::c_void;

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn pthread_override_qos_class_start_np(
        thread: libc::pthread_t,
        class: libc::qos_class_t,
        relpri: libc::c_int,
    ) -> RawOverride;
    fn pthread_override_qos_class_end_np(override_: RawOverride) -> libc::c_int;
}

/// A QoS override held on another thread: the thread's effective class is
/// raised to `class` until the value is dropped. This is the platform's
/// answer to a priority inversion, when urgent work queues behind a
/// lower-class thread.
///
/// The override must not outlive the thread it names.
#[cfg(target_os = "macos")]
pub struct Override {
    raw: RawOverride,
}

// SAFETY: the token is an opaque handle the `*_np` calls accept from any
// thread.
#[cfg(target_os = "macos")]
unsafe impl Send for Override {}

#[cfg(target_os = "macos")]
impl Override {
    /// Starts overriding `thread`'s QoS to `class`; the override ends on
    /// drop.
    ///
    /// # Panics
    ///
    /// When the system refuses the override.
    #[cfg(target_os = "macos")]
    #[must_use]
    pub fn start(thread: ThreadId, class: QosClass) -> Self {
        // SAFETY: `thread` names a live pthread, as the caller guarantees.
        let raw = unsafe { pthread_override_qos_class_start_np(thread.0, raw_class(class), 0) };
        assert!(!raw.is_null(), "pthread_override_qos_class_start_np failed");
        Self { raw }
    }
}

#[cfg(target_os = "macos")]
impl Drop for Override {
    fn drop(&mut self) {
        // SAFETY: `raw` is a live token from `start`, ended exactly once here.
        let rc = unsafe { pthread_override_qos_class_end_np(self.raw) };
        assert_eq!(rc, 0, "pthread_override_qos_class_end_np failed: {rc}");
    }
}

/// Threads need no QoS marking off macOS.
#[cfg(not(target_os = "macos"))]
pub fn set_current_thread(_: QosClass) {}

/// The QoS class the calling thread requested. An override another thread
/// holds on it is not reflected; [`scheduling_priority`] shows that.
///
/// # Panics
///
/// When the system reports a class this enum does not name.
#[cfg(target_os = "macos")]
#[must_use]
pub fn current_thread() -> QosClass {
    read_thread(ThreadId::current())
}

/// The number of performance cores: the default scan thread count, which the
/// caller passes to `KnnIndex::search`.
///
/// # Errors
///
/// When the platform does not report performance cores.
#[cfg(target_os = "macos")]
pub fn performance_cores() -> io::Result<usize> {
    let mut count: libc::c_int = 0;
    let mut len = size_of::<libc::c_int>();
    // SAFETY: `count` and `len` describe a writable c_int; the name is NUL-terminated.
    let rc = unsafe {
        libc::sysctlbyname(
            c"hw.perflevel0.physicalcpu".as_ptr(),
            (&raw mut count).cast(),
            &raw mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 {
        return Err(io::Error::last_os_error());
    }
    usize::try_from(count).map_err(|_| io::Error::other("negative performance core count"))
}

/// The QoS class `thread` requested, without overrides.
#[cfg(target_os = "macos")]
fn read_thread(thread: ThreadId) -> QosClass {
    use libc::qos_class_t as Raw;
    let mut raw = Raw::QOS_CLASS_UNSPECIFIED;
    let mut relative: libc::c_int = 0;
    // SAFETY: `raw` and `relative` are writable for the call, which reads
    // `thread`'s class.
    let rc = unsafe { libc::pthread_get_qos_class_np(thread.0, &raw mut raw, &raw mut relative) };
    assert_eq!(rc, 0, "pthread_get_qos_class_np failed: {rc}");
    match raw {
        Raw::QOS_CLASS_USER_INTERACTIVE => QosClass::UserInteractive,
        Raw::QOS_CLASS_USER_INITIATED => QosClass::UserInitiated,
        Raw::QOS_CLASS_UTILITY => QosClass::Utility,
        Raw::QOS_CLASS_BACKGROUND => QosClass::Background,
        other => panic!("unexpected QoS class {other:?}"),
    }
}

/// `thread`'s effective scheduling priority, including QoS overrides —
/// unlike [`current_thread`], this is what the scheduler actually uses.
/// The bands are the kernel's QoS-to-priority mapping: user-interactive is
/// the highest, background the lowest.
#[cfg(target_os = "macos")]
#[must_use]
pub fn scheduling_priority(thread: ThreadId) -> libc::c_int {
    // SAFETY: `thread` names a live pthread.
    let port = unsafe { libc::pthread_mach_thread_np(thread.0) };
    let mut info = std::mem::MaybeUninit::<libc::thread_extended_info>::uninit();
    let mut count = libc::THREAD_EXTENDED_INFO_COUNT;
    // SAFETY: `info` is writable for `count` integers; `port` names a live
    // thread.
    let kr = unsafe {
        libc::thread_info(
            port,
            libc::THREAD_EXTENDED_INFO as libc::thread_flavor_t,
            info.as_mut_ptr().cast::<libc::integer_t>(),
            &raw mut count,
        )
    };
    assert_eq!(kr, libc::KERN_SUCCESS, "thread_info failed: {kr}");
    // SAFETY: the kernel filled `info` for the returned count, which covers
    // `pth_curpri`.
    let info = unsafe { info.assume_init() };
    info.pth_curpri
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::{
        Override, QosClass, ThreadId, current_thread, performance_cores, read_thread,
        scheduling_priority, set_current_thread,
    };

    #[test]
    fn reports_at_least_one_core() {
        assert!(performance_cores().unwrap() >= 1);
    }

    #[test]
    fn each_class_round_trips() {
        for class in [
            QosClass::UserInteractive,
            QosClass::UserInitiated,
            QosClass::Utility,
            QosClass::Background,
        ] {
            // A thread of its own: the class sticks to the thread.
            std::thread::spawn(move || {
                set_current_thread(class);
                assert_eq!(current_thread(), class);
            })
            .join()
            .unwrap();
        }
    }

    #[test]
    fn an_override_raises_and_releases_the_target() {
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<ThreadId>();
        let (go_tx, go_rx) = std::sync::mpsc::channel::<()>();
        let worker = std::thread::spawn(move || {
            set_current_thread(QosClass::Background);
            ready_tx.send(ThreadId::current()).unwrap();
            go_rx.recv().unwrap();
        });
        let thread = ready_rx.recv().unwrap();
        assert_eq!(read_thread(thread), QosClass::Background);
        let base = scheduling_priority(thread);
        let over = Override::start(thread, QosClass::UserInteractive);
        let lifted = scheduling_priority(thread);
        eprintln!("base {base}, lifted {lifted}");
        assert!(lifted > base, "the override lifts the effective priority");
        drop(over);
        assert_eq!(
            scheduling_priority(thread),
            base,
            "ending the override restores the priority"
        );
        go_tx.send(()).unwrap();
        worker.join().unwrap();
    }
}
