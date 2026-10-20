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

/// Marks the calling thread with `class`.
///
/// # Panics
///
/// When the system rejects the class.
#[cfg(target_os = "macos")]
pub fn set_current_thread(class: QosClass) {
    use libc::qos_class_t as Raw;
    let raw = match class {
        QosClass::UserInteractive => Raw::QOS_CLASS_USER_INTERACTIVE,
        QosClass::UserInitiated => Raw::QOS_CLASS_USER_INITIATED,
        QosClass::Utility => Raw::QOS_CLASS_UTILITY,
        QosClass::Background => Raw::QOS_CLASS_BACKGROUND,
    };
    // SAFETY: the call only changes the calling thread's QoS class.
    let rc = unsafe { libc::pthread_set_qos_class_self_np(raw, 0) };
    assert_eq!(rc, 0, "pthread_set_qos_class_self_np failed: {rc}");
}

/// Threads need no QoS marking off macOS.
#[cfg(not(target_os = "macos"))]
pub fn set_current_thread(_: QosClass) {}

/// The QoS class of the calling thread.
///
/// # Panics
///
/// When the system reports a class this enum does not name.
#[cfg(target_os = "macos")]
#[must_use]
pub fn current_thread() -> QosClass {
    use libc::qos_class_t as Raw;
    let mut raw = Raw::QOS_CLASS_UNSPECIFIED;
    let mut relative: libc::c_int = 0;
    // SAFETY: `raw` and `relative` are writable for the call, which reads the
    // calling thread's class.
    let rc = unsafe {
        libc::pthread_get_qos_class_np(libc::pthread_self(), &raw mut raw, &raw mut relative)
    };
    assert_eq!(rc, 0, "pthread_get_qos_class_np failed: {rc}");
    match raw {
        Raw::QOS_CLASS_USER_INTERACTIVE => QosClass::UserInteractive,
        Raw::QOS_CLASS_USER_INITIATED => QosClass::UserInitiated,
        Raw::QOS_CLASS_UTILITY => QosClass::Utility,
        Raw::QOS_CLASS_BACKGROUND => QosClass::Background,
        other => panic!("unexpected QoS class {other:?}"),
    }
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

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::{QosClass, current_thread, performance_cores, set_current_thread};

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
}
