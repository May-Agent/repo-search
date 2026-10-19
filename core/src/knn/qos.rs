//! Scan-thread scheduling: user-interactive QoS and the default thread count.

use std::io;

/// Marks the calling thread as user-interactive work so the scheduler keeps it
/// on a performance core.
#[cfg(target_os = "macos")]
pub(super) fn set_user_interactive() {
    // SAFETY: the call only changes the calling thread's QoS class.
    let rc = unsafe {
        libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE, 0)
    };
    assert_eq!(rc, 0, "pthread_set_qos_class_self_np failed: {rc}");
}

/// Scan threads need no QoS marking off macOS.
#[cfg(not(target_os = "macos"))]
pub(super) fn set_user_interactive() {}

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

/// The number of hardware threads: the default scan thread count, which the
/// caller passes to `KnnIndex::search`.
///
/// # Errors
///
/// When the platform does not report its parallelism.
#[cfg(not(target_os = "macos"))]
pub fn performance_cores() -> io::Result<usize> {
    std::thread::available_parallelism().map(std::num::NonZero::get)
}

#[cfg(test)]
mod tests {
    #[test]
    fn reports_at_least_one_core() {
        assert!(super::performance_cores().unwrap() >= 1);
    }
}
