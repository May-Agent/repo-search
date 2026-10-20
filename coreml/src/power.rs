//! Host headroom for the GPU instance: mains power and a cool-enough SoC.

use std::ffi::c_void;
use std::io;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::mpsc;
use std::thread::JoinHandle;

use block2::RcBlock;
use objc2_core_foundation::{CFRetained, CFRunLoop, CFType, kCFRunLoopDefaultMode};
use objc2_foundation::{
    NSNotification, NSNotificationCenter, NSProcessInfo, NSProcessInfoThermalState,
    NSProcessInfoThermalStateDidChangeNotification,
};
use objc2_io_kit::{
    IOPSCopyPowerSourcesInfo, IOPSGetProvidingPowerSourceType, IOPSNotificationCreateRunLoopSource,
    kIOPMACPowerKey, kIOPMUPSPowerKey,
};

/// Why a [`ComputeUnits::CpuAndGpu`][crate::ComputeUnits] embedder may not be
/// created on this host right now.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum GpuBlocked {
    /// The providing power source is not mains power.
    #[error("power source {0:?} is not mains power")]
    NotMains(String),
    /// `NSProcessInfo.thermalState` is serious or critical: the SoC is already
    /// throttling and the GPU instance would throttle it further.
    #[error("thermal state {0:?} is serious or worse")]
    ThermalPressure(NSProcessInfoThermalState),
    /// `IOPSCopyPowerSourcesInfo` or `IOPSGetProvidingPowerSourceType` returned
    /// nothing; treated as not permitted rather than guessing.
    #[error("IOPowerSources could not name the providing power source")]
    PowerSourceUnknown,
}

/// `Ok(())` when a GPU embedder may be constructed: the machine is on mains
/// power (AC or UPS) and `thermalState` is below serious.
pub fn gpu_headroom() -> Result<(), GpuBlocked> {
    let state = NSProcessInfo::processInfo().thermalState();
    if state >= NSProcessInfoThermalState::Serious {
        return Err(GpuBlocked::ThermalPressure(state));
    }
    let source = power_source().ok_or(GpuBlocked::PowerSourceUnknown)?;
    for mains in [kIOPMACPowerKey, kIOPMUPSPowerKey] {
        if source == mains.to_str().expect("IOPS keys are ASCII") {
            return Ok(());
        }
    }
    Err(GpuBlocked::NotMains(source))
}

/// The providing power source name (`"AC Power"`, `"Battery Power"`, ...).
fn power_source() -> Option<String> {
    let blob = IOPSCopyPowerSourcesInfo()?;
    // SAFETY: `blob` is the snapshot IOPSCopyPowerSourcesInfo just returned.
    let source = unsafe { IOPSGetProvidingPowerSourceType(Some(&*blob)) }?;
    Some(source.to_string())
}

/// Called with the current [`gpu_headroom`] whenever it may have changed.
type OnChange = Box<dyn Fn(Result<(), GpuBlocked>) + Send + Sync>;

/// A run loop that may be stopped from another thread.
struct RunLoop(CFRetained<CFRunLoop>);

// SAFETY: `CFRunLoopStop` is documented as callable from any thread, and
// `stop` is the only method used on another thread.
unsafe impl Send for RunLoop {}

/// Reports [`gpu_headroom`] changes: power-source changes through IOKit and
/// thermal-state changes through `NSProcessInfo`, both event-driven.
///
/// Dropping the monitor removes both observers, stops the run loop and joins
/// its thread.
pub struct HeadroomMonitor {
    run_loop: RunLoop,
    thread: Option<JoinHandle<()>>,
}

impl HeadroomMonitor {
    /// Starts the monitor thread `repo-search-headroom`. `on_change` is
    /// called once on that thread with the current headroom, then on every
    /// power-source or thermal-state change; the thermal path runs on the
    /// thread that posts the notification, so it may be any thread.
    ///
    /// Returns once the thread's run loop is running.
    ///
    /// # Errors
    ///
    /// The thread could not be spawned, or IOKit refused the power-source
    /// notification source.
    pub fn start(on_change: OnChange) -> io::Result<Self> {
        let (ready_tx, ready_rx) = mpsc::channel();
        let thread = std::thread::Builder::new()
            .name("repo-search-headroom".to_owned())
            .spawn(move || monitor_thread(Arc::new(on_change), &ready_tx))?;
        match ready_rx.recv() {
            Ok(Ok(run_loop)) => Ok(Self {
                run_loop,
                thread: Some(thread),
            }),
            Ok(Err(error)) => {
                thread.join().expect("headroom thread panicked");
                Err(error)
            }
            Err(_) => match thread.join() {
                Err(panic) => std::panic::resume_unwind(panic),
                Ok(()) => unreachable!("the headroom thread reports before it exits"),
            },
        }
    }
}

impl Drop for HeadroomMonitor {
    fn drop(&mut self) {
        self.run_loop.0.stop();
        let thread = self.thread.take().expect("joined only here");
        if let Err(panic) = thread.join()
            && !std::thread::panicking()
        {
            std::panic::resume_unwind(panic);
        }
    }
}

/// `IOPSNotificationCreateRunLoopSource` callback; `context` is the
/// `OnChange` the monitor thread keeps alive until the source is removed.
unsafe extern "C-unwind" fn power_source_changed(context: *mut c_void) {
    // SAFETY: `context` is `Arc::as_ptr` of the thread's `Arc<OnChange>`,
    // which outlives the run loop source.
    let on_change = unsafe { &*context.cast::<OnChange>() };
    on_change(gpu_headroom());
}

/// The monitor thread: reports once, registers both observers, then runs its
/// run loop until the monitor stops it. `ready` carries the run loop back
/// from inside the running loop, so a stop can never precede the run.
fn monitor_thread(on_change: Arc<OnChange>, ready: &mpsc::Sender<io::Result<RunLoop>>) {
    on_change(gpu_headroom());
    let run_loop = CFRunLoop::current().expect("a thread has a run loop");
    // SAFETY: `power_source_changed` matches the callback type, and the
    // context outlives the source: it is removed below, before `on_change`
    // is dropped.
    let source = unsafe {
        IOPSNotificationCreateRunLoopSource(
            Some(power_source_changed),
            Arc::as_ptr(&on_change).cast_mut().cast(),
        )
    };
    let Some(source) = source else {
        ready
            .send(Err(io::Error::other(
                "IOPSNotificationCreateRunLoopSource returned no source",
            )))
            .expect("start receives the report");
        return;
    };
    // SAFETY: a CoreFoundation constant, initialised before any thread runs.
    let mode = unsafe { kCFRunLoopDefaultMode }.expect("the default run loop mode exists");
    run_loop.add_source(Some(&source), Some(mode));
    let thermal = {
        let on_change = Arc::clone(&on_change);
        let block = RcBlock::new(move |_: NonNull<NSNotification>| on_change(gpu_headroom()));
        // SAFETY: the name is the `NSProcessInfo` thermal notification, no
        // object or queue filters it, and the block is `Fn + Send + Sync`
        // through `OnChange`.
        unsafe {
            NSNotificationCenter::defaultCenter().addObserverForName_object_queue_usingBlock(
                Some(NSProcessInfoThermalStateDidChangeNotification),
                None,
                None,
                &block,
            )
        }
    };
    let report = RcBlock::new({
        let ready = ready.clone();
        move || {
            ready
                .send(Ok(RunLoop(
                    CFRunLoop::current().expect("a thread has a run loop"),
                )))
                .expect("start receives the report");
        }
    });
    // SAFETY: the mode is the run loop's default mode string, and the block
    // is a live `Fn()`.
    unsafe { run_loop.perform_block(Some(AsRef::<CFType>::as_ref(mode)), Some(&report)) };
    CFRunLoop::run();
    // SAFETY: `thermal` is the observer registered above.
    unsafe { NSNotificationCenter::defaultCenter().removeObserver(thermal.as_ref()) };
    run_loop.remove_source(Some(&source), Some(mode));
}
