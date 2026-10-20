//! Host headroom for the GPU instance: mains power and a cool-enough SoC.

use objc2_foundation::{NSProcessInfo, NSProcessInfoThermalState};
use objc2_io_kit::{
    IOPSCopyPowerSourcesInfo, IOPSGetProvidingPowerSourceType, kIOPMACPowerKey, kIOPMUPSPowerKey,
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
