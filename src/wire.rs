//! Conversions between core's types and the client crate's wire types.
//!
//! Neither crate depends on the other, so `From` cannot express these: the
//! orphan rule rejects an impl between two foreign types. Local traits carry
//! them at the daemon's boundary instead.

use repo_search_client::protocol::{GpuStatus, WireMode, WirePhase};
use repo_search_core::embed_api;
use repo_search_core::search::Mode;
use repo_search_core::store::Phase;

/// A core type with a wire counterpart.
pub trait ToWire {
    /// The wire counterpart.
    type Wire;

    /// This value on the wire.
    fn to_wire(self) -> Self::Wire;
}

/// A wire type with a core counterpart.
pub trait ToCore {
    /// The core counterpart.
    type Core;

    /// This value as core's type.
    fn to_core(self) -> Self::Core;
}

impl ToWire for Mode {
    type Wire = WireMode;

    fn to_wire(self) -> WireMode {
        match self {
            Self::Hybrid => WireMode::Hybrid,
            Self::Lexical => WireMode::Lexical,
            Self::Semantic => WireMode::Semantic,
        }
    }
}

impl ToCore for WireMode {
    type Core = Mode;

    fn to_core(self) -> Mode {
        match self {
            Self::Hybrid => Mode::Hybrid,
            Self::Lexical => Mode::Lexical,
            Self::Semantic => Mode::Semantic,
        }
    }
}

impl ToWire for Phase {
    type Wire = WirePhase;

    fn to_wire(self) -> WirePhase {
        match self {
            Self::FirstIndex => WirePhase::FirstIndex,
            Self::Maintained => WirePhase::Maintained,
        }
    }
}

impl ToWire for embed_api::GpuStatus {
    type Wire = GpuStatus;

    fn to_wire(self) -> GpuStatus {
        match self {
            Self::Absent { blocked } => GpuStatus::Absent { blocked },
            Self::Loading => GpuStatus::Loading,
            Self::Running => GpuStatus::Running,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_converts_both_ways() {
        for mode in [Mode::Hybrid, Mode::Lexical, Mode::Semantic] {
            assert_eq!(mode.to_wire().to_core(), mode);
        }
    }

    #[test]
    fn phase_and_gpu_convert() {
        assert_eq!(Phase::FirstIndex.to_wire(), WirePhase::FirstIndex);
        assert_eq!(Phase::Maintained.to_wire(), WirePhase::Maintained);
        assert_eq!(
            embed_api::GpuStatus::Absent {
                blocked: Some("why".into())
            }
            .to_wire(),
            GpuStatus::Absent {
                blocked: Some("why".into())
            }
        );
        assert_eq!(embed_api::GpuStatus::Loading.to_wire(), GpuStatus::Loading);
        assert_eq!(embed_api::GpuStatus::Running.to_wire(), GpuStatus::Running);
    }
}
