//! The host chip's class, read once at daemon startup.
//!
//! `machdep.cpu.brand_string` names the Apple silicon the server runs on:
//! `Apple M<n>` is the base class and `Apple M<n> Pro`, `Max` and `Ultra`
//! the wide one. On a base chip a GPU instance executing alongside the ANE
//! makes the normal queue finish later — shared memory bandwidth is the leading
//! explanation — so the scheduler keeps stage-1 work off it there. A brand
//! the parse does not recognise is a startup error, not a guess: falling
//! back would schedule on a measurement that was never made.

use std::io;

/// The chip's class, parsed from its brand string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChipClass {
    /// `Apple M<n>`: the GPU instance takes no stage-1 work and joins the
    /// background queue once stage 1 is done.
    Base,
    /// `Apple M<n> Pro`, `Max` or `Ultra`: the GPU serves stage-1 work.
    Wide,
}

impl ChipClass {
    /// The class a `machdep.cpu.brand_string` value belongs to, or `None`
    /// for a string that is not a recognised Apple M-series one. A virtual
    /// machine reports its host chip with a ` (Virtual)` suffix and is
    /// classed by that chip.
    pub fn of_brand(brand: &str) -> Option<Self> {
        let brand = brand.strip_suffix(" (Virtual)").unwrap_or(brand);
        let rest = brand.strip_prefix("Apple M")?;
        let mut parts = rest.splitn(2, ' ');
        let generation = parts.next().expect("splitn yields a first part");
        if generation.is_empty() || !generation.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        match parts.next() {
            None => Some(Self::Base),
            Some("Pro") | Some("Max") | Some("Ultra") => Some(Self::Wide),
            Some(_) => None,
        }
    }

    /// Whether the GPU instance takes stage-1 work on this chip.
    pub fn gpu_takes_stage1(self) -> bool {
        matches!(self, Self::Wide)
    }
}

impl std::fmt::Display for ChipClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Base => "base",
            Self::Wide => "wide",
        })
    }
}

/// Why the host's chip class could not be decided.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Reading `machdep.cpu.brand_string` failed.
    #[error("machdep.cpu.brand_string: {0}")]
    Read(#[source] io::Error),
    /// The brand string is not a recognised Apple M-series one; it is
    /// named in the error.
    #[error("unrecognised chip brand {0:?}")]
    Unknown(String),
}

/// The host's chip class: the brand string is read once and the decision
/// logged — the class and whether the GPU takes stage-1 work.
///
/// # Errors
///
/// [`Error::Read`] when the sysctl fails and [`Error::Unknown`] when the
/// brand is not a recognised Apple M-series string.
pub fn host() -> Result<ChipClass, Error> {
    let brand = brand_string().map_err(Error::Read)?;
    let class = ChipClass::of_brand(&brand).ok_or_else(|| Error::Unknown(brand.clone()))?;
    tracing::info!(
        %brand,
        %class,
        gpu_stage1 = class.gpu_takes_stage1(),
        "the host's chip class"
    );
    Ok(class)
}

/// The `machdep.cpu.brand_string` sysctl value, e.g. `Apple M4`.
fn brand_string() -> io::Result<String> {
    let mut len = 0usize;
    // SAFETY: a null `oldp` writes the value's size to `len`; the name is a
    // NUL-terminated literal.
    let rc = unsafe {
        libc::sysctlbyname(
            c"machdep.cpu.brand_string".as_ptr(),
            std::ptr::null_mut(),
            &raw mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 {
        return Err(io::Error::last_os_error());
    }
    let mut buf = vec![0u8; len];
    // SAFETY: `buf` has `len` writable bytes — the size the first call
    // reported — and `len` comes back holding the bytes written.
    let rc = unsafe {
        libc::sysctlbyname(
            c"machdep.cpu.brand_string".as_ptr(),
            buf.as_mut_ptr().cast(),
            &raw mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 {
        return Err(io::Error::last_os_error());
    }
    buf.truncate(len);
    std::ffi::CStr::from_bytes_until_nul(&buf)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "unterminated brand string"))?
        .to_str()
        .map(str::to_owned)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "non-UTF-8 brand string"))
}

#[cfg(test)]
mod tests {
    use super::ChipClass;

    #[test]
    fn parses_apple_m_brand_strings() {
        for (brand, class) in [
            ("Apple M1", ChipClass::Base),
            ("Apple M1 Pro", ChipClass::Wide),
            ("Apple M2 Max", ChipClass::Wide),
            ("Apple M3 Ultra", ChipClass::Wide),
            ("Apple M4", ChipClass::Base),
            ("Apple M1 (Virtual)", ChipClass::Base),
            ("Apple M2 Pro (Virtual)", ChipClass::Wide),
        ] {
            assert_eq!(ChipClass::of_brand(brand), Some(class), "{brand}");
        }
    }

    #[test]
    fn rejects_unrecognised_brand_strings() {
        for brand in [
            "",
            "Apple M",
            "Apple Mx",
            "Apple M1x",
            "Apple M1 ",
            "Apple M1 Pro Max",
            "Apple M1 (Virtual) (Virtual)",
            "Apple M1(Virtual)",
            "Apple A14",
            "Intel(R) Core(TM) i7-10700K CPU @ 3.80GHz",
        ] {
            assert_eq!(ChipClass::of_brand(brand), None, "{brand}");
        }
    }
}
