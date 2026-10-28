//! The token-counting interface every model tokenizer implements.

/// A unit-length embedding; only constructible by normalizing.
#[derive(Debug, Clone, PartialEq)]
pub struct UnitVector(Box<[f32]>);

impl UnitVector {
    /// Scales `v` to unit L2 length.
    ///
    /// # Panics
    ///
    /// When `v` is empty, has zero length, or holds a non-finite component: a model
    /// that produces such an output is broken, and a vector built from it would
    /// poison every similarity it takes part in.
    #[must_use]
    pub fn normalize(mut v: Vec<f32>) -> Self {
        assert!(!v.is_empty(), "cannot normalize an empty vector");
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            norm.is_finite() && norm > 0.0,
            "cannot normalize a vector of length {norm}"
        );
        let inv = norm.recip();
        for x in &mut v {
            *x *= inv;
        }
        Self(v.into_boxed_slice())
    }

    /// The components, of unit L2 length.
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }
}

/// Counts the model's tokens. The chunker enforces its cap on every chunk's
/// embedded document through this interface; implementations tokenize the
/// document [`crate::model::DOCUMENT`] formats.
pub trait TokenCounter: Send + Sync {
    /// The number of tokens `text` encodes to, special tokens included — the
    /// count the model consumes.
    fn count(&self, text: &str) -> usize;

    /// Whether the document the model embeds for `title` — its text is
    /// `text` joined — encodes to at most `max` tokens. Implementations may
    /// answer without encoding when a byte bound already decides it.
    fn fits(&self, title: &str, text: &[&str], max: usize) -> bool;

    /// An estimate of where the document the model embeds for `title` —
    /// `text` joined — runs out of tokens: the byte offset into the joined
    /// `text` the document's first `max` tokens end at, special tokens
    /// included. `None` when the document fits.
    ///
    /// This is an estimate, not a boundary: merges spanning the cut can
    /// move where the `max`-th token truly ends, so callers confirm it with
    /// [`TokenCounter::fits`].
    fn fit_end(&self, title: &str, text: &[&str], max: usize) -> Option<usize>;

    /// Tokens of the document the model embeds for `title` and `text`,
    /// special tokens included.
    fn document_tokens(&self, title: &str, text: &str) -> usize;
}

#[cfg(test)]
mod tests {
    use super::UnitVector;

    #[test]
    fn normalize_scales_to_unit_length() {
        let v = UnitVector::normalize(vec![3.0, 4.0]);
        assert_eq!(v.as_slice(), &[0.6, 0.8]);
    }

    #[test]
    #[should_panic(expected = "length 0")]
    fn normalize_rejects_zero() {
        let _ = UnitVector::normalize(vec![0.0, 0.0]);
    }

    #[test]
    #[should_panic(expected = "empty")]
    fn normalize_rejects_empty() {
        let _ = UnitVector::normalize(Vec::new());
    }
}
