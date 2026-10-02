//! Incremental and batch stop sequence matcher for streaming/local SLM decoders.
//!
//! Synthesized from `apfel-rs/src/core/stop_matcher.rs`.
//!
//! Buffers streaming tokens, detects partial stop sequence prefixes, and immediately halts
//! decoding when a candidate delimiter (e.g. `\n`, `}`, `Option:`) is hit.

use crate::error::{Result, ZevError};

/// Result returned from feeding an incremental chunk to the stop matcher.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopMatchResult {
    /// Safe chunk to emit immediately.
    Emit(String),
    /// A stop sequence was encountered. `emitted` contains any text prior
    /// to the sequence. The stop sequence itself is excluded.
    Matched {
        emitted: String,
        matched_seq: String,
    },
    /// Text is temporarily held back because it partially matches a stop sequence prefix.
    Holding,
}

/// An incremental, streaming stop-sequence matcher that holds back ambiguous suffixes
/// and terminates generation immediately upon matching a delimiter.
#[derive(Debug, Clone)]
pub struct StopSequenceMatcher {
    sequences: Vec<String>,
    buffer: String,
    matched: bool,
}

impl StopSequenceMatcher {
    pub fn new<I, S>(sequences: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let seqs: Vec<String> = sequences.into_iter().map(Into::into).collect();
        for s in &seqs {
            if s.is_empty() {
                return Err(ZevError::Internal(
                    "Stop sequence cannot be empty".to_string(),
                ));
            }
        }
        Ok(Self {
            sequences: seqs,
            buffer: String::new(),
            matched: false,
        })
    }

    pub fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }

    pub fn is_matched(&self) -> bool {
        self.matched
    }

    /// Process an incoming chunk of generated text.
    pub fn feed(&mut self, chunk: &str) -> StopMatchResult {
        if self.matched {
            return StopMatchResult::Holding;
        }

        if self.sequences.is_empty() {
            return StopMatchResult::Emit(chunk.to_string());
        }

        self.buffer.push_str(chunk);

        // Check for full match in current buffer
        for seq in &self.sequences {
            if let Some(pos) = self.buffer.find(seq) {
                self.matched = true;
                let emitted = self.buffer[..pos].to_string();
                let matched_seq = seq.clone();
                self.buffer.clear();
                return StopMatchResult::Matched {
                    emitted,
                    matched_seq,
                };
            }
        }

        // Check for partial match at the tail of the buffer
        let max_seq_len = self.sequences.iter().map(|s| s.len()).max().unwrap_or(0);
        let check_len = max_seq_len.min(self.buffer.len());
        let tail_start = self.buffer.len() - check_len;
        let tail = &self.buffer[tail_start..];

        for i in 0..tail.len() {
            let candidate_prefix = &tail[i..];
            for seq in &self.sequences {
                if seq.starts_with(candidate_prefix) {
                    let emit_len = tail_start + i;
                    if emit_len > 0 {
                        let to_emit = self.buffer[..emit_len].to_string();
                        self.buffer = self.buffer[emit_len..].to_string();
                        return StopMatchResult::Emit(to_emit);
                    }
                    return StopMatchResult::Holding;
                }
            }
        }

        // No partial matches: safe to emit buffer
        let to_emit = std::mem::take(&mut self.buffer);
        StopMatchResult::Emit(to_emit)
    }

    /// Flushes any remaining held text at end of stream.
    pub fn flush(&mut self) -> Option<String> {
        if self.buffer.is_empty() || self.matched {
            None
        } else {
            Some(std::mem::take(&mut self.buffer))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stop_matcher_exact_split() {
        let mut matcher = StopSequenceMatcher::new(vec!["\n\n", "User:"]).unwrap();

        let r1 = matcher.feed("Decision: appendicitis");
        assert_eq!(r1, StopMatchResult::Emit("Decision: appendicitis".into()));

        let r2 = matcher.feed("\n\nExtra rationale");
        assert_eq!(
            r2,
            StopMatchResult::Matched {
                emitted: "".into(),
                matched_seq: "\n\n".into(),
            }
        );
    }

    #[test]
    fn test_stop_matcher_partial_hold() {
        let mut matcher = StopSequenceMatcher::new(vec!["</decision>"]).unwrap();

        let r1 = matcher.feed("choice: billing </dec");
        assert_eq!(r1, StopMatchResult::Emit("choice: billing ".into()));

        let r2 = matcher.feed("ision> trailing");
        assert_eq!(
            r2,
            StopMatchResult::Matched {
                emitted: "".into(),
                matched_seq: "</decision>".into(),
            }
        );
    }
}
