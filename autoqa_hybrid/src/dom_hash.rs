//! DOM-tree structural hashing and mutation-delta computation.
//!
//! Python flattens a DOM snapshot into an ordered token stream (preorder walk
//! of `tag/id/name/role/text` tokens). This module then:
//! 1. hashes the stream cheaply (FNV-1a 64) — ideal for "did the UI change?",
//! 2. computes an LCS-based delta between two streams — ideal for "what moved?".
//!
//! LCS runs only after common prefix/suffix trimming, which keeps typical
//! agent steps (one widget changed) near O(1).

use pyo3::prelude::*;
use pyo3::types::PyModule;

/// Max LCS DP table cells before we bail out to a prefix/suffix-only diff.
const MAX_DP_CELLS: u64 = 64 * 1024 * 1024;

fn fnv1a_token(digest: u64, token: &str) -> u64 {
    let mut hash = digest;
    for byte in token.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100_0000_01b3); // FNV-1a 64 prime
    }
    // Token terminator so ["ab","c"], ["a","bc"] hash differently.
    hash ^= 0xff;
    hash = hash.wrapping_mul(0x100_0000_01b3);
    hash
}

/// Stable structural hash of a DOM token stream.
///
/// Returns an FNV-1a 64-bit digest; equal token sequences always hash equal,
/// and any reorder/insert/delete changes the digest.
#[pyfunction]
pub fn hash_tokens(tokens: Vec<String>) -> u64 {
    tokens
        .iter()
        .fold(0xcbf2_9ce4_8422_2325, |digest, token| fnv1a_token(digest, token))
}

/// Delta opcode for each entry returned by [`dom_delta`].
#[pyclass]
#[derive(Clone)]
pub struct DomDelta {
    /// `0` keep, `1` replace, `2` remove, `3` insert.
    #[pyo3(get)]
    pub op: u8,
    /// Index into the *before* stream (`-1` for inserts).
    #[pyo3(get)]
    pub before_index: i64,
    /// Index into the *after* stream (`-1` for removes).
    #[pyo3(get)]
    pub after_index: i64,
    /// The after-token for replace/insert operations.
    #[pyo3(get)]
    pub token: Option<String>,
}

fn keep(b: usize, a: usize) -> DomDelta {
    DomDelta { op: 0, before_index: b as i64, after_index: a as i64, token: None }
}

/// Coalesce adjacent remove→insert (or insert→remove) pairs into `replace`
/// ops. This matches human intuition: one changed widget is one diff entry.
fn coalesce_replaces(deltas: Vec<DomDelta>) -> Vec<DomDelta> {
    let mut out: Vec<DomDelta> = Vec::with_capacity(deltas.len());
    for d in deltas {
        match d.op {
            // remove right after a pending insert → upgrade insert to replace
            2 => {
                if let Some(prev) = out.last_mut() {
                    if prev.op == 3 {
                        prev.op = 1;
                        prev.before_index = d.before_index;
                        continue;
                    }
                }
                out.push(d);
            }
            // insert right after a pending remove → upgrade remove to replace
            3 => {
                if let Some(prev) = out.last_mut() {
                    if prev.op == 2 {
                        prev.op = 1;
                        prev.after_index = d.after_index;
                        prev.token = d.token;
                        continue;
                    }
                }
                out.push(d);
            }
            _ => out.push(d),
        }
    }
    out
}

/// LCS diff between two DOM token streams.
///
/// Emitted ops walk both indices forward; `keep/replace` consume one token
/// from each stream, `remove` consumes a before-token, `insert` produces an
/// after-token.
#[pyfunction]
pub fn dom_delta(before: Vec<String>, after: Vec<String>) -> Vec<DomDelta> {
    // ---- trim common prefix / suffix (the agent hot path) ----------------
    let mut prefix = 0usize;
    let prefix_len = before.len().min(after.len());
    while prefix < prefix_len && before[prefix] == after[prefix] {
        prefix += 1;
    }

    let mut suffix = 0usize;
    let suffix_len = prefix_len - prefix;
    while suffix < suffix_len
        && before[before.len() - 1 - suffix] == after[after.len() - 1 - suffix]
    {
        suffix += 1;
    }

    let b_mid = &before[prefix..before.len() - suffix];
    let a_mid = &after[prefix..after.len() - suffix];
    let (n, m) = (b_mid.len(), a_mid.len());

    let mut deltas: Vec<DomDelta> = (0..prefix).map(|i| keep(i, i)).collect();

    if (n as u64) * (m as u64) > MAX_DP_CELLS {
        // Degenerate case: too large for DP — emit faithful remove+insert ops.
        deltas.extend(
            b_mid.iter().enumerate().map(|(i, _)| DomDelta {
                op: 2,
                before_index: (prefix + i) as i64,
                after_index: -1,
                token: None,
            }),
        );
        deltas.extend(a_mid.iter().enumerate().map(|(i, token)| DomDelta {
            op: 3,
            before_index: -1,
            after_index: (prefix + i) as i64,
            token: Some(token.clone()),
        }));
    } else {
        /*
        Classic LCS DP; the backtracking walk below emits the op stream in
        reading order so Python can render a human-readable patch.
        */
        let mut dp = vec![0u32; (n + 1) * (m + 1)];
        for i in (0..n).rev() {
            for j in (0..m).rev() {
                dp[i * (m + 1) + j] = if b_mid[i] == a_mid[j] {
                    dp[(i + 1) * (m + 1) + j + 1] + 1
                } else {
                    u32::max(
                        dp[(i + 1) * (m + 1) + j],
                        dp[i * (m + 1) + j + 1],
                    )
                };
            }
        }

        let (mut i, mut j) = (0usize, 0usize);
        while i < n && j < m {
            if b_mid[i] == a_mid[j] {
                deltas.push(keep(prefix + i, prefix + j));
                i += 1;
                j += 1;
            } else if dp[(i + 1) * (m + 1) + j] >= dp[i * (m + 1) + j + 1] {
                deltas.push(DomDelta {
                    op: 2,
                    before_index: (prefix + i) as i64,
                    after_index: -1,
                    token: None,
                });
                i += 1;
            } else {
                deltas.push(DomDelta {
                    op: 3,
                    before_index: -1,
                    after_index: (prefix + j) as i64,
                    token: Some(a_mid[j].clone()),
                });
                j += 1;
            }
        }
        deltas.extend(
            b_mid[i..].iter().enumerate().map(|(k, _)| DomDelta {
                op: 2,
                before_index: (prefix + i + k) as i64,
                after_index: -1,
                token: None,
            }),
        );
        deltas.extend(
            a_mid[j..].iter().enumerate().map(|(k, token)| DomDelta {
                op: 3,
                before_index: -1,
                after_index: (prefix + j + k) as i64,
                token: Some(token.clone()),
            }),
        );
    }

    deltas.extend(
        (0..suffix)
            .map(|s| keep(before.len() - suffix + s, after.len() - suffix + s)),
    );

    coalesce_replaces(deltas)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hash_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(dom_delta, m)?)?;
    m.add_class::<DomDelta>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn stable_hashes_for_equal_streams() {
        let a = tokens(&["div", "#btn", "submit"]);
        let b = tokens(&["div", "#btn", "submit"]);
        assert_eq!(hash_tokens(a.clone()), hash_tokens(b));
    }

    #[test]
    fn different_streams_hash_differently() {
        let a = tokens(&["div", "#btn"]);
        let b = tokens(&["div", "# CTL"]);
        assert_ne!(hash_tokens(a), hash_tokens(b));
        // order matters
        assert_ne!(
            hash_tokens(tokens(&["a", "b"])),
            hash_tokens(tokens(&["b", "a"]))
        );
        // token boundaries matter
        assert_ne!(
            hash_tokens(tokens(&["ab", "c"])),
            hash_tokens(tokens(&["a", "bc"]))
        );
    }

    #[test]
    fn simple_replace_delta() {
        let before = tokens(&["a", "b", "c"]);
        let after = tokens(&["a", "x", "c"]);
        let delta = dom_delta(before, after);
        let ops: Vec<(u8, i64, i64)> = delta
            .iter()
            .map(|d| (d.op, d.before_index, d.after_index))
            .collect();
        assert_eq!(
            ops,
            vec![(0, 0, 0), (1, 1, 1), (0, 2, 2)],
            "keep, replace, keep"
        );
        assert_eq!(delta[1].token.as_deref(), Some("x"));
    }

    #[test]
    fn insert_transition_becomes_replace_or_insert() {
        // insert in the middle: b=[a, b] → a=[a, new, b]
        let delta = dom_delta(tokens(&["a", "b"]), tokens(&["a", "new", "b"]));
        // [a, new, b] vs [a, b]: LCS pairs new ze copy; emit either
        // replace(1→1, "new") or insert(1, "new") — never a confusing pair.
        let mutations: Vec<&DomDelta> = delta.iter().filter(|d| d.op != 0).collect();
        assert_eq!(mutations.len(), 1);
        assert!(matches!(mutations[0], d if d.op == 3 && d.after_index == 1
            || d.op == 1 && d.after_index == 1 && d.token.as_deref() == Some("new")));

        // remove in the middle
        let delta = dom_delta(tokens(&["a", "gone", "b"]), tokens(&["a", "b"]));
        assert!(delta.iter().any(|d| d.op == 2 && d.before_index == 1));
    }

    #[test]
    fn huge_unchanged_stream_trims_fast() {
        let mut big = Vec::new();
        for i in 0..20_000 {
            big.push(format!("tok{i}"));
        }
        let mut other = big.clone();
        other[10] = "CHANGED".to_string();
        let digest_a = hash_tokens(big.clone());
        let digest_b = hash_tokens(other.clone());
        assert_ne!(digest_a, digest_b);

        let delta = dom_delta(big, other);
        // one widget change is one coalesced replace(-style) mutation
        assert_eq!(delta.iter().filter(|d| d.op != 0).count(), 1);
        let keep_end = delta.last().unwrap();
        assert_eq!(keep_end.op, 0);
        assert_eq!(keep_end.before_index, 19_999);
    }

    #[test]
    fn empty_streams_are_valid() {
        assert_eq!(hash_tokens(Vec::new()), 0xcbf2_9ce4_8422_2325);
        assert!(dom_delta(Vec::new(), Vec::new()).is_empty());
    }
}
