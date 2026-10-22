//! Parallelized RGBA screenshot diffing engine.
//!
//! The Python layer is responsible for decoding screenshots (e.g. via
//! Pillow) and passing raw RGBA byte buffers here — no image decoding happens
//! in Rust, which keeps the FFI surface small and fast.
//!
//! Work is sharded by pixel-row and distributed across all cores with rayon;
//! per-row scans are branch-lean so LLVM auto-vectorizes the channel loop.

use pyo3::prelude::*;
use pyo3::types::PyModule;
use rayon::prelude::*;

/// One scanline comparison result: (`changed`, `min_x`, `max_x`).
#[inline]
fn diff_row(
    before: &[u8],
    after: &[u8],
    width: usize,
    y: usize,
    threshold: u8,
) -> (u64, Option<u32>, Option<u32>) {
    let mut changed: u64 = 0;
    let mut min_x: Option<u32> = None;
    let mut max_x: Option<u32> = None;

    for x in 0..width {
        let b = y * width + x;
        let i = b * 4;
        let d_r = before[i] ^ after[i];
        let d_g = before[i + 1] ^ after[i + 1];
        let d_b = before[i + 2] ^ after[i + 2];
        let d_a = before[i + 3] ^ after[i + 3];

        // `threshold at 0` = byte-equality; otherwise a per-channel
        // difference must strictly exceed `threshold` to flag the pixel.
        let differs = if threshold == 0 {
            d_r != 0 || d_g != 0 || d_b != 0 || d_a != 0
        } else {
            (before[i] as i16 - after[i] as i16).abs() > threshold as i16
                || (before[i + 1] as i16 - after[i + 1] as i16).abs() > threshold as i16
                || (before[i + 2] as i16 - after[i + 2] as i16).abs() > threshold as i16
                || (before[i + 3] as i16 - after[i + 3] as i16).abs() > threshold as i16
        };

        if differs {
            changed += 1;
            let x32 = x as u32;
            min_x.get_or_insert(x32);
            max_x = Some(x32);
        }
    }

    (changed, min_x, max_x)
}

/// Aggregated comparison report between two equal-size RGBA screenshots.
#[pyclass]
#[derive(Clone, Debug)]
pub struct DiffResult {
    /// Number of pixels whose color changed beyond `threshold`.
    #[pyo3(get)]
    pub changed_pixels: u64,
    /// `changed_pixels / (width * height)`.
    #[pyo3(get)]
    pub pixel_ratio: f64,
    /// Inclusive bounding box (`min_x`, `min_y`, `max_x`, `max_y`) of all
    /// changed pixels, or `None` when nothing changed.
    #[pyo3(get)]
    pub bbox: Option<(u32, u32, u32, u32)>,
    #[pyo3(get)]
    pub width: u32,
    #[pyo3(get)]
    pub height: u32,
}

fn both_buffers_valid(before: &[u8], after: &[u8], width: u32, height: u32) -> PyResult<usize> {
    if width == 0 || height == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "width and height must be positive.",
        ));
    }
    let expected = (width as u64 * height as u64 * 4) as usize;

    if before.len() != expected || after.len() != expected {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "RGBA buffers must be exactly {expected} bytes (4 * {width} * {height}); \
             got before={} after={}.",
            before.len(),
            after.len(),
        )));
    }
    Ok(expected)
}
/// Compare two RGBA screenshots pixel-by-pixel across all available cores.
#[pyfunction]
pub fn diff_screenshots(
    before: Vec<u8>,
    after: Vec<u8>,
    width: u32,
    height: u32,
    threshold: f64,
) -> PyResult<DiffResult> {
    let _ = both_buffers_valid(&before, &after, width, height)?;

    let threshold = threshold as u8;
    let width_us = width as usize;

    // Rows are processed in parallel; each returns its own scanline stats.
    let totals: Vec<(u64, Option<u32>, Option<u32>)> = (0..height as usize)
        .into_par_iter()
        .map(|y| diff_row(&before, &after, width_us, y, threshold))
        .collect();

    let changed: u64 = totals.iter().map(|(c, _, _)| c).sum();
    let min_y = totals.iter().position(|(c, _, _)| *c > 0);
    let max_y = totals.iter().rposition(|(c, _, _)| *c > 0);

    let bbox = match (min_y, max_y) {
        (Some(y0), Some(y1)) if changed > 0 => {
            let min_x = totals[y0].1.unwrap_or(0);
            let max_x = totals[y0..=y1]
                .iter()
                .filter_map(|(_, _, mx)| *mx)
                .max()
                .unwrap_or(0);
            Some((min_x, y0 as u32, max_x, y1 as u32))
        }
        _ => None,
    };

    let total_pixels = (width as f64) * (height as f64);
    Ok(DiffResult {
        changed_pixels: changed,
        pixel_ratio: changed as f64 / total_pixels,
        bbox,
        width,
        height,
    })
}

/// Fast path: similarity ratio between two equal-length RGBA buffers.
#[pyfunction]
pub fn similarity_ratio(before: Vec<u8>, after: Vec<u8>, threshold: f64) -> PyResult<f64> {
    if before.len() != after.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "buffers must be the same length; got before={} after={}.",
            before.len(),
            after.len(),
        )));
    }
    if before.is_empty() {
        return Ok(1.0);
    }
    if before.len() % 4 != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "RGBA buffers must be a multiple of 4 bytes (one byte per channel).",
        ));
    }

    let threshold = threshold as i16;
    let same = before
        .par_chunks(4)
        .enumerate()
        .filter(|(i, _px)| {
            for c in 0..4 {
                let d = (before[i * 4 + c] as i16 - after[i * 4 + c] as i16).abs();
                if d > threshold {
                    return false; // pixel changed beyond threshold
                }
            }
            true
        })
        .count() as f64;
    Ok(same / (before.len() / 4) as f64)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(diff_screenshots, m)?)?;
    m.add_function(wrap_pyfunction!(similarity_ratio, m)?)?;
    m.add_class::<DiffResult>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: u32, h: u32, rgba: [u8; 4]) -> Vec<u8> {
        rgba.iter().cycle().take((w * h * 4) as usize).copied().collect()
    }

    #[test]
    fn identical_buffers_are_unchanged() {
        let img = solid(64, 64, [10, 20, 30, 255]);
        let result = diff_screenshots(img.clone(), img, 64, 64, 0.0).unwrap();
        assert_eq!(result.changed_pixels, 0);
        assert_eq!(result.pixel_ratio, 0.0);
        assert!(result.bbox.is_none());
    }

    #[test]
    fn single_changed_pixel_produces_bbox() {
        let mut after = solid(4, 4, [0, 0, 0, 255]);
        // Pixel (2, 1) → row 1: channel R goes 0 → 255.
        after[(1 * 4 + 2) * 4] = 255;
        let result = diff_screenshots(solid(4, 4, [0, 0, 0, 255]), after, 4, 4, 0.0).unwrap();
        assert_eq!(result.changed_pixels, 1);
        assert_eq!(result.bbox, Some((2, 1, 2, 1)));
    }

    #[test]
    fn threshold_absorbs_small_deltas() {
        let before = solid(8, 8, [100, 100, 100, 255]);
        let after = solid(8, 8, [96, 100, 100, 255]); // R differs by 4
        // threshold 3 → changed, threshold 4 → not changed
        assert_eq!(
            diff_screenshots(before.clone(), after.clone(), 8, 8, 3.0)
                .unwrap()
                .changed_pixels,
            64
        );
        assert_eq!(
            diff_screenshots(before, after, 8, 8, 4.0)
                .unwrap()
                .changed_pixels,
            0
        );
    }

    #[test]
    fn dimension_mismatch_is_rejected() {
        let img = solid(4, 4, [0, 0, 0, 255]);
        let err = diff_screenshots(img.clone(), img.clone(), 2, 4, 0.0).unwrap_err();
        assert!(err.to_string().contains("RGBA buffers"));
    }

    #[test]
    fn similarity_ratio_bounds() {
        let a = solid(4, 4, [0, 0, 0, 255]);
        assert_eq!(similarity_ratio(a.clone(), a.clone(), 0.0).unwrap(), 1.0);
        let b = solid(4, 4, [255, 255, 255, 255]);
        assert_eq!(similarity_ratio(a, b, 0.0).unwrap(), 0.0);
    }
}
