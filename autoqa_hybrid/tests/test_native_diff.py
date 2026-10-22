"""Tests for the Rust vision diffing engine via the Python bindings."""
import math

import numpy as np
import pytest

from autoqa import native


def solid(w, h, rgba):
    return np.full((h, w, 4), rgba, dtype=np.uint8)


class TestDiffScreenshots:
    def test_identical_images_have_zero_diffs(self):
        img = solid(64, 64, (10, 20, 30, 255))
        result = native.diff_screenshots(img.tobytes(), img.tobytes(), 64, 64, 0.0)
        assert result.changed_pixels == 0
        assert result.pixel_ratio == 0.0
        assert result.bbox is None

    def test_single_changed_pixel_bbox(self):
        before = solid(4, 4, (0, 0, 0, 255))
        after = before.copy()
        after[1, 2, 0] = 255  # (x=2, y=1), channel R
        result = native.diff_screenshots(
            before.tobytes(), after.tobytes(), 4, 4, 0.0
        )
        assert result.changed_pixels == 1
        assert result.bbox == (2, 1, 2, 1)
        assert result.pixel_ratio == pytest.approx(1 / 16)

    def test_bbox_spans_multiple_regions(self):
        before = solid(8, 8, (0, 0, 0, 255))
        after = before.copy()
        after[0, 0, 0] = 255   # (0, 0)
        after[7, 6, 2] = 255   # (6, 7)
        result = native.diff_screenshots(
            before.tobytes(), after.tobytes(), 8, 8, 0.0
        )
        assert result.changed_pixels == 2
        assert result.bbox == (0, 0, 6, 7)

    def test_threshold_absorbs_small_deltas(self):
        before = solid(8, 8, (100, 100, 100, 255))
        after = solid(8, 8, (96, 100, 100, 255))
        assert (
            native.diff_screenshots(before.tobytes(), after.tobytes(), 8, 8, 3.0).changed_pixels
            == 64
        )
        assert (
            native.diff_screenshots(before.tobytes(), after.tobytes(), 8, 8, 4.0).changed_pixels
            == 0
        )

    def test_size_mismatch_raises_value_error(self):
        img = solid(4, 4, (0, 0, 0, 255))
        with pytest.raises(ValueError, match="RGBA buffers"):
            native.diff_screenshots(img.tobytes(), img.tobytes(), 2, 4, 0.0)

    def test_zero_dimensions_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            native.diff_screenshots(b"", b"", 0, 0, 0.0)


class TestSimilarityRatio:
    def test_identical_is_one(self):
        img = solid(4, 4, (0, 0, 0, 255))
        assert native.similarity_ratio(img.tobytes(), img.tobytes(), 0.0) == 1.0

    def test_inverted_is_zero(self):
        a = solid(4, 4, (0, 0, 0, 255))
        b = solid(4, 4, (255, 255, 255, 255))
        assert native.similarity_ratio(a.tobytes(), b.tobytes(), 0.0) == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            native.similarity_ratio(b"\x00" * 8, b"\x00" * 4, 0.0)


class TestNativeModuleIntro:
    def test_module_exposes_expected_surface(self):
        assert hasattr(native, "diff_screenshots")
        assert hasattr(native, "similarity_ratio")
        assert hasattr(native, "hash_tokens")
        assert hasattr(native, "dom_delta")
        info = native._native_info
        assert info.name == "autoqa-native-core"
        assert info.version == "0.1.0"
