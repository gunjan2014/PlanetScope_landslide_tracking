"""
Tests for postprocessing_functions.py

Covers pure-NumPy helper functions and the core velocity calculation:
    - offset_stats_pixel
    - offset_stats_aoi
    - calc_velocity
"""

import datetime
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import postprocessing_functions as pf


# ---------------------------------------------------------------------------
# offset_stats_pixel
# ---------------------------------------------------------------------------

class TestOffsetStatsPixel:
    def _grid(self, shape=(10, 10), fill=2.0):
        return np.full(shape, fill, dtype=float)

    def test_returns_five_stats(self):
        r = self._grid()
        result = pf.offset_stats_pixel(r, xcoord=5, ycoord=5)
        assert len(result) == 5

    def test_constant_array_stats(self):
        """Constant array: mean == median == p25 == p75 == value, std == 0."""
        r = self._grid(fill=4.0)
        mean, std, median, p25, p75 = pf.offset_stats_pixel(r, xcoord=5, ycoord=5)
        assert mean == pytest.approx(4.0)
        assert std == pytest.approx(0.0)
        assert median == pytest.approx(4.0)
        assert p25 == pytest.approx(4.0)
        assert p75 == pytest.approx(4.0)

    def test_nodata_becomes_nan(self):
        """Values equal to -9999 should be treated as NaN."""
        r = np.full((10, 10), -9999.0)
        mean, std, median, p25, p75 = pf.offset_stats_pixel(r, xcoord=5, ycoord=5)
        assert np.isnan(mean)

    def test_padding_enlarges_sample(self):
        """With pad=2 the sample is 5×5 = 25 pixels; std should be higher than pad=0."""
        r = np.arange(100, dtype=float).reshape(10, 10)
        # A single pixel has std=0 (one element); the wider window has std>0
        _, std_no_pad, *_ = pf.offset_stats_pixel(r, xcoord=5, ycoord=5, pad=0)
        _, std_pad, *_ = pf.offset_stats_pixel(r, xcoord=5, ycoord=5, pad=2)
        # Wider window contains more varied values → higher std
        assert std_pad > std_no_pad

    def test_take_velocity_false_scales_values(self):
        """take_velocity=False should scale r to m/yr before computing stats."""
        r = self._grid(fill=1.0)  # 1 pixel
        resolution = 3.0  # 3 m per pixel
        dt = 365  # 365 days
        # Expected: (1 * 3 / 365) * 365 = 3 m/yr
        mean, *_ = pf.offset_stats_pixel(
            r, xcoord=5, ycoord=5,
            resolution=resolution, dt=dt, take_velocity=False
        )
        assert mean == pytest.approx(3.0)

    def test_take_velocity_false_missing_args_returns_none(self):
        """Omitting resolution or dt when take_velocity=False → None."""
        r = self._grid()
        result = pf.offset_stats_pixel(r, xcoord=5, ycoord=5,
                                        take_velocity=False)
        assert result is None

    def test_mixed_values_percentiles(self):
        r = np.zeros((10, 10), dtype=float)
        r[5, 5] = 10.0
        r[4, 5] = 20.0
        r[6, 5] = 30.0
        mean, std, median, p25, p75 = pf.offset_stats_pixel(
            r, xcoord=5, ycoord=5, pad=1
        )
        # 9-pixel window contains 10, 20, 30 and six zeros
        assert mean == pytest.approx((10 + 20 + 30) / 9)


# ---------------------------------------------------------------------------
# offset_stats_aoi
# ---------------------------------------------------------------------------

class TestOffsetStatsAoi:
    def _raster_and_mask(self, shape=(10, 10), fill=5.0, mask_fill=1):
        r = np.full(shape, fill, dtype=float)
        mask = np.full(shape, mask_fill, dtype=int)
        return r, mask

    def test_returns_five_stats(self):
        r, mask = self._raster_and_mask()
        result = pf.offset_stats_aoi(r, mask, resolution=3.0)
        assert len(result) == 5

    def test_constant_array_stats(self):
        r, mask = self._raster_and_mask(fill=7.0)
        mean, std, median, p25, p75 = pf.offset_stats_aoi(r, mask, resolution=3.0)
        assert mean == pytest.approx(7.0)
        assert std == pytest.approx(0.0)
        assert p25 == pytest.approx(7.0)
        assert p75 == pytest.approx(7.0)

    def test_nodata_becomes_nan(self):
        r = np.full((5, 5), -9999.0)
        mask = np.ones((5, 5), dtype=int)
        mean, std, median, p25, p75 = pf.offset_stats_aoi(r, mask, resolution=3.0)
        assert np.isnan(mean)

    def test_mask_excludes_pixels(self):
        """Pixels where mask==0 should not contribute to the statistics."""
        r = np.array([[1.0, 2.0], [3.0, 4.0]])
        mask = np.array([[1, 0], [1, 0]])
        mean, *_ = pf.offset_stats_aoi(r, mask, resolution=1.0)
        # Only pixels with mask==1: values 1 and 3 → mean = 2
        assert mean == pytest.approx(2.0)

    def test_take_velocity_false_scales_correctly(self):
        r = np.full((5, 5), 1.0)
        mask = np.ones((5, 5), dtype=int)
        resolution = 3.0
        dt = 365
        mean, *_ = pf.offset_stats_aoi(
            r, mask, resolution=resolution, dt=dt, take_velocity=False
        )
        assert mean == pytest.approx(3.0)

    def test_take_velocity_false_missing_args_returns_none(self):
        r = np.full((5, 5), 1.0)
        mask = np.ones((5, 5), dtype=int)
        result = pf.offset_stats_aoi(r, mask, resolution=3.0, take_velocity=False)
        assert result is None

    def test_dimension_mismatch_returns_nan_tuple(self):
        """If r and mask have incompatible shapes, a graceful NaN tuple is returned."""
        r = np.full((5, 5), 2.0)
        # Create a mask that will cause an IndexError when used as r[mask == 1]
        # by making it a different-length 1-D array
        mask = np.array([1, 1, 1])  # incompatible shape with 5×5 r
        result = pf.offset_stats_aoi(r, mask, resolution=3.0)
        assert result is None or all(np.isnan(v) for v in result)


# ---------------------------------------------------------------------------
# calc_velocity
# ---------------------------------------------------------------------------

class TestCalcVelocity:
    def _setup_rasterio_mock(self, dx_val, dy_val, res=3.0, count=2,
                              valid=None):
        """Configure the rasterio mock to return synthetic raster data."""
        dx = np.full((10, 10), dx_val, dtype=float)
        dy = np.full((10, 10), dy_val, dtype=float)

        meta = {"transform": [res, 0, 0, 0, -res, 0], "count": count}

        src_mock = MagicMock()
        src_mock.meta = meta

        if valid is not None:
            valid_arr = np.full((10, 10), valid, dtype=float)
            src_mock.read.side_effect = lambda b: (
                dx if b == 1 else dy if b == 2 else valid_arr
            )
        else:
            src_mock.read.side_effect = lambda b: dx if b == 1 else dy

        rasterio_mock = sys.modules["rasterio"]
        ctx = rasterio_mock.open.return_value.__enter__
        ctx.return_value = src_mock
        return src_mock

    def test_velocity_magnitude_east(self):
        """1-pixel east displacement, 3 m/px, 365-day baseline → 3 m/yr."""
        self._setup_rasterio_mock(dx_val=1.0, dy_val=0.0, res=3.0)
        dt = datetime.timedelta(days=365)
        v, direction = pf.calc_velocity("fake.tif", dt)
        np.testing.assert_allclose(v, 3.0)

    def test_direction_east(self):
        """Pure eastward motion (dx>0, dy=0) → 90 ° from north."""
        self._setup_rasterio_mock(dx_val=1.0, dy_val=0.0)
        dt = datetime.timedelta(days=365)
        _, direction = pf.calc_velocity("fake.tif", dt)
        np.testing.assert_allclose(direction, 90.0, atol=1e-6)

    def test_direction_north(self):
        """Pure northward motion (dx=0, dy=1) → 0 ° from north."""
        self._setup_rasterio_mock(dx_val=0.0, dy_val=1.0)
        dt = datetime.timedelta(days=365)
        _, direction = pf.calc_velocity("fake.tif", dt)
        np.testing.assert_allclose(direction, 0.0, atol=1e-6)

    def test_direction_west(self):
        """Pure westward motion (dx=-1, dy=0) → 270 ° from north."""
        self._setup_rasterio_mock(dx_val=-1.0, dy_val=0.0)
        dt = datetime.timedelta(days=365)
        _, direction = pf.calc_velocity("fake.tif", dt)
        np.testing.assert_allclose(direction, 270.0, atol=1e-6)

    def test_negative_dt_inverts_velocity_direction(self):
        """Negative dt (ref newer than sec) should invert the displacement sign."""
        self._setup_rasterio_mock(dx_val=1.0, dy_val=0.0)
        dt_neg = datetime.timedelta(days=-365)
        _, direction_neg = pf.calc_velocity("fake.tif", dt_neg)
        # Inverted dx=-1, dy=0 → west = 270°
        np.testing.assert_allclose(direction_neg, 270.0, atol=1e-6)

    def test_medshift_subtracts_median(self):
        """With medShift=True and constant displacement, net velocity → 0."""
        self._setup_rasterio_mock(dx_val=5.0, dy_val=3.0)
        dt = datetime.timedelta(days=365)
        v, _ = pf.calc_velocity("fake.tif", dt, medShift=True)
        np.testing.assert_allclose(v, 0.0, atol=1e-9)

    def test_fixed_res_overrides_metadata(self):
        """fixed_res parameter should override the resolution read from file."""
        self._setup_rasterio_mock(dx_val=1.0, dy_val=0.0, res=3.0)
        dt = datetime.timedelta(days=365)
        v, _ = pf.calc_velocity("fake.tif", dt, fixed_res=10.0)
        np.testing.assert_allclose(v, 10.0)

    def test_valid_mask_applied_when_three_bands(self):
        """Band 3 is a validity mask; pixels where mask==0 should become NaN."""
        self._setup_rasterio_mock(dx_val=1.0, dy_val=0.0, count=3, valid=0)
        dt = datetime.timedelta(days=365)
        v, _ = pf.calc_velocity("fake.tif", dt)
        assert np.all(np.isnan(v))

    def test_velocity_scales_with_dt(self):
        """Halving dt should double velocity for the same displacement."""
        self._setup_rasterio_mock(dx_val=1.0, dy_val=0.0, res=3.0)
        v1, _ = pf.calc_velocity("fake.tif", datetime.timedelta(days=365))
        v2, _ = pf.calc_velocity("fake.tif", datetime.timedelta(days=730))
        np.testing.assert_allclose(v1, 2 * v2, rtol=1e-5)
