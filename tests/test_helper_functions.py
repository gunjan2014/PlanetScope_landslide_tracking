"""
Tests for helper_functions.py

Covers pure-logic helpers that do not require rasterio / GDAL at runtime:
    - min_max_scaler
    - fixed_val_scaler
    - percentile_scaler
    - windows_path_to_wsl
    - wsl_to_windows_path
    - get_scene_id
    - get_date
"""

import datetime

import numpy as np
import pytest

import helper_functions as helper


# ---------------------------------------------------------------------------
# min_max_scaler
# ---------------------------------------------------------------------------

class TestMinMaxScaler:
    def test_normal_array_range(self):
        """Output values should all lie within [0, 1]."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = helper.min_max_scaler(x)
        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(1.0)

    def test_normal_array_values(self):
        x = np.array([0.0, 5.0, 10.0])
        result = helper.min_max_scaler(x)
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])

    def test_single_element_returns_one(self):
        """A single-element array should return [1]."""
        x = np.array([42.0])
        result = helper.min_max_scaler(x)
        np.testing.assert_array_equal(result, np.array([1]))

    def test_empty_array_returns_empty(self):
        x = np.array([])
        result = helper.min_max_scaler(x)
        assert len(result) == 0

    def test_constant_array_produces_nan(self):
        """All-identical values cause 0/0; result should be NaN."""
        x = np.array([3.0, 3.0, 3.0])
        with pytest.warns(RuntimeWarning):
            result = helper.min_max_scaler(x)
        assert np.all(np.isnan(result))

    def test_negative_values(self):
        x = np.array([-2.0, 0.0, 2.0])
        result = helper.min_max_scaler(x)
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])

    def test_with_nan_ignores_nan(self):
        """nanmin / nanmax should ignore NaN entries."""
        x = np.array([0.0, np.nan, 10.0])
        result = helper.min_max_scaler(x)
        assert result[0] == pytest.approx(0.0)
        assert result[2] == pytest.approx(1.0)
        assert np.isnan(result[1])


# ---------------------------------------------------------------------------
# fixed_val_scaler
# ---------------------------------------------------------------------------

class TestFixedValScaler:
    def test_midpoint(self):
        result = helper.fixed_val_scaler(np.array([5.0]), xmin=0.0, xmax=10.0)
        np.testing.assert_allclose(result, [0.5])

    def test_at_minimum(self):
        result = helper.fixed_val_scaler(np.array([0.0]), xmin=0.0, xmax=10.0)
        np.testing.assert_allclose(result, [0.0])

    def test_at_maximum(self):
        result = helper.fixed_val_scaler(np.array([10.0]), xmin=0.0, xmax=10.0)
        np.testing.assert_allclose(result, [1.0])

    def test_array_input(self):
        x = np.array([0.0, 5.0, 10.0])
        result = helper.fixed_val_scaler(x, xmin=0.0, xmax=10.0)
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])

    def test_negative_range(self):
        result = helper.fixed_val_scaler(np.array([-5.0]), xmin=-10.0, xmax=0.0)
        np.testing.assert_allclose(result, [0.5])


# ---------------------------------------------------------------------------
# percentile_scaler
# ---------------------------------------------------------------------------

class TestPercentileScaler:
    def test_output_roughly_bounded(self):
        """The bulk of the output should lie near [0, 1]."""
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 100, 1000)
        result = helper.percentile_scaler(x)
        # Most values should be in [0, 1]; outliers may exceed range
        assert np.nanmedian(result) == pytest.approx(0.5, abs=0.1)

    def test_custom_percentiles(self):
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = helper.percentile_scaler(x, plow=0, pup=100)
        np.testing.assert_allclose(result, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                                             0.6, 0.7, 0.8, 0.9, 1.0],
                                   atol=1e-10)


# ---------------------------------------------------------------------------
# windows_path_to_wsl
# ---------------------------------------------------------------------------

class TestWindowsPathToWsl:
    def test_drive_letter_conversion(self):
        result = helper.windows_path_to_wsl("C:\\Users\\data\\file.tif")
        assert result == "/mnt/c/Users/data/file.tif"

    def test_lowercase_drive(self):
        result = helper.windows_path_to_wsl("D:\\data")
        assert result == "/mnt/d/data"

    def test_backslashes_without_drive(self):
        """Backslashes are converted even when no drive letter is present."""
        result = helper.windows_path_to_wsl("/home/user\\data\\file")
        assert result == "/home/user/data/file"

    def test_already_posix_path_unchanged(self):
        result = helper.windows_path_to_wsl("/home/user/data")
        assert result == "/home/user/data"


# ---------------------------------------------------------------------------
# wsl_to_windows_path
# ---------------------------------------------------------------------------

class TestWslToWindowsPath:
    def test_mnt_c_conversion(self):
        result = helper.wsl_to_windows_path("/mnt/c/Users/data/file.tif")
        assert result == "C:\\Users\\data\\file.tif"

    def test_uppercase_drive(self):
        result = helper.wsl_to_windows_path("/mnt/d/data")
        assert result == "D:\\data"

    def test_non_mnt_path_unchanged(self):
        result = helper.wsl_to_windows_path("/usr/local/bin")
        assert result == "/usr/local/bin"

    def test_roundtrip_with_windows_path_to_wsl(self):
        windows = "C:\\Users\\ariane\\data\\scene.tif"
        wsl = helper.windows_path_to_wsl(windows)
        back = helper.wsl_to_windows_path(wsl)
        assert back == windows


# ---------------------------------------------------------------------------
# get_scene_id
# ---------------------------------------------------------------------------

class TestGetSceneId:
    # PSB.SD / PS2.SD: four underscore-separated tokens before _1B_
    def test_psb_sd_l1b(self):
        fn = "20230601_123456_78_24a9_1B_AnalyticMS_SR.tif"
        assert helper.get_scene_id(fn) == "20230601_123456_78_24a9"

    def test_ps2_sd_l1b(self):
        fn = "20230601_123456_78_24a9_1B_AnalyticMS.tif"
        assert helper.get_scene_id(fn) == "20230601_123456_78_24a9"

    # PS2: three tokens before _1B_
    def test_ps2_l1b(self):
        fn = "20230601_123456_2459_1B_AnalyticMS_SR.tif"
        assert helper.get_scene_id(fn) == "20230601_123456_2459"

    def test_l3b_filename(self):
        fn = "20230601_123456_78_24a9_3B_AnalyticMS_SR.tif"
        assert helper.get_scene_id(fn) == "20230601_123456_78_24a9"

    def test_strips_directory(self):
        fn = "/path/to/data/20230601_123456_78_24a9_1B_AnalyticMS.tif"
        assert helper.get_scene_id(fn) == "20230601_123456_78_24a9"

    def test_no_processing_level_returns_none(self):
        fn = "20230601_123456_some_random_filename.tif"
        result = helper.get_scene_id(fn)
        assert result is None

    def test_windows_mixed_path(self):
        """Filenames with backslash separators (Windows paths) are handled."""
        fn = "C:\\data\\20230601_123456_78_24a9_1B_AnalyticMS.tif"
        assert helper.get_scene_id(fn) == "20230601_123456_78_24a9"


# ---------------------------------------------------------------------------
# get_date
# ---------------------------------------------------------------------------

class TestGetDate:
    def test_basic_date(self):
        sid = "20230601_123456_78_24a9"
        result = helper.get_date(sid)
        assert result == datetime.datetime(2023, 6, 1)

    def test_year_month_day(self):
        sid = "20201231_000000_00_0000"
        result = helper.get_date(sid)
        assert result == datetime.datetime(2020, 12, 31)

    def test_returns_datetime(self):
        sid = "20190315_080000_10_abcd"
        result = helper.get_date(sid)
        assert isinstance(result, datetime.datetime)
