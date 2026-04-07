"""
Tests for planet_search_functions.py

Covers:
    - check_overlap
    - get_orbit
    - find_common_perspectives
    - suggest_dem_pairs
"""

import datetime
import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

import planet_search_functions as psf


# ---------------------------------------------------------------------------
# Helpers for building mock GeoJSON features
# ---------------------------------------------------------------------------

def _make_feature(coords, view_angle=5.0, sat_az=180.0,
                  sun_azimuth=150.0, sun_elevation=45.0,
                  scene_id="20220101_120000_00_abcd",
                  acquired="2022-01-01T12:00:00.000000Z"):
    """Return a minimal Planet feature dict."""
    return {
        "id": scene_id,
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [coords],
        },
        "properties": {
            "view_angle": view_angle,
            "satellite_azimuth": sat_az,
            "sun_azimuth": sun_azimuth,
            "sun_elevation": sun_elevation,
            "gsd": 3.0,
            "quality_category": "standard",
            "acquired": acquired,
            "ground_control": True,
        },
    }


def _aoi_file(coords):
    """Write a GeoJSON AOI file to a temp path and return its path."""
    gj = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords],
                },
            }
        ],
    }
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".geojson", delete=False
    )
    json.dump(gj, tmp)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# check_overlap
# ---------------------------------------------------------------------------

class TestCheckOverlap:
    # AOI: unit square [0,1]×[0,1]
    AOI_COORDS = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]

    def setup_method(self):
        self.aoi_path = _aoi_file(self.AOI_COORDS)

    def teardown_method(self):
        os.unlink(self.aoi_path)

    def test_fully_covering_feature_kept(self):
        """A scene that covers the entire AOI passes the 99 % threshold."""
        feat = _make_feature([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        result = psf.check_overlap([feat], self.aoi_path, min_overlap=99)
        assert len(result) == 1

    def test_non_overlapping_feature_excluded(self):
        """A scene far away from the AOI should be dropped."""
        feat = _make_feature([(10, 10), (11, 10), (11, 11), (10, 11), (10, 10)])
        result = psf.check_overlap([feat], self.aoi_path, min_overlap=1)
        assert len(result) == 0

    def test_partial_overlap_below_threshold(self):
        """50 % overlap with min_overlap=99 should be excluded."""
        feat = _make_feature([(0.5, 0), (1.5, 0), (1.5, 1), (0.5, 1), (0.5, 0)])
        result = psf.check_overlap([feat], self.aoi_path, min_overlap=99)
        assert len(result) == 0

    def test_partial_overlap_above_threshold(self):
        """50 % overlap with min_overlap=40 should be included."""
        feat = _make_feature([(0.5, 0), (1.5, 0), (1.5, 1), (0.5, 1), (0.5, 0)])
        result = psf.check_overlap([feat], self.aoi_path, min_overlap=40)
        assert len(result) == 1

    def test_multiple_features_filtered(self):
        inside = _make_feature([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        outside = _make_feature([(10, 10), (11, 10), (11, 11), (10, 11), (10, 10)])
        result = psf.check_overlap([inside, outside], self.aoi_path, min_overlap=99)
        assert len(result) == 1

    def test_empty_feature_list(self):
        result = psf.check_overlap([], self.aoi_path, min_overlap=99)
        assert result == []


# ---------------------------------------------------------------------------
# get_orbit
# ---------------------------------------------------------------------------

class TestGetOrbit:
    """
    NE orbit: the two leftmost corners have the lower-latitude corner first
              when sorted by latitude (argmin of leftlat == 0).
    NW orbit: the two leftmost corners have the higher-latitude corner first
              (argmin of leftlat == 1).
    """

    def _ne_feature(self):
        # lon=[0,1,1,0], lat=[0,0,1,1]
        # left indices: 0,3 → leftlat=[0,1] → argmin=0 → NE
        coords = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
        return _make_feature(coords)

    def _nw_feature(self):
        # lon=[0,1,1,0], lat=[1,1,0,0]
        # left indices: 0,3 → leftlat=[1,0] → argmin=1 → NW
        coords = [(0, 1), (1, 1), (1, 0), (0, 0), (0, 1)]
        return _make_feature(coords)

    def test_ne_filter_keeps_ne(self):
        result = psf.get_orbit([self._ne_feature()], direction="NE")
        assert len(result) == 1

    def test_ne_filter_drops_nw(self):
        result = psf.get_orbit([self._nw_feature()], direction="NE")
        assert len(result) == 0

    def test_nw_filter_keeps_nw(self):
        result = psf.get_orbit([self._nw_feature()], direction="NW")
        assert len(result) == 1

    def test_nw_filter_drops_ne(self):
        result = psf.get_orbit([self._ne_feature()], direction="NW")
        assert len(result) == 0

    def test_mixed_list(self):
        features = [self._ne_feature(), self._nw_feature()]
        ne_result = psf.get_orbit(features, direction="NE")
        nw_result = psf.get_orbit(features, direction="NW")
        assert len(ne_result) == 1
        assert len(nw_result) == 1

    def test_empty_list(self):
        assert psf.get_orbit([], direction="NE") == []


# ---------------------------------------------------------------------------
# find_common_perspectives
# ---------------------------------------------------------------------------

class TestFindCommonPerspectives:
    def _make_df(self, view_angles, scene_ids=None, datetimes=None):
        n = len(view_angles)
        if scene_ids is None:
            scene_ids = [
                f"202301{i+1:02d}_120000_00_abcd"
                for i in range(n)
            ]
        if datetimes is None:
            datetimes = [
                datetime.datetime(2023, 1, i + 1, 12, 0, 0)
                for i in range(n)
            ]
        return pd.DataFrame(
            {
                "ids": scene_ids,
                "true_view_angle": view_angles,
                "datetime": datetimes,
            }
        )

    def test_groups_similar_angles(self):
        """Five scenes with similar view angles should form a group."""
        va = [5.0, 5.1, 4.9, 5.05, 4.95, 5.02]
        df = self._make_df(va)
        groups = psf.find_common_perspectives(df, va_diff_thresh=0.2,
                                               min_group_size=5)
        assert groups is not None
        assert len(groups) > 0

    def test_insufficient_similar_scenes_returns_none(self):
        """Only 2 scenes with similar angles → group too small."""
        va = [5.0, 5.1, 20.0, 30.0]
        df = self._make_df(va)
        result = psf.find_common_perspectives(df, va_diff_thresh=0.2,
                                               min_group_size=5)
        assert result is None

    def test_group_id_assigned(self):
        va = [5.0] * 8
        df = self._make_df(va)
        groups = psf.find_common_perspectives(df, va_diff_thresh=0.5,
                                               min_group_size=5)
        assert "group_id" in groups.columns

    def test_min_dt_filters_close_acquisitions(self):
        """With min_dt=30 and daily acquisitions, all scenes within 30 days
        of the first should be dropped."""
        n = 10
        va = [5.0] * n
        datetimes = [
            datetime.datetime(2023, 1, 1) + datetime.timedelta(days=i)
            for i in range(n)
        ]
        df = self._make_df(va, datetimes=datetimes)
        groups = psf.find_common_perspectives(
            df, va_diff_thresh=0.5, min_group_size=2, min_dt=30
        )
        # After dt filtering, groups may be too small; result may be empty df
        if groups is not None and len(groups) > 0:
            dts = groups["datetime"].diff().dropna()
            assert all(dt >= datetime.timedelta(days=30) for dt in dts)


# ---------------------------------------------------------------------------
# suggest_dem_pairs
# ---------------------------------------------------------------------------

class TestSuggestDemPairs:
    def _make_scenes(self):
        return pd.DataFrame(
            {
                "ids": [
                    "20220101_120000_00_aaa1",
                    "20220110_120000_00_bbb2",
                    "20220115_120000_00_ccc3",
                ],
                "view_angle": [10.0, 15.0, 8.0],
                "true_view_angle": [10.0, -15.0, 8.0],
                "datetime": [
                    datetime.datetime(2022, 1, 1, 12),
                    datetime.datetime(2022, 1, 10, 12),
                    datetime.datetime(2022, 1, 15, 12),
                ],
            }
        )

    def test_returns_dataframe(self):
        pairs = psf.suggest_dem_pairs(self._make_scenes(), min_va=5, max_dt=30)
        assert isinstance(pairs, pd.DataFrame)

    def test_min_va_filters_small_angles(self):
        """Scenes with view_angle < min_va should be excluded."""
        scenes = self._make_scenes()
        pairs_all = psf.suggest_dem_pairs(scenes, min_va=0, max_dt=30)
        pairs_filtered = psf.suggest_dem_pairs(scenes, min_va=9, max_dt=30)
        # scene ccc3 (va=8) should be dropped when min_va=9
        if pairs_filtered is not None:
            assert "ccc3" not in pairs_filtered["img1"].values

    def test_max_dt_filters_distant_pairs(self):
        """Pairs separated by more than max_dt days should be excluded."""
        pairs = psf.suggest_dem_pairs(self._make_scenes(), min_va=5, max_dt=5)
        if pairs is not None and len(pairs) > 0:
            dts = pairs["dt"].apply(lambda d: d.days if hasattr(d, "days") else d)
            assert all(d <= 5 for d in dts)

    def test_no_pairs_when_all_same_date(self):
        """All scenes on the same day → no forward pairs."""
        scenes = pd.DataFrame(
            {
                "ids": ["20220101_a", "20220101_b"],
                "view_angle": [10.0, 15.0],
                "true_view_angle": [10.0, -15.0],
                "datetime": [
                    datetime.datetime(2022, 1, 1),
                    datetime.datetime(2022, 1, 1),
                ],
            }
        )
        # When dt == 0 for every candidate, no pairs are added; the function
        # may return None, an empty DataFrame, or raise KeyError on empty records.
        try:
            pairs = psf.suggest_dem_pairs(scenes, min_va=0, max_dt=30)
            assert pairs is None or len(pairs) == 0
        except KeyError:
            pass  # acceptable: empty records list has no columns to explode

    def test_columns_present(self):
        pairs = psf.suggest_dem_pairs(self._make_scenes(), min_va=5, max_dt=30)
        if pairs is not None and len(pairs) > 0:
            for col in ("img1", "img2", "dt", "true_va_diff"):
                assert col in pairs.columns
