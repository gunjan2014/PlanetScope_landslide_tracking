"""
Tests for preprocessing_functions.py

Covers:
    - generate_matchfile_from_search
    - generate_matchfile_from_groups
    - match_common_perspectives_and_illumination
    - rate_match
"""

import datetime

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

import preprocessing_functions as pre


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scene_ids(dates):
    """Build valid PSB.SD-style scene IDs from a list of date strings."""
    return [f"{d}_120000_00_abcd" for d in dates]


# ---------------------------------------------------------------------------
# generate_matchfile_from_search
# ---------------------------------------------------------------------------

class TestGenerateMatchfileFromSearch:
    def _make_df(self, dates):
        return pd.DataFrame({"ids": _scene_ids(dates)})

    def test_basic_matching_returns_all_pairs(self):
        """Three scenes → 3 pairs (n*(n-1)/2)."""
        df = self._make_df(["20220101", "20220201", "20220301"])
        matches = pre.generate_matchfile_from_search(df)
        assert len(matches) == 3

    def test_returns_dataframe_with_ref_sec(self):
        df = self._make_df(["20220101", "20220201"])
        matches = pre.generate_matchfile_from_search(df)
        assert isinstance(matches, pd.DataFrame)
        assert "ref" in matches.columns
        assert "sec" in matches.columns

    def test_two_scenes_one_pair(self):
        df = self._make_df(["20220101", "20220201"])
        matches = pre.generate_matchfile_from_search(df)
        assert len(matches) == 1

    def test_dt_min_filters_close_pairs(self):
        """All pairs have dt < 5 days → everything filtered → None."""
        df = self._make_df(["20220101", "20220102", "20220103"])
        result = pre.generate_matchfile_from_search(df, dt_min=5)
        assert result is None

    def test_dt_min_keeps_distant_pairs(self):
        """Pairs with dt > dt_min days should survive; short pairs are filtered."""
        # Jan1→Feb1 = 31 days, Jan1→Mar1 = 59 days, Feb1→Mar1 = 28 days
        # With dt_min=29, only the 31-day and 59-day pairs survive (28 < 29)
        df = self._make_df(["20220101", "20220201", "20220301"])
        matches = pre.generate_matchfile_from_search(df, dt_min=29)
        assert matches is not None
        assert len(matches) == 2

    def test_dt_min_partial_filter(self):
        """Only the long-baseline pair should survive dt_min=60."""
        df = self._make_df(["20220101", "20220115", "20220401"])
        # Jan1→Jan15 = 14 days (dropped), Jan1→Apr1 = 90 days (kept),
        # Jan15→Apr1 = 76 days (kept)
        matches = pre.generate_matchfile_from_search(df, dt_min=60)
        assert matches is not None
        assert len(matches) == 2

    def test_single_scene_returns_no_pairs(self):
        """A single scene produces no matchable pairs (empty or error)."""
        df = self._make_df(["20220101"])
        # The function may raise KeyError or return an empty result for 1 scene
        try:
            matches = pre.generate_matchfile_from_search(df)
            assert matches is None or len(matches) == 0
        except KeyError:
            pass  # acceptable: no pairs can be formed from one scene


# ---------------------------------------------------------------------------
# generate_matchfile_from_groups
# ---------------------------------------------------------------------------

class TestGenerateMatchfileFromGroups:
    def _make_groups(self, dates_by_group):
        """dates_by_group: dict of {group_id: [date_str, ...]}"""
        rows = []
        for gid, dates in dates_by_group.items():
            for d in dates:
                rows.append({
                    "ids": f"{d}_120000_00_abcd",
                    "group_id": gid,
                })
        return pd.DataFrame(rows)

    def test_single_group_all_pairs(self):
        groups = self._make_groups({0: ["20220101", "20220201", "20220301"]})
        matches = pre.generate_matchfile_from_groups(groups)
        assert len(matches) == 3

    def test_two_groups_no_cross_group_pairs(self):
        groups = self._make_groups({
            0: ["20220101", "20220201"],
            1: ["20220301", "20220401"],
        })
        matches = pre.generate_matchfile_from_groups(groups)
        # 1 pair per group × 2 groups = 2 pairs
        assert len(matches) == 2

    def test_returns_ref_sec_columns(self):
        groups = self._make_groups({0: ["20220101", "20220201"]})
        matches = pre.generate_matchfile_from_groups(groups)
        assert "ref" in matches.columns
        assert "sec" in matches.columns

    def test_dt_min_filters_close_pairs(self):
        groups = self._make_groups({0: ["20220101", "20220102", "20220103"]})
        result = pre.generate_matchfile_from_groups(groups, dt_min=5)
        assert result is None

    def test_dt_min_keeps_distant_pairs(self):
        """Pairs with dt > dt_min days should survive; short pairs are filtered."""
        # Jan1→Feb1 = 31 days, Jan1→Mar1 = 59 days, Feb1→Mar1 = 28 days
        # With dt_min=29, only 31-day and 59-day pairs survive (28 < 29)
        groups = self._make_groups({0: ["20220101", "20220201", "20220301"]})
        matches = pre.generate_matchfile_from_groups(groups, dt_min=29)
        assert matches is not None
        assert len(matches) == 2

    def test_no_duplicate_pairs(self):
        """Pairs that appear in multiple groups should be deduplicated."""
        groups = self._make_groups({
            0: ["20220101", "20220201", "20220301"],
            1: ["20220101", "20220201"],
        })
        matches = pre.generate_matchfile_from_groups(groups)
        dupe = matches.duplicated(subset=["ref", "sec"])
        assert not dupe.any()


# ---------------------------------------------------------------------------
# match_common_perspectives_and_illumination
# ---------------------------------------------------------------------------

class TestMatchCommonPerspectivesAndIllumination:
    def _make_df(self, n=8, va_base=5.0, sun_az_base=150.0, sun_elev_base=45.0):
        ids = [f"202201{i+1:02d}_120000_00_abcd" for i in range(n)]
        return pd.DataFrame(
            {
                "ids": ids,
                "true_view_angle": [va_base + 0.05 * i for i in range(n)],
                "sun_azimuth": [sun_az_base + 0.5 * i for i in range(n)],
                "sun_elevation": [sun_elev_base + 0.2 * i for i in range(n)],
            }
        )

    def test_returns_dataframe(self):
        df = self._make_df(n=8)
        out = pre.match_common_perspectives_and_illumination(
            df, va_diff_thresh=0.5, sun_az_thresh=5, sun_elev_thresh=5,
            min_group_size=5, dt_min=1
        )
        assert isinstance(out, pd.DataFrame)

    def test_ref_sec_columns_present(self):
        df = self._make_df(n=8)
        out = pre.match_common_perspectives_and_illumination(
            df, va_diff_thresh=0.5, sun_az_thresh=5, sun_elev_thresh=5,
            min_group_size=5, dt_min=1
        )
        assert "ref" in out.columns
        assert "sec" in out.columns

    def test_tight_threshold_returns_fewer_pairs(self):
        df = self._make_df(n=8)
        out_loose = pre.match_common_perspectives_and_illumination(
            df, va_diff_thresh=2.0, sun_az_thresh=10, sun_elev_thresh=10,
            min_group_size=3, dt_min=1
        )
        out_tight = pre.match_common_perspectives_and_illumination(
            df, va_diff_thresh=0.01, sun_az_thresh=0.01, sun_elev_thresh=0.01,
            min_group_size=3, dt_min=1
        )
        assert len(out_loose) >= len(out_tight)

    def test_all_different_angles_returns_empty(self):
        df = pd.DataFrame(
            {
                "ids": [f"2022010{i}_120000_00_abcd" for i in range(1, 5)],
                "true_view_angle": [0.0, 10.0, 20.0, 30.0],
                "sun_azimuth": [0.0, 10.0, 20.0, 30.0],
                "sun_elevation": [0.0, 10.0, 20.0, 30.0],
            }
        )
        out = pre.match_common_perspectives_and_illumination(
            df, va_diff_thresh=0.1, sun_az_thresh=0.1, sun_elev_thresh=0.1,
            min_group_size=5, dt_min=1
        )
        assert len(out) == 0

    def test_dt_min_filters_same_day_pairs(self):
        """All scenes on the same day → dt=0 < dt_min=1 → empty output."""
        df = pd.DataFrame(
            {
                "ids": [f"20220101_12000{i}_00_abcd" for i in range(6)],
                "true_view_angle": [5.0] * 6,
                "sun_azimuth": [150.0] * 6,
                "sun_elevation": [45.0] * 6,
            }
        )
        out = pre.match_common_perspectives_and_illumination(
            df, va_diff_thresh=1.0, sun_az_thresh=10, sun_elev_thresh=10,
            min_group_size=5, dt_min=1
        )
        assert len(out) == 0


# ---------------------------------------------------------------------------
# rate_match
# ---------------------------------------------------------------------------

class TestRateMatch:
    def _make_footprint(self, xmin, xmax, ymin, ymax):
        return [
            [xmin, ymin],
            [xmax, ymin],
            [xmax, ymax],
            [xmin, ymax],
            [xmin, ymin],
        ]

    def test_returns_dataframe(self):
        infodf = pd.DataFrame(
            {
                "ids": ["20220101_120000_78_abcd", "20220201_120000_78_efgh"],
                "footprint": [
                    self._make_footprint(0, 1, 0, 1),
                    self._make_footprint(0.5, 1.5, 0, 1),
                ],
                "true_view_angle": [5.0, 5.2],
            }
        )
        matchdf = pd.DataFrame(
            {
                "ref": ["path/20220101_120000_78_abcd_1B_AnalyticMS.tif"],
                "sec": ["path/20220201_120000_78_efgh_1B_AnalyticMS.tif"],
            }
        )
        result = pre.rate_match(infodf, matchdf)
        assert isinstance(result, pd.DataFrame)

    def test_overlap_calculation(self):
        """50 % overlap between ref and sec footprints."""
        infodf = pd.DataFrame(
            {
                "ids": ["20220101_120000_78_abcd", "20220201_120000_78_efgh"],
                "footprint": [
                    self._make_footprint(0, 1, 0, 1),    # area = 1
                    self._make_footprint(0.5, 1.5, 0, 1),  # overlap = 0.5
                ],
                "true_view_angle": [5.0, 5.2],
            }
        )
        matchdf = pd.DataFrame(
            {
                "ref": ["path/20220101_120000_78_abcd_1B_AnalyticMS.tif"],
                "sec": ["path/20220201_120000_78_efgh_1B_AnalyticMS.tif"],
            }
        )
        result = pre.rate_match(infodf, matchdf)
        assert result["overlap"].iloc[0] == pytest.approx(50.0)

    def test_full_overlap(self):
        """Identical footprints → 100 % overlap."""
        footprint = self._make_footprint(0, 1, 0, 1)
        infodf = pd.DataFrame(
            {
                "ids": ["20220101_120000_78_abcd", "20220201_120000_78_efgh"],
                "footprint": [footprint, footprint],
                "true_view_angle": [5.0, 5.5],
            }
        )
        matchdf = pd.DataFrame(
            {
                "ref": ["path/20220101_120000_78_abcd_1B_AnalyticMS.tif"],
                "sec": ["path/20220201_120000_78_efgh_1B_AnalyticMS.tif"],
            }
        )
        result = pre.rate_match(infodf, matchdf)
        assert result["overlap"].iloc[0] == pytest.approx(100.0)

    def test_true_va_diff_column(self):
        """Check that view angle difference is calculated correctly."""
        infodf = pd.DataFrame(
            {
                "ids": ["20220101_120000_78_abcd", "20220201_120000_78_efgh"],
                "footprint": [
                    self._make_footprint(0, 1, 0, 1),
                    self._make_footprint(0, 1, 0, 1),
                ],
                "true_view_angle": [5.0, 8.0],
            }
        )
        matchdf = pd.DataFrame(
            {
                "ref": ["path/20220101_120000_78_abcd_1B_AnalyticMS.tif"],
                "sec": ["path/20220201_120000_78_efgh_1B_AnalyticMS.tif"],
            }
        )
        result = pre.rate_match(infodf, matchdf)
        va_diff = result["true_va_diff"].iloc[0]
        assert float(va_diff) == pytest.approx(3.0)
