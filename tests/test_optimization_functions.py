"""
Tests for optimization_functions.py

Covers the polynomial basis functions and the percentile_cut helper:
    - polyXY1, polyXY2, polyXY3
    - polyXYZ1, polyXYZ2, polyXYZ3
    - percentile_cut
"""

import numpy as np
import pytest

# Import only the pure-math symbols – avoids pulling in cv2, matplotlib, etc.
from optimization_functions import (
    polyXY1,
    polyXY2,
    polyXY3,
    polyXYZ1,
    polyXYZ2,
    polyXYZ3,
    percentile_cut,
)


# ---------------------------------------------------------------------------
# polyXY1  –  a*x + b*y + c
# ---------------------------------------------------------------------------

class TestPolyXY1:
    def test_all_zeros(self):
        assert polyXY1((0, 0), 1, 2, 3) == pytest.approx(3.0)

    def test_unit_x(self):
        assert polyXY1((1, 0), a=1, b=0, c=0) == pytest.approx(1.0)

    def test_unit_y(self):
        assert polyXY1((0, 1), a=0, b=1, c=0) == pytest.approx(1.0)

    def test_combined(self):
        # 2*3 + 4*5 + 6 = 6 + 20 + 6 = 32
        assert polyXY1((3, 5), a=2, b=4, c=6) == pytest.approx(32.0)

    def test_array_input(self):
        x = np.array([1.0, 2.0])
        y = np.array([0.0, 0.0])
        result = polyXY1((x, y), a=2, b=0, c=1)
        np.testing.assert_allclose(result, [3.0, 5.0])


# ---------------------------------------------------------------------------
# polyXY2  –  a*x² + b*y² + c*x*y + d*x + e*y + f
# ---------------------------------------------------------------------------

class TestPolyXY2:
    def test_offset_only(self):
        assert polyXY2((0, 0), 0, 0, 0, 0, 0, 5) == pytest.approx(5.0)

    def test_quadratic_x(self):
        # a*x² only: 3*4 = 12
        assert polyXY2((2, 0), 3, 0, 0, 0, 0, 0) == pytest.approx(12.0)

    def test_quadratic_y(self):
        assert polyXY2((0, 3), 0, 2, 0, 0, 0, 0) == pytest.approx(18.0)

    def test_cross_term(self):
        assert polyXY2((2, 3), 0, 0, 1, 0, 0, 0) == pytest.approx(6.0)

    def test_full_expression(self):
        # a=1,b=1,c=0,d=0,e=0,f=0  →  x²+y² at (3,4) = 9+16 = 25
        assert polyXY2((3, 4), 1, 1, 0, 0, 0, 0) == pytest.approx(25.0)

    def test_array_input(self):
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 1.0])
        result = polyXY2((x, y), 1, 1, 0, 0, 0, 0)
        np.testing.assert_allclose(result, [2.0, 5.0])


# ---------------------------------------------------------------------------
# polyXY3  –  degree-3 polynomial in x, y (note: parameter j is unused)
# ---------------------------------------------------------------------------

class TestPolyXY3:
    def test_offset_only(self):
        # All zero coefficients except k (the constant)
        assert polyXY3((0, 0), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7) == pytest.approx(7.0)

    def test_cubic_x(self):
        # a*x³ at x=2: 3*8 = 24
        assert polyXY3((2, 0), 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) == pytest.approx(24.0)

    def test_cubic_y(self):
        assert polyXY3((0, 2), 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0) == pytest.approx(16.0)

    def test_j_parameter_is_unused(self):
        """Parameter j appears in the signature but not in the formula.
        Changing j should not alter the result."""
        base = polyXY3((1, 1), 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        with_j = polyXY3((1, 1), 1, 1, 0, 0, 0, 0, 0, 0, 0, 99, 0)
        assert base == pytest.approx(with_j)

    def test_array_input(self):
        x = np.array([1.0, 2.0])
        y = np.array([0.0, 0.0])
        # a=1, rest=0 → x³
        result = polyXY3((x, y), 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        np.testing.assert_allclose(result, [1.0, 8.0])


# ---------------------------------------------------------------------------
# polyXYZ1  –  a*x + b*y + c*z + d
# ---------------------------------------------------------------------------

class TestPolyXYZ1:
    def test_constant(self):
        assert polyXYZ1((0, 0, 0), 1, 1, 1, 5) == pytest.approx(5.0)

    def test_unit_contributions(self):
        assert polyXYZ1((1, 0, 0), 2, 0, 0, 0) == pytest.approx(2.0)
        assert polyXYZ1((0, 3, 0), 0, 4, 0, 0) == pytest.approx(12.0)
        assert polyXYZ1((0, 0, 5), 0, 0, 6, 0) == pytest.approx(30.0)

    def test_combined(self):
        # 1*1 + 2*2 + 3*3 + 4 = 1+4+9+4 = 18
        assert polyXYZ1((1, 2, 3), 1, 2, 3, 4) == pytest.approx(18.0)


# ---------------------------------------------------------------------------
# polyXYZ2  –  degree-2 polynomial in x, y, z
# ---------------------------------------------------------------------------

class TestPolyXYZ2:
    def test_constant_only(self):
        assert polyXYZ2((0, 0, 0), 0, 0, 0, 0, 0, 0, 0, 0, 0, 7) == pytest.approx(7.0)

    def test_quadratic_x(self):
        assert polyXYZ2((3, 0, 0), 1, 0, 0, 0, 0, 0, 0, 0, 0, 0) == pytest.approx(9.0)

    def test_quadratic_y(self):
        assert polyXYZ2((0, 3, 0), 0, 1, 0, 0, 0, 0, 0, 0, 0, 0) == pytest.approx(9.0)

    def test_quadratic_z(self):
        assert polyXYZ2((0, 0, 3), 0, 0, 1, 0, 0, 0, 0, 0, 0, 0) == pytest.approx(9.0)

    def test_full_expression(self):
        # x=1,y=1,z=1 with a=b=c=d=e=f=g=h=i=j=1:
        # 1+1+1+1+1+1+1+1+1+1 = 10
        assert polyXYZ2((1, 1, 1), 1, 1, 1, 1, 1, 1, 1, 1, 1, 1) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# polyXYZ3  –  degree-3 polynomial in x, y, z
# ---------------------------------------------------------------------------

class TestPolyXYZ3:
    def test_constant_only(self):
        assert polyXYZ3((0, 0, 0), *([0]*19), 9) == pytest.approx(9.0)

    def test_cubic_x(self):
        # first coefficient is a (x³); at x=2 → 2*8 = 16
        coeffs = [2] + [0]*19
        assert polyXYZ3((2, 0, 0), *coeffs) == pytest.approx(16.0)

    def test_cubic_z(self):
        # third coefficient is c (z³)
        coeffs = [0, 0, 3] + [0]*17
        assert polyXYZ3((0, 0, 2), *coeffs) == pytest.approx(24.0)

    def test_array_input(self):
        x = np.array([1.0, 2.0])
        y = np.array([0.0, 0.0])
        z = np.array([0.0, 0.0])
        # a=1, rest zero → x³
        coeffs = [1] + [0]*19
        result = polyXYZ3((x, y, z), *coeffs)
        np.testing.assert_allclose(result, [1.0, 8.0])


# ---------------------------------------------------------------------------
# percentile_cut
# ---------------------------------------------------------------------------

class TestPercentileCut:
    def test_replaces_outliers_with_nan(self):
        dat = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0], dtype=float)
        result = percentile_cut(dat.copy(), plow=5, pup=95)
        # 100 is a high outlier and should become NaN
        assert np.isnan(result[-1])

    def test_inliers_unchanged(self):
        dat = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
        result = percentile_cut(dat.copy(), plow=0, pup=100)
        # With full percentile range, no values should be replaced
        assert not np.any(np.isnan(result))

    def test_custom_replace_value(self):
        dat = np.array([0.0, 5.0, 100.0], dtype=float)
        result = percentile_cut(dat.copy(), plow=5, pup=95, replace=0.0)
        assert result[-1] == pytest.approx(0.0)

    def test_low_outlier_replaced(self):
        dat = np.array([-100.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
        result = percentile_cut(dat.copy(), plow=5, pup=95)
        assert np.isnan(result[0])

    def test_existing_nans_ignored(self):
        """NaN values in input should not affect percentile calculation."""
        dat = np.array([1.0, 2.0, np.nan, 3.0, 4.0, 5.0], dtype=float)
        result = percentile_cut(dat.copy(), plow=0, pup=100)
        # NaN was already there; other values are within range
        assert np.isnan(result[2])
        assert not np.isnan(result[0])

    def test_uniform_array(self):
        dat = np.full(10, 3.0)
        result = percentile_cut(dat.copy(), plow=5, pup=95)
        # All values equal percentiles → values on boundary should remain
        assert not np.all(np.isnan(result))

    def test_two_d_array(self):
        dat = np.array([[1.0, 50.0], [2.0, 3.0]], dtype=float)
        result = percentile_cut(dat.copy(), plow=5, pup=95)
        assert np.isnan(result[0, 1])
