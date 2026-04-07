"""
conftest.py – pytest configuration loaded before any test module.

Heavy dependencies that are not available in the CI/test environment
(rasterio, GDAL, OpenCV, matplotlib, scikit-image, tqdm) are replaced by
lightweight MagicMock objects so that tests for pure-logic functions can be
imported and run without those libraries being installed.
"""

import os
import sys
from unittest.mock import MagicMock

import numpy as np

# ---------------------------------------------------------------------------
# Add the repository root to sys.path so the source modules are importable.
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ---------------------------------------------------------------------------
# Mock unavailable heavy dependencies
# ---------------------------------------------------------------------------
_MOCK_MODULES = [
    "rasterio",
    "osgeo",
    "osgeo.gdal",
    "osgeo.gdalconst",
    "osgeo.ogr",
    "cv2",
    "matplotlib",
    "matplotlib.pyplot",
    "matplotlib.gridspec",
    "mpl_toolkits",
    "mpl_toolkits.axes_grid1",
    "skimage",
    "skimage.exposure",
    "tqdm",
]

for _mod in _MOCK_MODULES:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# Give rasterio.float32 a real NumPy dtype so arithmetic in save_file works.
sys.modules["rasterio"].float32 = np.float32
