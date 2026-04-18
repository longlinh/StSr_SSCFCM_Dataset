# Landsat 8 Dataset — StSr-SSCFCM

Dataset for the paper: **"Semi-Supervised Collaborative Fuzzy Clustering With a Softmax Regression-Based Self-Training Strategy for Landcover Classification From Remote Sensing Imagery"**

Authors: Xuan Hoang Nguyen, Dinh Sinh Mai, Long Giang Nguyen, Trong Hop Dang.

Public mirror: <https://github.com/longlinh/StSr_SSCFCM_Dataset>

## Overview

This dataset contains Landsat 8 OLI/TIRS satellite imagery for three regions in Vietnam, preprocessed and labeled for semi-supervised land cover classification research. The data is designed to simulate real-world collaborative learning scenarios where different sites have varying amounts of labeled data.

## Data Source

- **Satellite**: Landsat 8 OLI/TIRS (NASA/USGS, downloaded from USGS EarthExplorer)
- **Processing Level**: Surface Reflectance (SR)
- **Spatial Resolution**: 30 meters/pixel
- **Temporal Coverage**: 2020-2023 median composite (cloud-free)
- **Source Platform**: Google Earth Engine
- **Coordinate System**: EPSG:4326 (WGS84)

## Regions

| Region | Code | Dimensions | Total Pixels | Labeled Pixels | Label Ratio |
|--------|------|------------|--------------|----------------|-------------|
| Hanoi | hn | 1201 × 1001 | 1,202,201 | 38,593 | 3.21% |
| Thanh Hoa | th2 | 981 × 825 | 809,325 | 54,544 | 6.74% |
| Ho Chi Minh City | hcm2 | 1336 × 1114 | 1,488,304 | 116,937 | 7.86% |
| **Total** | - | - | **3,499,830** | **210,074** | **6.00%** |

## Geographic Coordinates

| Region | Center (Lat, Lon) | Bounding Box |
|--------|-------------------|--------------|
| Hanoi | 21.08°N, 105.85°E | 105.58°E - 106.12°E, 20.85°N - 21.30°N |
| Thanh Hoa | 19.92°N, 105.62°E | 105.40°E - 105.84°E, 19.74°N - 20.11°N |
| Ho Chi Minh City | 10.92°N, 106.67°E | 106.37°E - 106.97°E, 10.67°N - 11.18°N |

## Files per Region (7 files each)

| # | File | Description | Data Type |
|---|------|-------------|-----------|
| 1 | `{region}_z50_SR_B2.tif` | Blue band (0.452-0.512 µm) | Float32 |
| 2 | `{region}_z50_SR_B3.tif` | Green band (0.533-0.590 µm) | Float32 |
| 3 | `{region}_z50_SR_B4.tif` | Red band (0.636-0.673 µm) | Float32 |
| 4 | `{region}_z50_SR_B5.tif` | NIR band (0.851-0.879 µm) | Float32 |
| 5 | `{region}_z50_RGB.tif` | RGB composite (B4, B3, B2) | UInt8 |
| 6 | `{region}_z50_labeled_{X}pct.tif` | Semi-supervised label mask (region-based) | UInt8 |
| 7 | `{region}_z50_labeled_{X}pct.png` | Label visualization | RGB |

`{X}` = `3` (hn), `7` (th2), `8` (hcm2) — rounded labeling ratios appearing in filenames.

## Land Cover Classes (6 classes)

| Value | Class | Description | Color (PNG) |
|-------|-------|-------------|-------------|
| 0 | Unlabeled | No label assigned | Light gray |
| 1 | Forest | Dense tree cover | Dark green |
| 2 | Vegetation | Shrubs, grass, sparse trees | Forest green |
| 3 | Water/Wetland | Wetlands, shallow water | Cyan |
| 4 | Water | Deep water bodies, rivers | Blue |
| 5 | Urban | Built-up areas, roads | Gray |
| 6 | Agriculture | Cropland, paddy fields | Bright green |

**Note**: In the TIF label files, class values are 1-6 (0 = unlabeled). In the source labels, classes are 0-5.

## Labeling Methodology

### Region-based Labeling
Labels are assigned using a **circular region strategy** to simulate realistic ground-truth collection scenarios:

- **Method**: Circular regions with radius = 30 pixels (~900m)
- **Selection**: Stratified random sampling ensuring class balance
- **Constraint**: `same_class_only=True` — each circle contains only one land cover class
- **Purpose**: Simulates field surveys where experts label contiguous areas

### Varying Label Ratios
Different regions have different labeling ratios to simulate collaborative learning scenarios:
- **Hanoi (3.21%)**: Sparse labels — benefits most from collaborative learning
- **Thanh Hoa (6.74%)**: Moderate labels
- **Ho Chi Minh City (7.86%)**: Dense labels — knowledge source for other regions

### Labeling Parameters
```python
seed = 42
radius = 30  # pixels
n_regions_per_class = {
    'hn': 10,    # 10 regions × 6 classes = 60 circles  (3.21% labels)
    'th2': 14,   # 14 regions × 6 classes = 84 circles  (6.74% labels)
    'hcm2': 24,  # 24 regions × 6 classes = 144 circles (7.86% labels)
}
```

## Preprocessing

1. **Band Selection**: B2 (Blue), B3 (Green), B4 (Red), B5 (NIR)
2. **Cloud Masking**: Using QA_PIXEL band from Landsat 8
3. **Temporal Compositing**: Median composite to remove clouds/shadows
4. **Normalization**: Z-score standardization per band (for clustering)

## Usage Example

```python
import rasterio
import numpy as np

# Load spectral bands
region = "hn"
bands = []
for b in ["B2", "B3", "B4", "B5"]:
    with rasterio.open(f"{region}/{region}_z50_SR_{b}.tif") as src:
        bands.append(src.read(1))

# Stack bands: (H, W, 4)
X = np.stack(bands, axis=-1)

# Load semi-supervised labels
with rasterio.open(f"{region}/{region}_z50_labeled_3pct.tif") as src:
    labels = src.read(1)
    # 0 = unlabeled, 1-6 = classes

# Flatten for clustering
X_flat = X.reshape(-1, 4)  # (N, 4)
y_flat = labels.flatten()  # (N,)
```

## Citation

If you use this dataset in your research, please cite:

```bibtex
@article{nguyen2026stsrsscfcm,
  title   = {Semi-Supervised Collaborative Fuzzy Clustering With a Softmax
             Regression-Based Self-Training Strategy for Landcover
             Classification From Remote Sensing Imagery},
  author  = {Nguyen, Xuan Hoang and Mai, Dinh Sinh and
             Nguyen, Long Giang and Dang, Trong Hop},
  journal = {Applied Soft Computing},
  year    = {2026}
}
```

## License

This dataset is provided for **research purposes only**. The original Landsat 8 imagery is courtesy of NASA/USGS and is in the public domain. The derived labels and preprocessing are provided under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

## Contact

For questions about this dataset, please open an issue on this repository or contact the corresponding author (hoangnx@haui.edu.vn).
