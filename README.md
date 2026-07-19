# StSr-SSCFCM — Datasets

Data accompanying the paper:

> **Semi-Supervised Collaborative Fuzzy Clustering With a Softmax Regression-Based Self-Training Strategy for Landcover Classification From Remote Sensing Imagery**
> Xuan Hoang Nguyen, Dinh Sinh Mai, Long Giang Nguyen, Trong Hop Dang.

The paper reports **four experiments**. This repository hosts the two Landsat-8 datasets that we
built ourselves (Experiments 2 and 3). Experiments 1 and 4 use established public benchmarks and
are **not** redistributed here — please obtain them from their original providers (links below) so
that the canonical versions and citations are preserved.

| # | Experiment | Dataset | Where to get it |
|---|-----------|---------|-----------------|
| 1 | Parameter sanity-check | Statlog **Landsat Satellite** (satimage) | UCI ML Repository, ID 146 — <https://archive.ics.uci.edu/dataset/146/statlog+landsat+satellite> |
| 2 | Single-region classification | **Landsat-8 Hanoi** | This repo → `landsat8/hn/` |
| 3 | Collaborative classification | **Landsat-8 Hanoi + Thanh Hoa + Ho Chi Minh City** | This repo → `landsat8/{hn,th2,hcm2}/` |
| 4 | Hyperspectral validation | **Indian Pines** (AVIRIS, 200 bands, 16 classes) | Purdue / GIC benchmark — <https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Indian_Pines> |

---

## Landsat-8 data (Experiments 2 and 3)

### Acquisition and preprocessing

| Property | Value |
|---|---|
| Satellite | Landsat 8 OLI/TIRS (NASA/USGS) |
| Processing level | Surface Reflectance (SR) |
| Composite | 2020–2023 median, cloud-free |
| Source platform | Google Earth Engine |
| Spatial resolution | 30 m/pixel |
| Coordinate system | EPSG:4326 (WGS84) |
| Spectral bands used | B2 (blue), B3 (green), B4 (red), B5 (NIR) |

### Regions

| Region | Code | Size (W × H) | Pixels | Bounding box (lon / lat) |
|--------|------|--------------|--------|--------------------------|
| Hanoi | `hn` | 1201 × 1001 | 1,202,201 | 105.580–106.120°E / 20.850–21.300°N |
| Thanh Hoa | `th2` | 981 × 825 | 809,325 | 105.400–105.840°E / 19.740–20.110°N |
| Ho Chi Minh City | `hcm2` | 1336 × 1114 | 1,488,304 | 106.370–106.970°E / 10.675–11.175°N |
| **Total** | | | **3,499,830** | |

### Ground truth — ESA WorldCover 2021

Reference labels come from an **independent expert source**, not from any clustering output:
ESA WorldCover 2021 (10 m, v200; CC BY 4.0), DOI [10.5281/zenodo.7254221](https://doi.org/10.5281/zenodo.7254221).

The 10 m product is aggregated to the Landsat grid by majority vote and remapped to five
land-cover classes. Shrubland is merged into Forest because it covers < 0.06 % of the study areas.

| Value | Class | Hanoi | Thanh Hoa | Ho Chi Minh |
|-------|-------|-------|-----------|-------------|
| 0 | Water | 4.39 % | 2.67 % | 4.09 % |
| 1 | Built-up | 26.58 % | 9.25 % | 30.07 % |
| 2 | Agriculture | 44.74 % | 44.84 % | 29.06 % |
| 3 | Perennial vegetation | 11.30 % | 5.35 % | 9.78 % |
| 4 | Forest | 12.99 % | 37.89 % | 27.00 % |

`255` denotes NoData (none of the three regions contain NoData pixels).

Each region was verified at pixel level after construction: grid alignment against the Landsat
bands (shape, CRS, affine transform), class distribution and NoData share, plus a visual overlay
check — the latter is included as `*_gt_validation.png`.

### Files per region

```
landsat8/{region}/
  {region}_z50_SR_B2.tif                     Blue  (0.452-0.512 um), Float32
  {region}_z50_SR_B3.tif                     Green (0.533-0.590 um), Float32
  {region}_z50_SR_B4.tif                     Red   (0.636-0.673 um), Float32
  {region}_z50_SR_B5.tif                     NIR   (0.851-0.879 um), Float32
  {region}_z50_RGB.tif                       RGB composite (B4,B3,B2), UInt8
  {region}_z50_WorldCover2021_5class.tif     Ground truth, UInt8, values 0-4 (255 = NoData)
  {region}_z50_WorldCover2021_5class_rgb.tif Ground truth, colour-coded for display
  {region}_z50_gt_validation.png             Visual check: RGB | ground truth overlay
```

---

## Reproducing the experimental protocol

Labels and evaluation splits are **generated at run time from the ground truth** — they are not
stored as files, so that any seed can be reproduced exactly.

**Semi-supervised labels.** Circular labelled regions are sampled per class
(radius 30 px, `n_regions_per_class` = 10 / 14 / 24 for `hn` / `th2` / `hcm2`), giving:

| Region | Labelled pixels | Ratio |
|--------|-----------------|-------|
| Hanoi | 45,218 | 3.76 % |
| Thanh Hoa | 58,280 | 7.20 % |
| Ho Chi Minh City | 106,568 | 7.16 % |

**Spatially disjoint train/test split.** Each image is tiled into non-overlapping 32 × 32 pixel
blocks; every pixel of a block is assigned wholly to train or test (~30 % of blocks held out for
testing). All labelled pixels lie in the training partition, and external metrics (ACC, F1, ARI,
NMI) are computed on the test partition only. This prevents a test pixel from ever being spatially
adjacent to a labelled one.

Each configuration is run over 10 random seeds (varying labelled-region placement, initialisation
and the block split). Fixed parameters: `m` = 2, `eta` = 0.95, `gamma` = 0.01, `SEED` = 42 (base).

---

## Loading example

```python
import rasterio, numpy as np

region, root = "hn", "landsat8/hn"
bands = [rasterio.open(f"{root}/{region}_z50_SR_B{b}.tif").read(1) for b in (2, 3, 4, 5)]
X = np.stack(bands, axis=-1).reshape(-1, 4)          # (N, 4) spectral features

with rasterio.open(f"{root}/{region}_z50_WorldCover2021_5class.tif") as src:
    y = src.read(1).reshape(-1)                       # (N,) labels 0-4, 255 = NoData
valid = y != 255
```

Features are z-score normalised **per site** (zero mean, unit variance per band, estimated on that
site's own pixels) before clustering — identically for every method compared in the paper.

---

## Licence and citation

- Landsat-8 imagery: courtesy of the U.S. Geological Survey, public domain.
- Ground truth derived from ESA WorldCover 2021 v200, licensed **CC BY 4.0** — please cite
  Zanaga et al. (2022), DOI [10.5281/zenodo.7254221](https://doi.org/10.5281/zenodo.7254221).
- Derived products in this repository are released under **CC BY 4.0**.

If you use this data, please cite the paper above and the ESA WorldCover product.

---

## Revision note

Earlier versions of this repository (before July 2026) shipped label masks named
`*_labeled_{3,7,8}pct.tif`, which had been produced by running FCM on the imagery itself. Those
files were removed: using clustering output as reference labels makes the evaluation circular.
The current release evaluates exclusively against the independent ESA WorldCover ground truth
described above.
