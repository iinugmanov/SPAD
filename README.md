# SPAD

SPAD (Seismogenic Patches Detection) identifies seismogenic patches from
an earthquake catalog. The catalog is declustered with a nearest-neighbour
proximity in time, space and magnitude. Background events are then reduced
to a level set of a DPS-style density, and those dense events are cut into
patches by centroid-linkage clustering. The paper interprets these patches
as the seismic expression of tectonic asperities. Locations of strong
earthquakes are not an input.

The three stages below follow Ostapchuk and Nugmanov (2026). Formulas and
the dense-event selection as implemented are in
[`docs/algorithm.md`](docs/algorithm.md).

## Algorithm

1. **Declustering.** For each event the nearest earlier event minimizes
   `eta = T * R`, with `T` and `R` the time and distance rescaled by the
   Gutenberg-Richter `b` value and the fractal dimension `d`. A
   space-time window (`0.02 * 10**(0.5*M)` km and `0.04 * 10**(0.55*M)`
   days) supplies a preferred neighbour when a pair falls inside it.
   A two-component Gaussian mixture of `log10(eta)` sets the threshold
   `eta0`. Events with `eta < eta0` are the clustered mode (`Aftershock`);
   the rest are background.
2. **Dense events.** On the background, the radius `r` is the power mean
   of the positive pairwise distances at a negative exponent `q`. The
   density `P` uses the kernel `1 - d/r`. The point itself is excluded,
   and `P` is not divided by its maximum. Dense events are the level set
   `{P >= alpha}` with `alpha = (1 - beta) * max(P)`. That is the simple
   choice `X^1(alpha)` of Agayan et al. (2014). The script does not
   iterate to the maximal alpha-perfect set. A grid of `q` and `beta` is
   scored by an empirical SPAD heuristic that is not part of DPS: the mean
   distance between peaks of the density histogram of all background
   events and peaks of the histogram of the dense subset. On a tie the
   larger number of clusters is kept. Inside the search those clusters
   are a centroid-linkage cut at the radius `r`.
3. **Patches.** The selected dense events are cut again by centroid
   linkage, at the global mode of the background pairwise-distance
   histogram (the maximum of the Freedman-Diaconis histogram). The script
   writes cluster labels, a distance histogram and a dendrogram. It does
   not build convex hulls.

Declustering and the dense-event stage use a 3D distance: haversine
separation plus the depth difference, with Earth radius 6371 km.

## Pipeline

Declustering is shared. Patch identification and event trees are separate
branches. They do not read each other's outputs.

```text
catalog.csv
    |
    v
spad/declustering.py
    catalog_with_aftershock.csv
    disturb_eta.pdf, GMM.pdf, Declustered_T_R.pdf
    |
    +-------------------------------+
    |                               |
    | rename columns                |
    v                               v
spad/identify_patches.py     spad/form_event_trees.py
    dense events, clusters        catalog_with_tree.csv
    histogram, dendrogram             |
                                      v
                              spad/classify_event_types.py
                                  catalog_with_types.csv
                                      |
                                      v
                              spad/plot_event_trees.py
                                  group_<id>_tree.png
```

`form_event_trees.py` chains rows with `passed_filter`, not rows with
`Aftershock`. Those two flags are different: `passed_filter` means the
space-time window contained a neighbour, while `Aftershock` means
`min_eta` is below the mixture threshold.

## Input

### Declustering, trees and event types

| Column | Type | Role |
| --- | --- | --- |
| `longitude` | float, degrees | Horizontal position |
| `latitude` | float, degrees | Horizontal position |
| `z_proj` | float, km | Depth used in the 3D distance |
| `time` | datetime | Mixed ISO-8601 text; parsed as UTC |
| `mag` | float | Magnitude |

```csv
longitude,latitude,z_proj,time,mag
160.25,52.10,30.0,2020-03-01 04:12:00+00:00,4.6
```

The script does not project hypocenters. In the paper, events were
projected onto the slab with Slab2 before this step; `z_proj` is whatever
depth the file already contains.

After declustering, `event_id` is the row index of the catalog sorted by
time, and `nearest_neighbor` is an `event_id`.

### SPAD

| Column | Type | Example |
| --- | --- | --- |
| `lat` | float, degrees | 37.7749 |
| `lon` | float, degrees | -122.4194 |
| `depth` | float, km | 10.0 |
| `time` | datetime, ISO 8601 | 2023-01-01 12:30:45.123456+00:00 |
| `magnitude` | float | 4.5 |
| `Aftershock` | boolean | False |

Rows with `Aftershock == True` are removed. If the column is missing,
every row is kept. `magnitude` and `time` are required and are not used
in the distance or the clustering.

A declustered file can be renamed into this schema. `z_proj` is copied
into `depth` with no further calculation:

```python
df = pd.read_csv("catalog_with_aftershock.csv")
df = df.rename(columns={
    "latitude": "lat",
    "longitude": "lon",
    "z_proj": "depth",
    "mag": "magnitude",
})
df.to_csv("spad_input.csv", index=False)
```

## Outputs

**Declustering** (`catalog_with_aftershock.csv`): the input columns plus
`event_id`, `min_eta`, `nearest_neighbor`, `min_T`, `min_R`,
`passed_filter`, `Aftershock`. Figures: `disturb_eta.pdf` (distribution
of `log10(eta)`), `GMM.pdf` (mixture and threshold), `Declustered_T_R.pdf`
(joint distribution of `log10(T)` and `log10(R)`).

**SPAD:** background events with `dense_group_id` (`-1` outside the dense
subset); dense events with `cluster`; a histogram table
(`Distance_all_km`, `Frequency_all`, `Distance_dense_km`,
`Frequency_dense`); the histogram and dendrogram as PDF.

**Trees:** `tree_id`, `parent` (`0` at the root, `nearest_neighbor` inside
a tree, `-1` otherwise), `group_id` (0 if the event is not in a kept
tree). A tree is kept only when more than one filtered event shares the
same root.

**Event types:** groups with `group_id != 0` that have a `parent == 0`
row. Added columns: `time_from_head_day`, `dist_from_head_km`,
`time_diff_seq`, `dist_diff_seq`, `EQtype` (`mainshock`, `foreshock`,
`aftershock`). One PNG per group, axes time and distance from the parent,
arrows along `nearest_neighbor`.

## Parameters

| Parameter | Where | Meaning |
| --- | --- | --- |
| `--b-value` | declustering | Gutenberg-Richter `b`. Example on the command line: `1`. The paper's Kamchatka fit is `b = 0.93`. |
| `--d-value` | declustering | Fractal dimension `d` (`d_f` in the paper). Example: `1.6`. |
| `--n-trials`, `--random-seed` | declustering | Mixture refits. Defaults `30` and `2`, which are the original script values. |
| `--interpolation` | declustering | Heatmap interpolation. Default `bicubic`. |
| `q`, `beta` | SPAD, fixed grid | `q` in `numpy.arange(-2.9, -0.1, 0.1)`, `beta` in `numpy.arange(-1.0, 1.0, 0.1)`. Not command-line arguments. |

The mixture threshold is estimated from the catalog. The paper's
`lg eta0 = -1.72` is the value obtained for the Kamchatka catalog, not a
constant in the script.

Time gaps in declustering and in `time_diff_seq` are
`astype("int64") // 10**9 / 86400`, which is days when the timestamp is
`datetime64[ns]` (pandas 2.x).

## Installation

The scripts were written for Python 3.11.7.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the commands below from the repository root.

## Usage

```bash
python -m spad.declustering \
    --catalog catalog.csv \
    --b-value 1 \
    --d-value 1.6 \
    --output-dir results/decluster

python -m spad.identify_patches \
    --input spad_input.csv \
    --dense-events results/dense_events.csv \
    --clustered-results results/patches.csv \
    --histogram-csv results/histogram.csv \
    --histogram-plot results/histogram.pdf \
    --dendrogram-plot results/dendrogram.pdf

python -m spad.form_event_trees \
    --input results/decluster/catalog_with_aftershock.csv \
    --output results/catalog_with_tree.csv

python -m spad.classify_event_types \
    --input results/catalog_with_tree.csv \
    --output results/catalog_with_types.csv

python -m spad.plot_event_trees \
    --input results/catalog_with_types.csv \
    --output-dir results/event_trees
```

Each script also accepts `--help`.

## Citation

Ostapchuk, A.; Nugmanov, I. Background Seismicity Highlights Tectonic
Asperities. *Geosciences* **2026**, *16*, 38.
<https://doi.org/10.3390/geosciences16010038>

Agayan, S. M.; Bogoutdinov, Sh. R.; Dobrovolsky, M. N. Discrete Perfect
Sets and Their Application in Cluster Analysis. *Cybernetics and Systems
Analysis* **2014**, *50*(2), 176–190.
