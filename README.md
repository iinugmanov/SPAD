# SPAD

SPAD (Seismogenic Patches Detection) identifies seismogenic patches from
an earthquake catalog. The catalog is declustered with a nearest-neighbour
proximity in time, space and magnitude. Background events are then
described by an alpha-filtration of a DPS density. The exponent `q` sets
the radius. Persistent components of that filtration are the patches.
The paper interprets dense background events as the seismic expression
of tectonic asperities. Locations of strong earthquakes are not an input.

Declustering follows Ostapchuk and Nugmanov (2026). The clustering stage
is the filtration below, not a single density level set and not a
centroid-linkage cut. Formulas are in [`docs/algorithm.md`](docs/algorithm.md).

`q` is chosen by the user. The scan, the null exceedance and the
bootstrap frequencies are the diagnostics. A max-patches summary and a
plateau summary are optional helpers. Neither helper is a best `q`.

## Algorithm

1. **Declustering.** For each event the nearest earlier event minimizes
   `eta = T * R`, with `T` and `R` the time and distance rescaled by the
   Gutenberg-Richter `b` value and the fractal dimension `d`. A
   space-time window (`0.02 * 10**(0.5*M)` km and `0.04 * 10**(0.55*M)`
   days) supplies a preferred neighbour when a pair falls inside it.
   A two-component Gaussian mixture of `log10(eta)` sets the threshold
   `eta0`. Events with `eta < eta0` are the clustered mode (`Aftershock`);
   the rest are background.
2. **Radius and density.** On the background, `r(q)` is the power mean
   of the positive pairwise distances at a negative exponent `q`. The
   distance is a haversine separation of latitude and longitude (Earth
   radius 6371 km) combined with the depth difference. Depth is either
   the catalog `depth` or the slab depth `-z_proj`. The DPS density uses
   the kernel `1 - d/r`, includes the event itself, and divides by the
   maximum row sum. An alpha-perfect set is the iterated trim of Agayan
   et al. (2014): a point stays only while its density on the surviving
   set is at least alpha. Sixty alpha levels run from the minimum density
   to the largest alpha with a non-empty perfect set.
3. **Tree and patches.** Components are the connected components of the
   radius graph, of size at least 5. The condensed tree follows a
   component until it splits or disappears. Persistence is the length of
   its alpha interval. One reading (the max-patches score) is excess of
   mass on that persistence, with each patch equal to the component at
   birth. The other reading (the plateau helper) keeps nodes that live
   at least three levels and were not born at the first level, and takes
   the component at the median level of its life as the core. The user
   decides which `q` to keep.

## Pipeline

Declustering is shared. The filtration and the event trees are separate
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
    v                               v
spad.cli qscan / patches /     spad/form_event_trees.py
ensemble / exceedance              catalog_with_tree.csv
    per-q table, patches               |
    null and bootstrap pickles         v
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

The declustering script does not project hypocenters. In the paper,
events were projected onto the slab with Slab2 before this step;
`z_proj` is whatever depth the file already contains.

After declustering, `event_id` is the row index of the catalog sorted by
time, and `nearest_neighbor` is an `event_id`.

### Filtration

| Column | Type | Role |
| --- | --- | --- |
| `lat` or `latitude` | float, degrees | Epicentre |
| `lon` or `longitude` | float, degrees | Epicentre |
| `depth` | float, km | Used when `--depth catalog` |
| `z_proj` | float, km | Slab depth is `-z_proj` when `--depth slab` |
| `Aftershock` | boolean | Rows equal to true are removed. Optional. |

`--depth catalog` (the default) uses `depth`. `--depth slab` uses
`-z_proj` and, for shift and uniform nulls, replaces depth with the
slab-surface interpolant of the real `-z_proj` values. Bootstrap
replicas keep the depth of the events they retain.

`magnitude` and `time` are not used by the filtration.

## Outputs

**Declustering** (`catalog_with_aftershock.csv`): the input columns plus
`event_id`, `min_eta`, `nearest_neighbor`, `min_T`, `min_R`,
`passed_filter`, `Aftershock`. Figures: `disturb_eta.pdf` (distribution
of `log10(eta)`), `GMM.pdf` (mixture and threshold), `Declustered_T_R.pdf`
(joint distribution of `log10(T)` and `log10(R)`).

**Filtration.** `qscan` writes a table with, at each `q`, the radius,
the birth-set patch count and event count, the peak-similarity
tie-break, the plateau-rule patch count, the core event count and the
core fraction. `patches` writes event labels and a patch summary at one
user-given `q`. Labels in the event file are 1-based; 0 means the event
is outside that reading. `eom_patch` is the birth-set excess-of-mass
label. `core_patch` is the plateau-rule core at that same `q` (it is not
the result of searching the plateau). `ensemble` writes one pickle per
replica under `--out-dir`. `exceedance` writes `z`, empirical tail
probabilities and, when bootstrap replicas are present, the
reproducibility table. Figures are written only to paths passed on the
command line.

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
| `--depth` | filtration | `catalog` or `slab`. |
| `--q-min`, `--q-max`, `--q-step` | filtration | Default grid `-4` to `-1` in steps of `0.1` (31 values). Step `0.2` is the max-patches grid. |
| `--q` | `patches` | One exponent. Required. Not chosen by the program. |
| `--n-shift50`, `--n-shift100`, `--n-unif`, `--n-boot` | ensemble | Replica counts. Seeds are `1000+i`, `2000+i`, `3000+i`, `4000+i`. |
| `--workers` | ensemble | Spawned processes. `1` runs in the current process. |

The mixture threshold is estimated from the catalog. The paper's
`lg eta0 = -1.72` is the value obtained for the Kamchatka catalog, not a
constant in the script.

The filtration defaults are 60 alpha levels and a minimum component size
of 5. Null shift uses `sigma` of 50 or 100 km. The uniform null uses a
75 km buffer. Bootstrap replicas keep 80% of the events without
replacement.

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

python -m spad.cli qscan \
    --catalog background.csv \
    --depth slab \
    --output results/qscan.csv

python -m spad.cli patches \
    --catalog background.csv \
    --depth slab \
    --q -2.5 \
    --events results/events.csv \
    --patches results/cores.csv

python -m spad.cli ensemble \
    --catalog background.csv \
    --depth slab \
    --out-dir results/ensemble \
    --n-shift50 2 \
    --n-boot 2 \
    --workers 2

python -m spad.cli exceedance \
    --runs results/ensemble \
    --output results/exceedance.csv \
    --bootstrap-output results/bootstrap.csv

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

`qscan` prints the max-patches helper and the plateau helper after the
table. Those lines restate the definitions in `docs/algorithm.md`. They
are not a selected `q`. Pick `q`, then use `patches` at that value.

Each script also accepts `--help`.

## Tests

```bash
python -m pytest
```

The tests use small synthetic catalogs. They check the distance, tree
invariants, seed stability of the null and bootstrap generators, and a
frozen `q` scan.

## Citation

Ostapchuk, A.; Nugmanov, I. Background Seismicity Highlights Tectonic
Asperities. *Geosciences* **2026**, *16*, 38.
<https://doi.org/10.3390/geosciences16010038>

Agayan, S. M.; Bogoutdinov, Sh. R.; Dobrovolsky, M. N. Discrete Perfect
Sets and Their Application in Cluster Analysis. *Cybernetics and Systems
Analysis* **2014**, *50*(2), 176–190.
