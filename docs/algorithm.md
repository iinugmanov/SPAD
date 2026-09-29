# SPAD algorithm

SPAD separates background seismicity from clustered seismicity and then
describes the background with a density filtration. The 2026 paper
reads dense background events as the seismic expression of tectonic
asperities. Locations of strong earthquakes are not an input.

Declustering follows Ostapchuk and Nugmanov (2026). The clustering
stage in this repository is an alpha-filtration of the canonical DPS
density of Agayan, Bogoutdinov and Dobrovolsky (2014). It builds a
condensed tree over the density level and reads persistent components
off that tree. The exponent `q` is an input. The program does not
claim a best `q`.

The filtration does not cut one level set with centroid linkage, and
it does not build convex-hull contours. Event trees are a separate
post-declustering tool. They are not part of the filtration.

## 1. Background seismicity

For earthquakes `i` and `j`, with `i` earlier than `j`, the proximity in
the paper is

```text
eta_ij = t_ij * (r_ij ** d_f) * 10 ** (-b * M_i)    if t_ij > 0
eta_ij = +infinity                                 if t_ij <= 0
```

`t_ij` is the inter-event time, `r_ij` the distance, `M_i` the magnitude
of the earlier event, `b` the Gutenberg-Richter parameter and `d_f` the
fractal dimension of the epicenters. The nearest neighbour of `j` is the
earlier event with the smallest `eta`.

The same quantity is factored as `eta = T * R` with

```text
T = t_ij * 10 ** (-b * M_i / 2)
R = (r_ij ** d_f) * 10 ** (-b * M_i / 2)
```

In the declustering script, `t_ij` is in days and `r_ij` is in
kilometres. The horizontal part is a haversine distance with Earth radius
6371 km, combined with the absolute depth difference:

```text
r = sqrt(horizontal_km ** 2 + depth_difference_km ** 2)
```

The depth column is `z_proj`, used as given. The declustering script
does not project hypocenters onto a slab. In the paper that projection
was done before declustering, with Slab2.

A second rule, stated in the paper, drops events inside a space-time
window of a magnitude `M` event: radius `0.02 * 10 ** (0.5 * M)` km and
duration `0.04 * 10 ** (0.55 * M)` days. The script always computes two
neighbours. One uses every earlier event. The other keeps `eta` finite
only when the raw distance and the raw time fall inside that window,
using the earlier event's magnitude. If the filtered neighbour has a
finite `eta`, it is kept and `passed_filter` is true. Otherwise the
unfiltered neighbour is kept. The script does not test whether the later
event is smaller than `M`.

`log10(eta)` of the selected neighbours is modeled as a two-component
Gaussian mixture (`covariance_type="full"`, `n_init=50`, `tol=1e-6`,
`max_iter=500`, `init_params="k-means++"`). The script fits the mixture
`n_trials` times (30 when run from the command line, seeds `2, 3, ...`).
For each fit it takes the point between the two component means where the
weighted normal densities are closest. It then keeps the fit whose
intersection is closest to the most common intersection rounded to two
decimals. The linear threshold is `eta0 = 10 ** intersection`. Rows with
`min_eta < eta0` are marked `Aftershock` (clustered mode). Non-finite
`min_eta` is replaced, before that comparison, by one more than the
largest finite value.

For the Kamchatka catalog the paper reports the separating value
`lg eta0 = -1.72`. That number is a result for that catalog. The script
recomputes the threshold from the file it is given. Equation (1) of the
paper gives `b = 0.93` for the same catalog (`log N = 7.25 - 0.93 M`).
The fractal dimension is an input of the script; the paper does not
quote a single default for it. The command-line help uses `1` and `1.6`
only as examples.

The distance routine reads its three columns as latitude, longitude and
depth. Declustering passes `latitude`, `longitude`, `z_proj` in that
order.

Time gaps use `astype("int64") // 10**9`, which is seconds when the
timestamp is `datetime64[ns]` (pandas 2.x).

## 2. Distance in the clustering stage

The clustering distance is the same 3D combination, with Earth radius
6371 km. The haversine argument is clipped to `[0, 1]`:

```text
d = sqrt(haversine(lat, lon)^2 + (depth_i - depth_j)^2)
```

Two depth choices are available.

* Catalog depth: the `depth` column, in kilometres.
* Slab depth: `depth = -z_proj`. This is the depth on the slab surface
  for a catalog whose `z_proj` is the signed distance to that surface.
  Latitude and longitude stay the catalog values. An inverse of
  EPSG:3395 is provided only as a coordinate check; it is not the
  distance.

Null epicentres are not on the original slab samples. In slab mode their
depth is assigned by a `SlabSurface` interpolant: linear interpolation
of the real events' `-z_proj` in local kilometres (equirectangular about
the mean latitude), and the nearest real epicentre outside the convex
hull. Epicentres that coincide after rounding those local coordinates
to 0.001 km are averaged first.

The radius at a negative exponent `q` is the power mean of the positive
pairwise distances:

```text
r(q) = (mean of d^q) ** (1/q)
```

Zero distances are left out of the mean. `r(q)` is recomputed for every
catalog, unless a run is explicitly scored at a fixed radius (the
fixed-`r` null comparison uses the real catalog's `r(q)`).

## 3. DPS density and the alpha-filtration

On a catalog `X` with radius `r`, the kernel weight is `1 - d/r` for
every pair with `d <= r`, including an event with itself (weight 1).
`C` is the maximum row sum. The density is that row sum divided by `C`:

```text
P(x) = (1/C) * sum_{y: d(x,y) <= r} (1 - d(x,y)/r)
```

`P` includes the point and is normalised by `C`. An alpha-perfect set
is the limit of the iteration that starts from `X` (or from a supplied
subset) and repeatedly keeps points whose density, recomputed on the
surviving subset and still divided by the original `C`, is at least
`alpha`. That is the canonical construction in Agayan et al. (2014),
not a single level set of `P`.

The alpha grid has 60 steps. `alpha[0]` is `min P`. `alpha[60]` is the
largest alpha with a non-empty perfect set, found by 40 bisection steps
starting from the set at `min P`. The sets that enter the tree are the
perfect sets at `alpha[0]` through `alpha[59]`. `alpha[60]` is only the
terminal value of a component that never dies inside the grid.

If a perfect set is not nested in the previous one, the level is still
used. The solution records `nested_ok = false` and does not repair the
level.

Components at a level are the connected components of the radius graph
(a single-linkage cut at `r`) inside the perfect set. Components smaller
than `m_min = 5` are dropped. The condensed tree keeps a component while
its points carry exactly one surviving child label. The component dies
when it splits into two or more such children, which are born at that
level, or when no child remains. A component still alive at the last
level dies at `alpha[60]`. Persistence is

```text
pi = alpha[death] - alpha[birth]
```

The node is alive on levels `birth, ..., death-1`.

### Birth-set reading (max-patches score)

Excess of mass on persistence selects a non-overlapping set of nodes.
Children replace a parent only when their total persistence is strictly
greater; a tie keeps the parent. Nodes born at the first level compete.
The patch is the component at birth. The score at that `q` is the number
of patches, then the number of events in those patches. The remaining
tie-break is the mean bin-distance between peaks of the density
histogram of all events and peaks of the histogram of the birth-set
events (shared Freedman-Diaconis bins). That peak comparison is an
empirical tie-break. It is not part of DPS. It is `+inf` when either
histogram has no peak.

The max-patches helper evaluates this score on

```text
q = -4.0, -3.8, ..., -1.0
```

and reports the maximiser. It is a summary of that grid, not a best `q`.

### Core reading (plateau helper)

A node is eligible when it lives at least 3 levels (`death - birth >= 3`;
on a 120-level grid the same fraction is 6) and it was not born at level
0. A node born at `alpha[0] = min P` is a component of the whole
catalog. Its birth is cut off by the start of the grid, so its
persistence is not used. Eligible nodes compete by persistence. A tie
keeps the parent. A node that is not eligible does not compete; an
eligible descendant of it still can.

The core of a selected node is the component at the median level of its
life,

```text
k_med = birth + floor((death - 1 - birth) / 2)
```

not the level just before death. Core size is at least `m_min`.

On the grid `q = -4.0, -3.9, ..., -1.0` (31 values), neighbouring
solutions match when they have the same number of cores and every core
has a best Jaccard match of at least 0.8 against the cores at the next
`q`. Jaccard is `|A ∩ B| / |A ∪ B|` on event indices. The match is
per patch. An event-weighted mean is not used. Cores inside one solution
are disjoint, so with equal counts a threshold above 0.5 is a one-to-one
match. Two empty solutions count as the same, and they still cannot form
a valid plateau.

A plateau is a maximal run of matching neighbours. It is valid when it
covers at least two grid points and the common patch count is at least
2. The helper reports the longest valid plateau. A tie goes to the
plateau whose first `q` is larger. The reported `q` is the middle grid
point; for an even length it is the upper of the two middle points.
The patches at that `q` are the cores. If the plateau touches `q = -4`
or `q = -1`, it is flagged as edge-censored. The flag does not move the
reported `q`. If no valid plateau exists, the helper reports none.

This helper is defined on that grid and those thresholds. It is not a
claim that the reported `q` is optimal. On any other grid the same
comparison only describes that scan.

## 4. Null catalogs, bootstrap, exceedance

Null catalogs keep the event count of the real catalog.

* Epicentre shift, `sigma = 50` km, seeds `1000 + i`. Each event moves
  by an isotropic Gaussian horizontal vector (`sigma` per component).
  The generator also draws `U(-10, 10)` km of depth and clips at 0.
* Epicentre shift, `sigma = 100` km, seeds `2000 + i`, same construction.
* Uniform in the union of 75 km horizontal discs about the real
  epicentres, seeds `3000 + i`. Draws are uniform in longitude and in
  `sin(latitude)` inside a bounding box, then rejected outside the
  discs. The generator assigns a permutation of the input depths.
* Bootstrap: 80% of the events without replacement, seeds `4000 + i`,
  indices returned in increasing order. Depth is the depth already
  chosen for those real events. Bootstrap replicas are not nulls.

In slab mode the shift and uniform depths drawn by the generator are
discarded and replaced by the slab-surface interpolant. The generator
is still called in full, so the seed stream matches a run that keeps
the drawn depths. A destination longitude is folded into `(-180, 180]`.
When every input longitude is positive, a negative destination
longitude is then increased by 360.

Each null catalog is solved twice on the same `q` grid: at its own
`r(q)`, and at the real catalog's `r` (fixed-radius comparison).
Bootstrap replicas are solved only at their own `r(q)`.

At each `q` the exceedance of a real score over one null model is

```text
z = (real - mean(null)) / sd(null)
```

with the sample standard deviation (`ddof=1`). If that deviation is
zero, `z` is 0 when the real value equals the null mean and signed
infinity otherwise. The empirical tail probabilities are

```text
p_upper = (1 + #{null >= real}) / (n_null + 1)
p_lower = (1 + #{null <= real}) / (n_null + 1)
```

The scores are the plateau-rule patch count, the core fraction, the
birth-set patch count, and the birth-set event fraction.

Bootstrap reproducibility, at each `q`, restricts every real core to
the replica's events and takes its best Jaccard match against the
replica's cores. The reported weight is the core size on the full real
catalog. `wmean_freq05` is the size-weighted mean of the fraction of
replicas with Jaccard at least 0.5. `frac_ge08` is the fraction of real
cores whose Jaccard-at-least-0.5 frequency is itself at least 0.8.
A replica with no core scores 0 against every real core. These
frequencies describe stability. They do not choose `q`.

## 5. Choosing q

Choose `q` from the scan, from the exceedance table (same `q` and fixed
real radius) and from the bootstrap frequencies. The max-patches helper
and the plateau helper are optional summaries with the definitions
above. Neither one is an automatic best `q`.

## Event trees

`form_event_trees.py` does not read `Aftershock`. It chains events with
`passed_filter` through `nearest_neighbor` until the neighbour did not
pass the filter. That neighbour's id is `tree_id`. A tree is kept only
when more than one filtered event shares it. The root row, whose
`event_id` equals that id, receives `parent = 0` and `tree_id` equal to
its own id. Other events in the tree receive `parent = nearest_neighbor`.
All other rows receive `parent = -1` and `group_id = 0`. Kept trees are
numbered from 1 in order of appearance.

`classify_event_types.py` drops `group_id == 0`. For each remaining group
that has a `parent == 0` row it measures time and horizontal distance
from that row, and the time and distance gaps along the group ordered by
time. `EQtype` follows the largest `mag` in the group, as described in
the module docstring.

`plot_event_trees.py` draws one figure per `group_id`. Arrows follow
`nearest_neighbor` when the neighbour is inside the group.

## Behaviour left as implemented

These are properties of the numerics. They are not corrected in this
package.

* Shift longitudes are folded into `(-180, 180]`, and then increased by
  360 when they are negative and every input longitude is positive. A
  shift across the antimeridian is stored near 360 rather than as a
  small negative longitude.
* Slab nulls draw the shift's depth jitter, or the uniform catalog's
  depth permutation, and then replace that depth with the interpolant.
  The draw still consumes the random stream.
* The alpha supremum is a 40-step bisection with a warm start at
  `min P`. A level that is not nested in the previous perfect set is
  kept; only `nested_ok` changes.
* Component labels are the order returned by SciPy
  `connected_components`. Patch numbers follow sorted node ids of the
  condensed tree, which follow that order.
* Excess of mass keeps the parent when the children's total persistence
  equals the parent's (`>` for the birth-set reading, `>=` for the
  plateau reading).
* Two neighbouring empty solutions count as the same in the plateau
  comparison. They cannot form a valid plateau, because a valid plateau
  needs at least two patches.
* Bootstrap Jaccard divides by patch size in the restricted labeling.
  A real core that is absent from an 80% subsample has size 0 there,
  and the ratio is undefined.
* The slab surface averages duplicated epicentres after rounding local
  east/north coordinates to 0.001 km, and uses the nearest epicentre
  outside the convex hull.
* The peak-similarity tie-break uses 10 bins when the Freedman-Diaconis
  width is 0, and returns `+inf` when either histogram has no peak.

## References

Ostapchuk, A.; Nugmanov, I. Background Seismicity Highlights Tectonic
Asperities. *Geosciences* 2026, *16*, 38.
<https://doi.org/10.3390/geosciences16010038>

Agayan, S. M.; Bogoutdinov, Sh. R.; Dobrovolsky, M. N. Discrete Perfect
Sets and Their Application in Cluster Analysis. *Cybernetics and Systems
Analysis* 2014, *50*(2), 176–190.
