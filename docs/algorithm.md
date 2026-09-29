# SPAD algorithm

SPAD (Seismogenic Patches Detection) separates background seismicity from
clustered seismicity and groups the background into dense sets. The paper
interprets those sets, and the larger patches that contain them, as the
seismic expression of tectonic asperities. The method does not use the
locations of strong earthquakes as input.

The paper organizes the method in three stages. The code in this
repository implements declustering, a level set of a DPS-style density,
and a centroid-linkage cut of the dense events. Convex-hull patch
contours are not computed. Event trees are a separate post-declustering
tool; they are not one of the three stages.

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

The depth column is `z_proj`, used as given. The script does not project
hypocenters onto a slab. In the paper that projection was done before
declustering, with Slab2.

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
depth. Declustering passes `latitude`, `longitude`, `z_proj` in that order.

## 2. Dense events

Background events are reduced to a level set of a DPS-style density.
The construction is the simple choice `X^1(alpha)` of Agayan,
Bogoutdinov and Dobrovolsky (2014). The script does not iterate to the
maximal alpha-perfect set.

For a set `W` and a negative exponent `q`, the localization radius is
the power mean of the positive pairwise distances `D(W)`:

```text
r_q(W) = (sum(d ** q) / |D(W)|) ** (1 / q)
```

The density at `w` uses the kernel `1 - d/r`. The point itself is
excluded, and the sum is not divided by `max(P)`:

```text
P(w) = sum over other xi with d(w, xi) <= r of (1 - d(w, xi) / r)
```

The level is

```text
alpha = (1 - beta) * max(P)
```

Dense events are `{w : P(w) >= alpha}`. When `max(P)` is 0 the code
returns `alpha = 0`, which is the same product.

The script searches

```text
q    = numpy.arange(-2.9, -0.1, 0.1)
beta = numpy.arange(-1.0, 1.0, 0.1)
```

`arange` does not include the stop value, so `beta` runs from `-1.0`
through `0.9` and `q` from `-2.9` up to, but not including, `-0.1`.
A configuration is skipped when fewer than two events pass `alpha`.
Among the rest, `q` and `beta` are chosen by an empirical SPAD
heuristic that is not part of DPS. The heuristic minimizes the mean
bin-distance between peaks of the density histogram of all background
events and peaks of the histogram of the dense subset. On a tie it
keeps the larger number of clusters. Those clusters are a
centroid-linkage cut of the dense subset at the radius `r` of that
configuration. They are stored as `dense_group_id` (relabeled from 0;
events outside the subset are `-1`).

Rows with `Aftershock == True` are removed before this stage. If the
column is absent, every row is kept. `magnitude` and `time` must be
present; the distance calculation does not use them.

Agayan, S. M.; Bogoutdinov, Sh. R.; Dobrovolsky, M. N. Discrete Perfect
Sets and Their Application in Cluster Analysis. *Cybernetics and Systems
Analysis* 2014, 50(2), 176–190.

## 3. Seismogenic patches

The final patches are a second centroid-linkage cut
(`method="centroid"`) of the selected dense events. The cut distance is
the global mode of the pairwise distances of the background catalog:
the maximum of the Freedman-Diaconis histogram (the center of its
tallest bin). The labels are written to `cluster`. The script saves the
pairwise-distance histogram (40 bins in the figure; the mode itself
uses the Freedman-Diaconis count) and the dendrogram. It does not
compute convex hulls.

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
