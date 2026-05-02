# HAS Algorithm — Implementation Plan

**Status:** scaffold present (`src/algorithms/has.jl`); core not yet implemented.  
**Reference:** Marchetti & Kleist Lab — <https://github.com/KleistLab/HAS>

---

## 1. Algorithm overview

Hierarchical Adaptive Sampling (HAS) is an exact continuous-time Monte Carlo
algorithm for large biochemical / epidemiological reaction networks.  It
achieves **O(log N) time per event** — independent of rate heterogeneity —
by organising per-node hazards in a complete binary tree of partial sums.

### Data structures

```
                 Λ (root = total rate)
               /          \
           Λ_L              Λ_R
          /    \           /    \
       Λ_LL  Λ_LR       Λ_RL  Λ_RR
       ...                        ...
    a_1  a_2  a_3  a_4  ...    a_N    (leaves = per-node hazards)
```

* `tree[1]` = root = total rate Λ
* `tree[k]` = sum of all leaf hazards under node `k`
* Leaf for node `v` (1-indexed): `tree[N_tree + v - 1]` where `N_tree` is
  the smallest power of 2 ≥ N (pad inactive leaves with 0).
* Parent of node `k`: `k >> 1`; children: `2k`, `2k+1`.

### Event selection — O(log N)

```
v ← 1                     # start at root
while v is an internal node:
    p_left = tree[2v] / tree[v]
    if rand() < p_left:
        v ← 2v            # go left
    else:
        v ← 2v + 1        # go right
return leaf_index(v)
```

### Per-event update — O(Δ log N)

When node `v` fires and changes compartment:

1. Recompute `a_v` (spontaneous + infection hazard) → O(1)
2. Update tree: `tree[leaf(v)] = a_v`, then propagate differences up:
   ```
   k ← parent(leaf(v))
   while k ≥ 1:
       tree[k] = tree[2k] + tree[2k+1]
       k ← k >> 1
   ```
   → O(log N) path
3. For each neighbour `u` of `v` (their infection hazard changed):
   repeat step 1–2 for `u` → O(Δ log N) total

### Time advance

Standard: `Δt = Exp(1) / Λ` where `Λ = tree[1]`.

---

## 2. Integration with `NetworkOutbreaks.jl`

### Struct

```julia
struct HAS <: OutbreakAlgorithm end
```

Already present in `src/algorithms/has.jl` (stub).

### Dispatch entry point

```julia
function _simulate_impl(::HAS, spec::OutbreakSpec, seed::UInt64, keep::Symbol)
    ...
end
```

### Internal state (planned)

```julia
mutable struct HASState
    tree::Vector{Float64}       # length 2 * N_tree; tree[1] = total rate
    N_tree::Int                 # next power of 2 ≥ N
    node_hazard::Vector{Float64} # length N; per-node cached hazard
end
```

Helper functions needed:
* `_has_leaf(v, N_tree)::Int`  — index of leaf for node v
* `_has_update!(state, v, new_h)` — update leaf and propagate up
* `_has_sample_node(state, rng)::Int` — O(log N) tree traversal
* `_has_total_hazard(v, ...)::Float64` — same as CR's `_cr_total_hazard`

---

## 3. Per-event update flow

```
event fires at node v:
  1. sample Δt = randexp(rng) / tree[1]; advance t
  2. sample v via _has_sample_node
  3. sample transition (spont vs infection, then specific tr)
  4. apply state change
  5. _has_update!(state, v, new_hazard(v))
  6. for u in neighbors(g, v):
         _has_update!(state, u, new_hazard(u))
```

---

## 4. Integration with `TimeVaryingNetwork` (planned)

When a graph update fires at time `t_update`:

1. Apply `add_edge!` / `rem_edge!` to the working copy of the graph.
2. For the two endpoint nodes `(src, dst)`: recompute their infection hazard
   and call `_has_update!` for each → 2 × O(log N) = O(log N).
3. Continue the main loop.

This is the same O(log N) cost per graph-update event as per reaction event.

---

## 5. Validation strategy

| Test | Method |
|---|---|
| Determinism | Same seed → identical trajectory (two calls, compare `.times` and `.counts`) |
| Statistical agreement with DirectSSA | 40-sim ensemble on k=4 r-regular N=400 SIS, max rel. diff ≤ 0.05 |
| Scalability benchmark | N ∈ {500, 2000, 10000}, time per event vs. DirectSSA and CR |
| Heterogeneous graph | Barabási–Albert (power-law degree) N=2000: confirm O(log N) per event |

---

## 6. Estimated effort

| Component | Lines | Notes |
|---|---|---|
| `HASState` struct + constructor | ~30 | |
| `_has_leaf`, `_has_update!` | ~20 | |
| `_has_sample_node` | ~20 | Tree traversal |
| `_has_total_hazard` | reuse `_cr_total_hazard` | |
| `_simulate_impl(::HAS, ...)` | ~80 | Main loop |
| Tests | ~40 | Determinism + agreement |
| **Total** | **~190** | |

---

## 7. References

* Marchetti et al. (KleistLab/HAS): Cython reference implementation.
* Gibson & Bruck (2000): next-reaction method — same O(log N) event
  selection, different data structure (priority queue vs. sum tree).
* Slepoy, Thompson & Plimpton (2008): Composition–Rejection — O(1) amortised
  via logarithmic bucketing; complementary to HAS.
