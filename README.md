# NetworkOutbreaks.jl

Stochastic compartmental epidemic simulation on contact networks, with
adapters for
[EdgeBasedModels.jl](https://github.com/sdwfrost/EdgeBasedModels.jl) and
[NodeBasedModels.jl](https://github.com/sdwfrost/NodeBasedModels.jl).

## Quick start

```julia
using NetworkOutbreaks, Graphs, StableRNGs

model = OutbreakModel([:S, :I, :R],
                      [:I],
                      [OutbreakTransition(:S, :I, 0.5, :infection),
                       OutbreakTransition(:I, :R, 1.0, :spontaneous)])

g    = random_regular_graph(500, 4; rng = StableRNG(42))
spec = OutbreakSpec(model   = model,
                   network  = g,
                   initial  = SeedFraction(:I => 0.02),
                   tspan    = (0.0, 50.0))

traj = simulate(spec; algorithm = DirectSSA(), seed = 1)
ens  = simulate_ensemble(spec; nsims = 40, seed = 2, parallel = true)
t, μI = mean_curve(ens, :I)
```

## Algorithms

| Type | Description |
|---|---|
| `DirectSSA()` | Gillespie direct method — O(N) per event |
| `NextReaction()` | Gibson–Bruck next-reaction — O(log N) per event, local updates |
| `CompositionRejection()` | Slepoy *et al.* 2008 — O(1) amortised via log-bucket composition + rejection; fastest for fixed-degree graphs at N ≳ 2000 |
| `HAS()` | Hierarchical Adaptive Sampling — O(log N) deterministic via sum tree; **not yet implemented** (see `docs/HAS_PLAN.md`) |

Pass any algorithm to `simulate` or `simulate_ensemble`:

```julia
traj = simulate(spec; algorithm = CompositionRejection(), seed = 42)
ens  = simulate_ensemble(spec; nsims = 80, seed = 1, algorithm = NextReaction(),
                         parallel = true)
```

## Time-varying networks

Wrap a mutable `AbstractGraph` with a sorted list of edge-level updates to
get a `TimeVaryingNetwork`.  `DirectSSA` and `NextReaction` interleave the
graph mutations with reactions automatically:

```julia
using Graphs
g = random_regular_graph(500, 3)
updates = [
    (t = 10.0, src = 1, dst = 42, action = :add),
    (t = 20.0, src = 7, dst = 13, action = :remove),
]
tvn  = TimeVaryingNetwork(deepcopy(g), updates)
spec = OutbreakSpec(model = model, network = tvn,
                    initial = SeedFraction(:I => 0.05),
                    tspan   = (0.0, 40.0))
ens  = simulate_ensemble(spec; nsims = 50, seed = 2, algorithm = NextReaction())
```

A plain `AbstractGraph` passed to `OutbreakSpec` is auto-wrapped in
`StaticNetwork`; all existing call sites continue to work unchanged.

## Known issues

### Circular-dependency precompilation warning

When loading `NetworkOutbreaks` together with `EdgeBasedModels` or
`NodeBasedModels`, Julia emits a warning of the form:

```
┌ Warning: Circular dependency detected.
│ Precompilation will be skipped for dependencies in this cycle:
│  ┌ NetworkOutbreaks
│  │  NodeBasedModels
│  └── NetworkOutbreaks → NetworkOutbreaksEdgeBasedModelsExt
```

This warning is cosmetic. The root cause is a mutual dependency:

- `NetworkOutbreaks` ships two
  [package extensions](https://pkgdocs.julialang.org/v1/creating-packages/#Conditional-loading-of-code-in-packages-(Extensions))
  that activate when `EdgeBasedModels` or `NodeBasedModels` is present in the
  environment. Each extension defines an `OutbreakModel(...)` constructor that
  converts the respective package's model type.
- `EdgeBasedModels` and `NodeBasedModels` both depend on `NetworkOutbreaks`
  (they call `simulate_ensemble` in their own code and tests).

Julia's precompilation graph therefore contains a cycle:
`NetworkOutbreaks → ext/EBMExt → EdgeBasedModels → NetworkOutbreaks`.
Julia handles this cycle gracefully at runtime — all packages and their
extensions load correctly — but skips caching the affected modules, so
compilation is repeated on each start-up.

**Possible fix (not yet implemented):** relocate the two adapter
constructors from `NetworkOutbreaks`'s `ext/` directory into corresponding
extension files inside `EdgeBasedModels` and `NodeBasedModels` themselves
(activated when `NetworkOutbreaks` is loaded). This reversal removes the
cycle. The refactor was deferred because it would require editing the
downstream package sources and is low-risk but non-trivial (the adapters
are ~60 lines each).
