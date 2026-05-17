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

| Type | Description | `TimeVaryingNetwork` |
|---|---|:---:|
| `DirectSSA()` | Gillespie direct method — O(N) per event | ✓ |
| `NextReaction()` | Gibson–Bruck next-reaction — O(log N) per event, local updates | ✓ |
| `CompositionRejection()` | Slepoy *et al.* 2008 — O(1) amortised via log-bucket composition + rejection; fastest for fixed-degree graphs at N ≳ 2000 | ✗ |
| `HAS()` | Hierarchical Adaptive Sampling — O(log N) deterministic via sum tree | ✓ |

Pass any algorithm to `simulate` or `simulate_ensemble`:

```julia
traj = simulate(spec; algorithm = CompositionRejection(), seed = 42)
ens  = simulate_ensemble(spec; nsims = 80, seed = 1, algorithm = NextReaction(),
                         parallel = true)
```

## Time-varying networks

Wrap a mutable `AbstractGraph` with a sorted list of edge-level updates to
get a `TimeVaryingNetwork`.  `DirectSSA`, `NextReaction`, and `HAS` interleave
the graph mutations with reactions automatically. `CompositionRejection`
currently rejects time-varying topologies with an `ArgumentError`.

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

### Package-extension adapters

`NetworkOutbreaks` ships two
[package extensions](https://pkgdocs.julialang.org/v1/creating-packages/#Conditional-loading-of-code-in-packages-(Extensions))
that activate when `EdgeBasedModels` or `NodeBasedModels` is present in the
environment. Each extension defines an `OutbreakModel(...)` constructor that
converts the respective package's model type.

These adapters are declared through `[weakdeps]`, so `NetworkOutbreaks` does
not force either deterministic modelling package to load in ordinary use.
If an environment still reports a circular precompilation warning, resolve it
against the current `Project.toml` files so the weak-dependency metadata is
used.

## Release highlights

- Four simulation algorithms: `DirectSSA`, `NextReaction`,
  `CompositionRejection`, and `HAS`.
- `TimeVaryingNetwork` and `MultiplexNetwork` support.
- **Interventions**: `ScheduledRateChange`, `ScheduledStateChange` (vaccination
  pulses), `ThresholdIntervention` (reactive policies) — processed by `DirectSSA`.
- **Contact tracing**: `:contact_trace` transition type for rate-based
  neighbour-dependent quarantine (preserves Gillespie correctness).
- Threaded deterministic ensembles via `simulate_ensemble`.
- Weak-dependency adapters from `EdgeBasedModels` and `NodeBasedModels`.

## License

NetworkOutbreaks.jl is licensed under the MIT License.
