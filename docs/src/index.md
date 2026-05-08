# NetworkOutbreaks.jl

Stochastic compartmental epidemic simulation on contact networks, with
adapters for
[EdgeBasedModels.jl](https://epirecip.es/EdgeBasedModels.jl/) and
[NodeBasedModels.jl](https://epirecip.es/NodeBasedModels.jl/).

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/epirecipes/NetworkOutbreaks.jl")
```

## Quick start

```julia
using NetworkOutbreaks, Graphs, StableRNGs

model = OutbreakModel([:S, :I, :R],
                      [:I],
                      [OutbreakTransition(:S, :I, 0.5, :infection),
                       OutbreakTransition(:I, :R, 1.0, :spontaneous)])

g    = random_regular_graph(500, 4; rng = StableRNG(42))
spec = OutbreakSpec(model   = model,
                    network = g,
                    initial = SeedFraction(:I => 0.02),
                    tspan   = (0.0, 50.0))

traj = simulate(spec; algorithm = DirectSSA(), seed = 1)
ens  = simulate_ensemble(spec; nsims = 40, seed = 2, parallel = true)
```

## Algorithms

| Type | Description | `TimeVaryingNetwork` |
|---|---|:---:|
| `DirectSSA()` | Gillespie direct method — O(N) per event | ✓ |
| `NextReaction()` | Gibson–Bruck next-reaction — O(log N) per event | ✓ |
| `CompositionRejection()` | Slepoy *et al.* 2008 — O(1) amortised | ✗ |
| `HAS()` | Hierarchical Adaptive Sampling — O(log N) deterministic | ✓ |

## Documentation contents

- [Vignettes](vignettes.md) — algorithm comparison and reinfection-counting
  validation.
- [API reference](api.md) — exported types and functions.

## Companion packages

- [EdgeBasedModels.jl](https://epirecip.es/EdgeBasedModels.jl/) — edge-based
  compartmental models on configuration-model networks.
- [NodeBasedModels.jl](https://epirecip.es/NodeBasedModels.jl/) — node- and
  pair-level approximations.

## License

NetworkOutbreaks.jl is licensed under the MIT License.
