# NetworkOutbreaks.jl

NetworkOutbreaks runs exact stochastic simulations of epidemics on contact networks. It samples
continuous-time Markov chains on explicit graphs, on graphs sampled from a network descriptor,
on dynamic graphs and on well-mixed populations. It also produces the reference ensembles
against which EdgeBasedModels and NodeBasedModels are validated.

Version 0.2 is built on NetworkEpiCore (NEC), whose names it re-exports:

- a model is a NEC `ContactModel`;
- a network is a NEC `NetworkDescriptor` or a `Graphs.AbstractGraph`;
- seeding is a NEC `SeedSpec`.

The verb is [`simulate`](@ref). The 0.1 package extensions for EdgeBasedModels and
NodeBasedModels are gone: `OutbreakModel(x, p)` accepts anything that `contact_model` accepts.
The changes from 0.1 are listed in [Migrating from 0.1](migration.md).

## High level: model + descriptor

```julia
using NetworkOutbreaks

ens = simulate(sir_model(), ConfigurationNetwork(PoissonDegree(5)); N = 1000,
               p = Dict(:τ => 1/6, :γ => 1/4), initial = SeedFraction(:I => 0.01),
               tspan = (0.0, 60.0), nsims = 20, seed = 1)
final_size(ens)                     # one value per run: the fraction ever infected
```

Each run draws a fresh graph. Graph r uses the random stream `stable_rng(seed + r)` and SSA
run r uses `stable_rng(seed + 2³² + r)`.

## Low level: `OutbreakModel` + `OutbreakSpec`

```julia
model = OutbreakModel(seir_model(), Dict(:τ => 1/6, :σ => 1/5, :γ => 1/4))
g, info = sample_graph(ConfigurationNetwork(PoissonDegree(5)), 1000; rng = NetworkOutbreaks.stable_rng(7))
spec = OutbreakSpec(model = model, network = g, initial = SeedFraction(:E => 0.01), tspan = (0.0, 150.0))
traj = simulate(spec; algorithm = NextReaction(), seed = 1)
ens  = simulate_ensemble(spec; nsims = 40, seed = 2)
```

## Samplers

| sampler | method | multiplex | `TimeVaryingNetwork` | `DynamicGraph` | interventions |
|---|---|:---:|:---:|:---:|:---:|
| `DirectSSA()` | Gillespie direct method | ✓ | ✓ | ✗ | ✓ |
| `NextReaction()` | Gibson–Bruck | ✓ | ✓ | ✓ | ✓ |
| `CompositionRejection()` | Slepoy et al. 2008 | ✗ | ✗ | ✗ | ✗ |
| `HAS()` | hierarchical adaptive sampling | ✓ | ✓ | ✓ | ✓ |
| `MassActionSSA()` | count-level SSA of `WellMixed(κ)` | – | – | – | ✓ |
| `FleetingContactSSA()` | fleeting contacts (`MFSHNetwork`) | – | – | – | ✓ |

Each ✗ means the sampler raises an `ArgumentError` that names the alternatives, and a dash
means the combination does not apply (the sampler uses its own network type).

## Scenarios

`simulate(scenario(:sir_pois5))` runs a shared NEC scenario with its own settings.
`scenario_summary(sc)` loads the committed summary of its ensemble. See
[Reference ensembles](validation.md).

## Companion packages

- [NetworkEpiCore.jl](https://github.com/epirecipes/NetworkEpiCore.jl): the shared model,
  network, scenario and morphism objects.
- [EdgeBasedModels.jl](https://epirecip.es/EdgeBasedModels.jl/): edge-based compartmental
  models.
- [NodeBasedModels.jl](https://epirecip.es/NodeBasedModels.jl/): pairwise and other node-level
  closures.
