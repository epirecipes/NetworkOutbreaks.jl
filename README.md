# NetworkOutbreaks.jl

Exact stochastic simulation of epidemics on contact networks. The package samples
continuous-time Markov chains on explicit graphs, on graphs sampled from a network descriptor,
on dynamic graphs and on well-mixed populations. It also produces the **reference ensembles**
against which the deterministic packages are validated.

Version 0.2 is built on [NetworkEpiCore.jl](../NetworkEpiCore.jl) (NEC), whose names it
re-exports:

- a model is a NEC `ContactModel`, with contacts `S + J → X + J` at a per-contact rate τ;
- a network is a NEC `NetworkDescriptor` or a `Graphs.AbstractGraph`;
- seeding is a NEC `SeedSpec`.

The verb is `simulate`. The same model and network go to
[EdgeBasedModels.jl](../EdgeBasedModels.jl) (`edge_based`) and
[NodeBasedModels.jl](../NodeBasedModels.jl) (`node_based`). The three packages load together
without name clashes. The 0.1 package extensions for EdgeBasedModels and NodeBasedModels are
gone: `OutbreakModel(x, p)` accepts anything that `contact_model` accepts. Changes from 0.1 are
listed in [MIGRATION.md](MIGRATION.md).

## Quick start

### High level: model + descriptor

This level draws a fresh graph for every run, and uses the random streams of the validation
protocol:

```julia
using NetworkOutbreaks

ens = simulate(sir_model(), ConfigurationNetwork(PoissonDegree(5)); N = 1000,
               p = Dict(:τ => 1/6, :γ => 1/4), initial = SeedFraction(:I => 0.01),
               tspan = (0.0, 60.0), nsims = 20, seed = 1)
final_size(ens)                     # one value per run: the fraction ever infected
```

Any front end gives the same model. For example, `contact_model(rn)` for a Catalyst
`ReactionSystem` (`using Catalyst`) or `contact_model(sys)` for a ModelingToolkit system can
replace `sir_model()`.

### Low level: `OutbreakModel` + `OutbreakSpec`

```julia
model = OutbreakModel(seir_model(), Dict(:τ => 1/6, :σ => 1/5, :γ => 1/4))   # numeric transitions
g, info = sample_graph(ConfigurationNetwork(PoissonDegree(5)), 1000; rng = NetworkOutbreaks.stable_rng(7))
spec = OutbreakSpec(model = model, network = g, initial = SeedFraction(:E => 0.01), tspan = (0.0, 150.0))
traj = simulate(spec; algorithm = NextReaction(), seed = 1)
final_size(traj)
ens  = simulate_ensemble(spec; nsims = 40, seed = 2)          # one graph, many runs
```

`OutbreakModel(compartments, infectious, transitions)` with `OutbreakTransition`s is still
available for hand-written models. Removals `X → ∅` go to an added absorbing compartment
`:removed`. `final_size` is the fraction **ever infected**. Infection status is decided from the
structure of the model, so tracing, quarantine and vaccination are not counted as infections.

### A shared scenario

```julia
sc  = scenario(:sir_pois5)            # NEC registry: model, network, parameters, seeding, time grid, SimConfig
ref = scenario_summary(sc)            # the committed summary of its ensemble (data/scenarios)
ens = simulate(sc)                    # or re-run it
```

## Samplers

| sampler | method | multiplex | `TimeVaryingNetwork` | `DynamicGraph` | interventions |
|---|---|:---:|:---:|:---:|:---:|
| `DirectSSA()` | Gillespie direct method, O(N) per event | ✓ | ✓ | ✗ | ✓ |
| `NextReaction()` | Gibson–Bruck, O(log N) per event | ✓ | ✓ | ✓ | ✓ |
| `CompositionRejection()` | Slepoy et al. 2008, O(1) amortised | ✗ | ✗ | ✗ | ✗ |
| `HAS()` | hierarchical adaptive sampling (sum tree) | ✓ | ✓ | ✓ | ✓ |
| `MassActionSSA()` | count-level SSA of a well-mixed population (`WellMixed(κ)`) | – | – | – | ✓ |
| `FleetingContactSSA()` | fleeting contacts (`MFSHNetwork`) | – | – | – | ✓ |

Each ✗ means the sampler raises an `ArgumentError` that names the alternatives, and a dash
means the combination does not apply (the sampler uses its own network type). The samplers are exact for the same
CTMC, so their results must agree. On `:sir_pois5_N1000` the mean final sizes of DirectSSA,
CompositionRejection and HAS lie within |z| < 1 of the committed NextReaction ensemble
(vignette 05, §5).

## Networks and processes

- `sample_graph(desc, N; rng)` returns `(graph, info)` for these descriptors:
  - configuration networks. `PoissonDegree` is sampled as G(N, p), `RegularDegree` as a
    random regular graph, and other degree distributions as erased configuration graphs;
  - `ExplicitGraph`;
  - `MultitypeNetwork`, as a `TypedGraph`;
  - `ClusteredNetwork`;
  - `MultiplexNetwork`, as a named `MultiplexGraph`;
  - `DegreeCorrelatedNetwork`, from joint degrees;
  - `WellMixed`;
  - `DynamicNetwork`, which gives the initial graph;
  - `MFSHNetwork`, as `FleetingContacts`.
- `DynamicGraph(g, process)` rewires the graph during a run, with `NeighbourExchangeProcess(η)`
  or `DormantContactProcess`. `simulate(model, DynamicNetwork(…); N)` builds it.
- `TimeVaryingNetwork(g, updates)` applies scheduled edge additions and removals. Adding an
  edge that is already present, or removing one that is absent, is an error.
- Contacts can be restricted to a layer: `Contact(…; layer = :home)` counts only neighbours in
  that layer of a `MultiplexGraph`.
- Interventions are `ScheduledRateChange`, `ScheduledStateChange` (for example vaccination
  pulses with `from = [:S]`) and `ThresholdIntervention`, combined in an `InterventionPlan`.

## Validation: the reference ensembles

`data/scenarios/` holds **53 committed summaries**. There is one for every registered NEC
scenario and its N-scaling variants, plus the unaligned companions of the small-seed
scenarios. `missing_scenario_summaries()` is empty. Each summary follows the protocol of design
§E:

- **Size.** N = 10⁴ nodes and 200 runs, with a **fresh graph per run**. The neighbour-exchange
  scenarios use N = 5000 and 100 runs.
- **Streams.** `stable_rng(base_seed + r)` for graph r and `stable_rng(base_seed + 2³² + r)`
  for run r. These are StableRNGs seeded through splitmix64, because adjacent raw seeds are
  correlated.
- **Seeding.** Exactly ρN seeds, placed uniformly.
- **Conditioning.** SIR runs must be major outbreaks (`MajorOutbreak(0.05)`), and SIS runs must
  survive. Both the conditioned and unconditioned statistics are stored, together with
  P(major) and its 95% Wilson interval.
- **Keys.** Summaries are keyed by `scenario_hash(sc)` and `ALGORITHM_REVISION` (currently
  "2"). With `NETEPI_STRICT_CACHE=1`, a missing or stale summary is an error.

`regenerate_scenarios` (or `scripts/regenerate_scenarios.jl`) rebuilds them.

The table shows values read from the committed summaries. "Mean final size" is taken over the
major runs, with its standard error. "Expected R∞" is the NEC value stored in the scenario:
the edge-based fixed point, which is the large-N limit.

| scenario | network | N | runs | P(major) (95% CI) | mean final size ± SE | expected R∞ |
|---|---|---:|---:|---|---|---:|
| `:sir_reg6` | 6-regular | 10⁴ | 200 | 1.000 (0.981, 1.000) | 0.9292 ± 0.0003 | 0.9295 |
| `:sir_pois5` | Poisson(5) | 10⁴ | 200 | 1.000 (0.981, 1.000) | 0.8001 ± 0.0005 | 0.8002 |
| `:sir_nb4` | negative binomial | 10⁴ | 200 | 1.000 (0.981, 1.000) | 0.6405 ± 0.0007 | 0.6408 |
| `:sir_bim` | bimodal | 10⁴ | 200 | 1.000 (0.981, 1.000) | 0.4959 ± 0.0009 | 0.4956 |
| `:sir_pl` | power law | 10⁴ | 200 | 1.000 (0.981, 1.000) | 0.2881 ± 0.0012 | 0.2898 |
| `:seir_pois5` | Poisson(5) | 10⁴ | 200 | 1.000 (0.981, 1.000) | 0.8002 ± 0.0005 | 0.8002 |
| `:sir_wm5` | well mixed (`MassActionSSA`) | 10⁴ | 200 | 1.000 (0.981, 1.000) | 0.8001 ± 0.0006 | 0.8002 |
| `:sir_pois5_1seed` | Poisson(5), one seed | 10⁴ | 2000 | 0.5955 (0.574, 0.617) | 0.7966 ± 0.0002 | 0.7968 |

With a single seed, P(major) is the **infector-side** branching-process probability. It is
0.6094 for this scenario, which is not R∞ = 0.7968. The ensemble value 0.5955 lies below it at
N = 10⁴ (vignette 05, §2).

**N-scaling.** `:sir_pois5`, `:sir_bim`, `:sir_pl` and `:sir_clust_s2t2` are also committed at
N = 10³ (2000 runs) and N = 10⁵ (20 runs). An exact limit shows a D∞ that falls towards the
Monte Carlo floor, while a biased approximation levels off. For example, the edge-based model on
`:sir_pl` has D∞(I) = 0.0118, 0.0027 and 0.0014, while the constant-closure pairwise model has
0.0311, 0.0217 and 0.0200 (vignette 05, §4).

## Vignettes

The pages are in [`vignettes/`](vignettes/) and are rendered with a frozen cache. See
`vignettes/README.md` for which scenarios each page uses.

| page | topic |
|---|---|
| [01](vignettes/01_sis_reinfection/index.md) | SIS on a 3-regular network: pairwise and reinfection counting against the ensemble |
| [02](vignettes/02_algorithms_dynamic_networks/index.md) | samplers and dynamic networks: neighbour exchange |
| [03](vignettes/03_interventions/index.md) | continuous vaccination and vaccination pulses |
| [04](vignettes/04_contact_tracing/index.md) | neighbour-triggered quarantine |
| [05](vignettes/05_validation/index.md) | the validation protocol: scenarios, conditioning, alignment, N-scaling, sampler agreement |
| [06](vignettes/06_processes/index.md) | contact processes: dormant contacts, fleeting contacts, mass action |

## Lean proofs

`proofs/` (`NetworkOutbreaksProofs/EventSemantics.lean`) is a small standalone Lake project about
event semantics. It has no `sorry` and no `axiom`. It is not part of the NetworkEpiCore trusted
library, the SA-PASS alignment audit has not been run on it, and nothing cites it. The simulator
is validated by the tests and the ensembles above, not by proofs. The citable Lean library is
`NetworkEpiCore.jl/proofs`.

## Installation

```julia
using Pkg
Pkg.develop([PackageSpec(path = "NetworkEpiCore.jl"), PackageSpec(path = "NetworkOutbreaks.jl")])
```

## License

MIT
