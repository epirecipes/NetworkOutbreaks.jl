# Migrating to NetworkOutbreaks 0.2

NetworkOutbreaks 0.2 is built on **NetworkEpiCore** (NEC), the core it shares with EdgeBasedModels and
NodeBasedModels. A model is a NEC `ContactModel`, a network is a NEC descriptor or an explicit graph, and seeding
is a NEC `SeedSpec`. NetworkOutbreaks re-exports the NEC bindings, so `using NetworkOutbreaks` is enough, and
`using NetworkEpiCore, EdgeBasedModels, NodeBasedModels, NetworkOutbreaks` loads without name clashes. The
low-level layer (`OutbreakModel`, `OutbreakSpec`, `simulate(spec)`, the four samplers, interventions, ensembles and
observables) is unchanged apart from the points below.

```julia
using NetworkOutbreaks

# the new high-level call: a fresh graph per run, the random streams of DESIGN §J.7
ens = simulate(seir_model(), ConfigurationNetwork(PoissonDegree(5)); N = 10_000,
               p = Dict(:τ => 1/6, :σ => 1/5, :γ => 1/4), initial = SeedFraction(:E => 0.01),
               tspan = (0.0, 150.0), nsims = 100, seed = 20260926)
final_size(ens)                                   # per run: the fraction ever infected

# the low level, as in 0.1
model = OutbreakModel(seir_model(), Dict(:τ => 1/6, :σ => 1/5, :γ => 1/4))
spec  = OutbreakSpec(model = model, network = random_regular_graph(1000, 5),
                     initial = SeedFraction(:E => 0.01), tspan = (0.0, 150.0))
traj  = simulate(spec; algorithm = NextReaction(), seed = 1)

ens = simulate(scenario(:sir_pois5))              # a shared validation scenario (NEC registry)
```

## Models

| 0.1 | 0.2 | Behaviour in 0.2 |
|---|---|---|
| `OutbreakModel(prog::EdgeBasedModels.DiseaseProgression, p)` (package extension) | `OutbreakModel(contact_model(prog), p)`; `OutbreakModel(prog, p)` still works when EdgeBasedModels is loaded | the extensions are **deleted**. `OutbreakModel(x, p)` converts anything `contact_model` accepts, and EdgeBasedModels and NodeBasedModels define `contact_model` for their 0.1 types. The result equals the 0.1 extension's output field by field when every transmission rate is positive (checked by EdgeBasedModels' test suite). A stage whose transmission rate evaluates to 0 differs: 0.1 dropped its infection and marked it non-infectious, 0.2 keeps the contact with rate 0 and the stage stays infectious (see below) |
| `OutbreakModel(m::NodeBasedModels.CompartmentalModel, p)` (package extension) | the same call, through `contact_model` | as above. The infectious compartments are now the non-susceptible infectors of the contacts, whatever their rates, rather than the 0.1 `infectious` flags, and each infection has its catalyst in `via` |
| `EdgeBasedModels.sir_model()` etc. with parameters `:β, :γ` | NEC `sir_model()` with parameters **`:τ`**, `:γ` (τ is the per-contact rate) | `OutbreakModel(sir_model(), Dict(:β => …))` is an error listing the missing `τ` |
| — | `OutbreakModel(cm::ContactModel, p; network)` | new. Contacts become `:infection` (or `:contact_trace`, see below) with `via = [infector]` and the per-contact rate of the model's rate convention (applied with the nominal `mean_degree(network)`); node transitions become `:spontaneous`; removals `X → ∅` go to an added absorbing compartment `:removed` |
| `OutbreakModel(compartments, infectious, transitions)` | the same, with new keywords `susceptible` and `strata` | the model now records its susceptible compartments (inferred when not given: the non-infectious sources of `:infection` transitions), which choose where unseeded nodes go, and the stratum of each compartment of a stratified model (`OutbreakModel(cm)` reads it from the species labels of `stratify`; empty otherwise) |

**`:contact_trace` is kept** (DESIGN §A.5 planned to deprecate it; §J.8 and the WP3 review require a tracing flag).
It is a contact with the same hazard as `:infection` that never counts as an infection. `OutbreakModel(cm)` uses it
for every contact that the NEC typing does not classify as an infection: a contact whose recipient is already
infected (quarantine of exposed contacts `E + I → Eq + I`, superinfection) or whose recipient or product is
susceptible (awareness `S + Sa → 2Sa` between susceptible classes). Tracing of susceptibles (`S + D → Q + D`, product
not infectious) is an `:infection` contact that the structural rule does not count, because Q never becomes
infectious. Declare aware or otherwise protected susceptible classes in the `ContactModel` (`susceptible = [:S, :Sa]`),
so that awareness is not read as an infection.

## Networks

| 0.1 | 0.2 | Behaviour in 0.2 |
|---|---|---|
| `MultiplexNetwork(layers, layer_rates)` (the graph container) | `MultiplexGraph(layers, layer_rates; names)` | **renamed**: `MultiplexNetwork` is now NEC's multiplex *descriptor* (`MultiplexNetwork(:home => RegularDegree(3), …)`), re-exported (`NetworkOutbreaks.MultiplexNetwork === NetworkEpiCore.MultiplexNetwork`). The 0.1 call `MultiplexNetwork(graphs, rates)` is a `MethodError` whose message gives the migration to `MultiplexGraph`. The layers now have names (default `:layer1, :layer2, …`) |
| every contact acted on every layer of a multiplex | `OutbreakTransition(…; layer = :home)`; a NEC `Contact(…; layer = :home)` keeps its layer in `OutbreakModel(cm)` | new: a contact on a named layer counts only the catalyst neighbours linked in the `MultiplexGraph` layer of that name (`sample_graph(::MultiplexNetwork, N)` names the layers after the descriptor). `layer = :all` (the default) is the 0.1 behaviour. A named layer on a network without it is an `ArgumentError` (an `AdmissibilityError` from `simulate(model, desc; …)`) |
| `TimeVaryingNetwork`: an `:add` of a present edge or a `:remove` of an absent one was a silent no-op | the same type | **error** when the `OutbreakSpec` is built: a simple graph has no edge multiplicities, so overlapping contacts on the same pair must be merged into one interval. Updates before `tspan[1]` are the network's past (skipped, not checked); node ranges, self-loops and actions are checked there too |
| — | `sample_graph(desc, N; rng) -> (graph, info::GraphInfo)` | new: one graph from a descriptor: configuration networks, `ExplicitGraph`, `MultitypeNetwork` (a `TypedGraph`), `ClusteredNetwork`, `MultiplexNetwork` (a named `MultiplexGraph`), `WellMixed` (the complete graph), `DegreeCorrelatedNetwork` (joint degrees), `DynamicNetwork` (the initial graph) and `MFSHNetwork` (`FleetingContacts`, stub counts); other descriptors are an `ArgumentError` |
| — | `DynamicGraph(g, process)`, `NeighbourExchangeProcess(η)`, `DormantContactProcess`, `evolve_graph!` | new: graphs rewired by a graph process during a run (NextReaction and HAS); `simulate(model, DynamicNetwork(…); N)` builds them |
| multiplex networks ran in DirectSSA only | the same `MultiplexGraph` | NextReaction and HAS accept it too (CompositionRejection still refuses it) |
| — | `SampledNetwork(desc, N, graphs)` | the network recorded in the spec of an ensemble whose runs drew fresh graphs; `simulate(spec)` refuses it |
| directed graphs were accepted (wrong hazards) | — | **error** (since the 0.1 fixes of WP3): contact graphs must be undirected and free of self-loops |

## Samplers and scenarios (new)

- `MassActionSSA()`: the count-level Gillespie sampler of a well-mixed population; `simulate(model, WellMixed(κ); N)`
  uses it (its contacts are the complete graph with rates κτ/(N − 1), lumped).
- `FleetingContactSSA()` on `FleetingContacts`: mean-field social heterogeneity (`MFSHNetwork`), every contact with a
  fresh partner.
- `simulate(sc::Scenario)` runs a NEC scenario with its settings; `scenario_ensemble`, `summarise`,
  `scenario_summary` and `regenerate_scenarios` build and cache the reference summaries of the shared scenarios
  (DESIGN §E).

## Seeding

`SeedSpec`, `SeedFraction`, `SeedNodes` and the new `SeedCount` are NEC's types (`NetworkOutbreaks.SeedFraction ===
NetworkEpiCore.SeedFraction`), with the 0.1 constructors (`SeedFraction(:I => 0.01)`, `SeedNodes(:I => [1, 2];
default = :S)`) and a new `default` keyword on all three.

- **Counts** (verified issue N01). n_X = ρ_X N is rounded to the nearest integer with **ties away from zero** (0.1:
  ties to even, per compartment, with the leftover pushed into another compartment). A positive ρ_X that rounds to 0
  nodes is an `ArgumentError` (0.1 silently seeded nothing: `SeedFraction(:I => 0.001)` on 499 nodes), as are counts
  that exceed N. Fractions outside [0, 1], NaN, duplicate compartments and sums above 1 are rejected when the
  `SeedFraction` is built (0.1 could leave nodes unwritten or write out of bounds).
- **Background compartment.** Unseeded nodes go to the specification's `default`, else to the model's first
  susceptible compartment that the specification does not name. When the specification names every susceptible
  compartment, that compartment absorbs the rounding if the specification covers the whole population
  (`SeedFraction(:S => 0.9, :I => 0.1)` never leaks a node into R or E, as 0.1 did); otherwise the rest goes to the
  first compartment that is neither susceptible, infected nor named (R), and an error says so if there is none.
- **Placement.** Seeded nodes are chosen uniformly without replacement, disjointly, in the order of the
  specification. The node sets differ from 0.1 (which iterated a `Dict`), so seeded runs do not reproduce 0.1 sample
  paths; the distributions are unchanged.
- **Multitype seeding** (DESIGN §J.6): `SeedFraction(:I_a => ρ)` is a fraction of all N nodes, placed on type a.
  That needs a network that records node types. On a graph without them the seeding assigns the strata, so a
  seeding that leaves the background to the rule and would put the unseeded nodes of several strata into one (for
  `stratify(sir_model(), [:a, :b])`, `SeedFraction(:I_a => 0.05)` would put all of them into `S_a`) is an
  `ArgumentError`: name the susceptible class of every stratum, use `SeedNodes`, or pass `default`.

## Random numbers and trajectories

- `seed::Integer` now builds `NetworkOutbreaks.stable_rng(seed)`: a `StableRNG` seeded with the splitmix64 mix of
  `seed + golden` (0.1: a Xoshiro stream). Adjacent raw `StableRNG` seeds are strongly correlated (first-draw
  correlation −0.43); `stable_rng(s)` and `stable_rng(s + 1)` are not. `simulate(spec; rng =
  NetworkOutbreaks.stable_rng(s))` reproduces `simulate(spec; seed = s)`. `ALGORITHM_REVISION` is `"2"`.
- `simulate(model, net; seed = b)` draws graph r with `stable_rng(b + r)` and runs run r with seed `b + 2^32 + r`
  (DESIGN §J.7), so each run and its graph can be regenerated alone.
- `traj.seed` is a `Union{Nothing, UInt64}`: it is `nothing` for a run driven by an explicit `rng` (0.1 used 0 for
  them as well as for `seed = 0`).
- `state_at(traj, t)` is right-continuous at every snapshot time: several snapshots at one time (an intervention
  applied at `t0`) resolve to the last one, as in `mean_curve` and `quantile_band` (0.1 returned the state before an
  intervention at `t0`).

## Observables

- `final_size(traj)` is the fraction of nodes **ever infected, including the seeds** (not "ever left S"). Infected
  compartments are defined structurally (see the `_infected_mask` docstring): latent seeds count, tracing,
  quarantine and awareness preserve infection status, and vaccination and tracing of susceptibles are not
  infections. `final_size(traj; recovered = X)` returns the fraction in `X` at the end, and an unknown name is an
  error.
- Infection status does not depend on the rate values (DESIGN §A.5, §J.8). `OutbreakModel(cm, p)` marks every
  non-susceptible infector of a contact infectious, even when the rate of that contact is 0, and susceptible
  catalysts (awareness) are never infectious. So `final_size` of an SIR run with `τ = 0` and `SeedFraction(:I => ρ)`
  is ρ (the seeds), as for any τ > 0, and the unseeded remainder of `SeedFraction(:S => 0.5, :I => 0.1)` in SEIR goes
  to R, not to E.
- `final_size`, `reinfection_histogram`, `compartment` and `compartments` are NEC generics with NetworkOutbreaks
  methods; `compartment(traj, X)` is the fraction of nodes in `X` at each snapshot.

## Removed

- `ext/NetworkOutbreaksEdgeBasedModelsExt.jl` and `ext/NetworkOutbreaksNodeBasedModelsExt.jl`, with the
  `[weakdeps]`/`[extensions]` entries: NetworkOutbreaks no longer depends on EdgeBasedModels, NodeBasedModels or
  ModelingToolkit, not even in its tests.
- `test/test_eon_patterns.jl` (superseded by `test/suites/00_legacy_eon.jl`).
