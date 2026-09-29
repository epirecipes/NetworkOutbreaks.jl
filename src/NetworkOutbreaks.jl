"""
    NetworkOutbreaks

Exact stochastic simulation of epidemics on contact networks: continuous-time Markov chains
sampled with `DirectSSA`, `NextReaction`, `CompositionRejection` or `HAS`.

NetworkOutbreaks builds on NetworkEpiCore, whose bindings it re-exports: a model is a
`ContactModel` (contacts `S + I → 2I` at a per-contact rate τ and node transitions; `sir_model()`
and friends, Catalyst and ModelingToolkit models through `contact_model`), a network is a
`NetworkDescriptor` (`ConfigurationNetwork(PoissonDegree(5))`, `ExplicitGraph(g)`, …) or an
explicit `Graphs.AbstractGraph`, and seeding is a `SeedSpec` (`SeedFraction`, `SeedCount`,
`SeedNodes`). The verb is [`simulate`](@ref):

```julia
using NetworkOutbreaks
ens = simulate(sir_model(), ConfigurationNetwork(PoissonDegree(5)); N = 1000,
               p = Dict(:τ => 1/6, :γ => 1/4), initial = SeedFraction(:I => 0.01),
               tspan = (0.0, 60.0), nsims = 20, seed = 1)
final_size(ens)                      # one value per run: the fraction ever infected
```

The low-level layer is an [`OutbreakModel`](@ref) (compartments and numeric transitions; built
from a `ContactModel` by `OutbreakModel(cm, p)`) with an [`OutbreakSpec`](@ref) (model, network,
seeding, time span), run by `simulate(spec; algorithm, seed)`. MIGRATION.md lists the changes
from NetworkOutbreaks 0.1.
"""
module NetworkOutbreaks

using Graphs
using Random
using StatsBase: sample, Weights
import DataStructures
using DataStructures: MutableBinaryMinHeap, update!

# NetworkEpiCore: NetworkOutbreaks adds methods to its generics and re-exports the bindings it
# exports (the same bindings, so NetworkEpiCore, EdgeBasedModels, NodeBasedModels and
# NetworkOutbreaks load together without ambiguity; design §A.7, §A.8). The generics below are
# imported explicitly so that any file of this package can add methods to them.
using NetworkEpiCore
import NetworkEpiCore: final_size, reinfection_histogram, epidemic_probability, compartment,
                       compartments, population_fraction, mean_degree, contact_model,
                       canonical_text

# Re-export every NetworkEpiCore export, including the multiplex descriptor `MultiplexNetwork` (the graph container
# that NetworkOutbreaks 0.1 called `MultiplexNetwork` is `MultiplexGraph`; the 0.1 call gets a migration hint, see
# `__init__`). (`names` also lists `public` names from Julia 1.11 on; only the exports are re-exported.)
for name in names(NetworkEpiCore)
    name === :NetworkEpiCore && continue
    (!isdefined(Base, :isexported) || Base.isexported(NetworkEpiCore, name)) && @eval export $name
end

# NetworkOutbreaks' own names. Files of later work packages carry their own `export` lines
# (design §G.1).
export
    # Model
    OutbreakModel,
    OutbreakTransition,
    # Spec (the seeding types SeedSpec, SeedFraction, SeedCount and SeedNodes are
    # NetworkEpiCore's, re-exported above)
    OutbreakSpec,
    # Network types
    AbstractContactNetwork,
    StaticNetwork,
    TimeVaryingNetwork,
    MultiplexGraph,
    SampledNetwork,
    # State / events / trajectory
    OutbreakState,
    OutbreakEvent,
    OutbreakTrajectory,
    OutbreakEnsemble,
    # Algorithms
    OutbreakAlgorithm,
    DirectSSA,
    NextReaction,
    CompositionRejection,
    HAS,
    # Top-level
    simulate,
    simulate_ensemble,
    sample_graph,
    # Observables
    compartment_series,
    times,
    events,
    state_at,
    mean_curve,
    quantile_band,
    # Interventions
    AbstractIntervention,
    ScheduledRateChange,
    ScheduledStateChange,
    ThresholdIntervention,
    InterventionPlan

# Include order: a file may use the types of every file included before it in signatures and
# struct fields; function bodies may call functions of any file. This file and the include list
# are owned by WP16; later work packages never edit them (design §G.1).

# The model, networks, interventions, seeding and state (WP16; interventions.jl: WP3)
include("model.jl")
include("network.jl")
include("interventions.jl")
include("spec.jl")
include("state.jl")
include("events.jl")

# The shared sampler machinery (WP3)
include("algorithms/common.jl")

# Network processes (WP26; stretch: WP36b dormant contacts, WP36c MFSH). Before the samplers,
# so that their types can appear in the samplers' signatures.
include("processes/common.jl")
include("processes/neighbour_exchange.jl")
include("processes/dormant.jl")
include("processes/mfsh.jl")

# The samplers (WP3; next_reaction.jl, has.jl and mass_action.jl: WP26)
include("algorithms/direct.jl")
include("algorithms/next_reaction.jl")
include("algorithms/composition_rejection.jl")
include("algorithms/has.jl")
include("algorithms/mass_action.jl")

# Ensembles and observables (WP3)
include("ensemble.jl")
include("analysis.jl")

# simulate(model, net; N, …), simulate(model, g), simulate(sc::Scenario), the sample_graph
# generic and its fallback (WP16)
include("convenience.jl")

# Graph generators: sample_graph methods per descriptor (WP25; stretch: WP36a joint degrees)
include("generators/common.jl")
include("generators/configuration.jl")
include("generators/wellmixed.jl")
include("generators/multitype.jl")
include("generators/clustered.jl")
include("generators/multiplex.jl")
include("generators/joint_degree.jl")

# The scenario runner: conditioning, alignment, summaries, cache, regeneration (WP27)
include("validation/conditioning.jl")
include("validation/runner.jl")
include("validation/summarise.jl")
include("validation/cache.jl")

function __init__()
    # The NetworkOutbreaks 0.1 call MultiplexNetwork(graphs, rates) now reaches NetworkEpiCore's descriptor and fails
    # with a MethodError; this adds the migration to MultiplexGraph to its message (src/network.jl).
    Base.Experimental.register_error_hint(_multiplex_migration_hint, MethodError)
    return nothing
end

end # module
