module NetworkOutbreaks

using Graphs
using Random
using StatsBase: sample, Weights
using DataStructures

export
    # Model
    OutbreakModel,
    OutbreakTransition,
    # Spec & seeding
    OutbreakSpec,
    SeedFraction,
    SeedNodes,
    # Network types
    AbstractContactNetwork,
    StaticNetwork,
    TimeVaryingNetwork,
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
    # Observables
    compartment_series,
    times,
    events,
    state_at,
    reinfection_histogram,
    final_size,
    mean_curve,
    quantile_band

include("model.jl")
include("network.jl")
include("spec.jl")
include("state.jl")
include("events.jl")
include("algorithms/common.jl")
include("algorithms/direct.jl")
include("algorithms/next_reaction.jl")
include("algorithms/composition_rejection.jl")
include("algorithms/has.jl")
include("ensemble.jl")
include("analysis.jl")

end # module
