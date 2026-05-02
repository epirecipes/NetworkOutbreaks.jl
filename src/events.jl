#=
events.jl

Event records and trajectory container.

`OutbreakTrajectory` stores:
- a vector of event times,
- a parallel matrix of compartment counts (snapshots after each event),
- the full per-node infection-count vector at the end of the run,
- the seed and algorithm name for reproducibility metadata.

`compartment_series(traj, sym)` returns the integer count of compartment
`sym` over the saved time grid.

`(traj::OutbreakTrajectory)(t)` returns the compartment-count vector at
time `t` via piecewise-constant interpolation (binary search).
=#

struct OutbreakEvent
    time::Float64
    transition_index::Int  # index into model.transitions
    node::Int              # the node whose state changed
end

struct OutbreakTrajectory
    model::OutbreakModel
    times::Vector{Float64}
    counts::Matrix{Int}                # C × length(times)
    final_infection_counts::Vector{Int}
    events::Vector{OutbreakEvent}
    seed::UInt64
    algorithm::Symbol
end

times(t::OutbreakTrajectory) = t.times
events(t::OutbreakTrajectory) = t.events

function compartment_series(t::OutbreakTrajectory, sym::Symbol)
    haskey(t.model.index_of, sym) ||
        throw(ArgumentError("unknown compartment $(sym)"))
    return @view t.counts[t.model.index_of[sym], :]
end

function compartment_series(t::OutbreakTrajectory)
    return Dict{Symbol, Vector{Int}}(
        c => Vector(t.counts[i, :])
        for (i, c) in enumerate(t.model.compartments)
    )
end

function state_at(t::OutbreakTrajectory, query_t::Real)
    if query_t <= t.times[1]
        return Vector(t.counts[:, 1])
    elseif query_t >= t.times[end]
        return Vector(t.counts[:, end])
    end
    k = searchsortedlast(t.times, Float64(query_t))
    return Vector(t.counts[:, k])
end

(traj::OutbreakTrajectory)(t::Real) = state_at(traj, t)
