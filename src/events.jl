#=
events.jl (owner: WP16)

Event records and the trajectory container. A trajectory stores snapshots of the compartment counts (after each
event, or on a time grid), the per-node infection counts at the end of the run, and the seed and algorithm for
reproducibility. Every observable reads the snapshots right-continuously: the value at time t is the state after
every snapshot at or before t, so an intervention applied at t0 is visible at t0 (as in `mean_curve`).
=#

"""
    OutbreakEvent(time, transition_index, node)

One event of a trajectory recorded with `keep = :events`: at `time`, `node` changed state through
`model.transitions[transition_index]`. Moves made by interventions are not logged as events.
"""
struct OutbreakEvent
    time::Float64
    transition_index::Int  # index into model.transitions
    node::Int              # the node whose state changed
end

"""
    OutbreakTrajectory

One simulated run:

- `model`: the `OutbreakModel` that was simulated;
- `times` and `counts`: snapshot times and the C × length(times) matrix of compartment counts after each
  snapshot (after every event, or on a time grid with `keep = :grid`);
- `final_infection_counts`: per node, how many times it entered an infected compartment (seeds count once), at the
  end of the time span; see `final_size` and `reinfection_histogram`;
- `events`: the event log (empty unless `keep = :events`);
- `seed`: a `Union{Nothing, UInt64}`. For a run seeded by a number (`seed = s`, or no `seed` and no `rng`, when a
  seed is drawn) it is that seed, and `simulate(spec; seed = traj.seed)` reproduces the run. For a run driven by an
  explicit `rng` it is `nothing`;
- `algorithm`: the name of the sampler.

`traj(t)` is `state_at(traj, t)`.
"""
struct OutbreakTrajectory
    model::OutbreakModel
    times::Vector{Float64}
    counts::Matrix{Int}                # C × length(times)
    final_infection_counts::Vector{Int}
    events::Vector{OutbreakEvent}
    seed::Union{Nothing, UInt64}
    algorithm::Symbol
end

"""
    times(traj::OutbreakTrajectory) -> Vector{Float64}

The snapshot times of a trajectory.
"""
times(t::OutbreakTrajectory) = t.times

"""
    events(traj::OutbreakTrajectory) -> Vector{OutbreakEvent}

The event log of a trajectory (empty unless it was simulated with `keep = :events`).
"""
events(t::OutbreakTrajectory) = t.events

"""
    compartment_series(traj, X::Symbol) -> AbstractVector{Int}
    compartment_series(traj) -> Dict{Symbol,Vector{Int}}

The count of compartment `X` at each snapshot time `times(traj)` (a view), or every compartment's series.
"""
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

"""
    state_at(traj, t) -> Vector{Int}

The compartment counts at time `t`: the state after every snapshot at or before `t` (right-continuous, so several
snapshots at one time, such as an intervention applied at the start, resolve to the last one). Before the first
snapshot it is the first snapshot. The same convention is used by `mean_curve` and `quantile_band`.
"""
function state_at(t::OutbreakTrajectory, query_t::Real)
    isnan(query_t) && throw(ArgumentError("state_at: the time is NaN"))
    k = query_t < t.times[1] ? 1 : searchsortedlast(t.times, Float64(query_t))
    return Vector(t.counts[:, k])
end

(traj::OutbreakTrajectory)(t::Real) = state_at(traj, t)
