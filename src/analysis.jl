#=
analysis.jl

Common observables computed from one or many trajectories.

Each trajectory has its own irregular event time grid. To average across runs we resample onto a common grid via
piecewise-constant (right-continuous) interpolation.
=#

"""
    mean_curve(ens, sym; tgrid = nothing) -> (t, μ)

Return `(t, μ)` where `t` is a common time grid and `μ[k]` is the mean count in compartment `sym` at time `t[k]`
across **all** trajectories of the ensemble. If `tgrid` is `nothing`, a 200-point grid over the spec's `tspan` is
used.

!!! warning "Unconditioned finite-N mean"
    This is the unconditional mean over every run, including runs that went extinct early, on the absolute time
    axis. With few initial seeds it is not comparable to a deterministic (edge-based, pairwise or mass-action) limit,
    which assumes no extinction and no random delay: condition on major outbreaks and align runs in time first
    (the scenario runner does this), or seed enough nodes (ρN ≳ 10–20) that extinction is negligible.
"""
function mean_curve(ens::OutbreakEnsemble, sym::Symbol;
                    tgrid::Union{Nothing, AbstractVector{<:Real}} = nothing)
    t = _time_grid(ens, tgrid)
    μ = zeros(Float64, length(t))
    for traj in ens.trajectories
        c = compartment_series(traj, sym)
        for (k, tk) in pairs(t)
            μ[k] += _interp(traj.times, c, tk)
        end
    end
    μ ./= length(ens.trajectories)
    return t, μ
end

"""
    quantile_band(ens, sym; q = (0.025, 0.975), tgrid = nothing) -> (t, lo, hi)

Return `(t, lo, hi)` where `lo[k]` and `hi[k]` are the `q[1]` and `q[2]` quantiles of the count in compartment `sym`
at time `t[k]` across all trajectories.

!!! warning "Unconditioned band"
    Extinct runs are included, so with few seeds the lower edge collapses to 0 during the epidemic; see
    [`mean_curve`](@ref).
"""
function quantile_band(ens::OutbreakEnsemble, sym::Symbol;
                       q::Tuple{<:Real, <:Real} = (0.025, 0.975),
                       tgrid::Union{Nothing, AbstractVector{<:Real}} = nothing)
    t = _time_grid(ens, tgrid)
    n = length(ens.trajectories)
    samples = Matrix{Float64}(undef, n, length(t))
    for (i, traj) in pairs(ens.trajectories)
        c = compartment_series(traj, sym)
        for (k, tk) in pairs(t)
            samples[i, k] = _interp(traj.times, c, tk)
        end
    end
    lo = [quantile(view(samples, :, k), q[1]) for k in 1:length(t)]
    hi = [quantile(view(samples, :, k), q[2]) for k in 1:length(t)]
    return t, lo, hi
end

"""
    final_size(traj::OutbreakTrajectory; recovered = nothing) -> Float64
    final_size(ens::OutbreakEnsemble; recovered = nothing) -> Vector{Float64}

The fraction of nodes that were ever infected during the run, **including the seeds** (the observable
`:cumulative` at the end of the run, on a per-node basis).

A node counts if it started in an *infected* compartment, or entered one at any time by any event (a contact, a
node-local transition or an intervention such as an importation). The infected compartments are the infectious
ones plus those on a path from an infection (an `:infection` contact with an infectious catalyst whose product
leads to infectiousness) to infectiousness that does not pass through a susceptible compartment. So latent (E) seeds
count from `t = 0`, also in models where latent or infectious nodes can themselves be traced or quarantined
(`E → Q`, `E → Eq → Iq`, `I → Q`), while recovered, vaccinated (`S → V`), traced (`S → Q` by `:contact_trace`) and
aware (`S → Sa` via `Sa`) nodes do not count unless they were infected. Each node counts once, however often it was
reinfected (see [`reinfection_histogram`](@ref)); moving between infected compartments (`E → I`, `E → Eq`,
superinfection `I1 → I2`) is not a new infection.

With `recovered` (a compartment name or a collection of names), return instead the fraction of nodes in those
compartments at the end of the run; an unknown name is an `ArgumentError`.

The ensemble method returns one value per trajectory.
"""
function final_size(traj::OutbreakTrajectory; recovered = nothing)
    n = length(traj.final_infection_counts)
    recovered === nothing && return count(>(0), traj.final_infection_counts) / n
    syms = recovered isa Symbol ? (recovered,) : Tuple(recovered)
    isempty(syms) && throw(ArgumentError("`recovered` must name at least one compartment"))
    total = 0
    for s in syms
        s isa Symbol || throw(ArgumentError("`recovered` must be a Symbol or a collection of Symbols; got $(s)"))
        haskey(traj.model.index_of, s) ||
            throw(ArgumentError("unknown compartment $(s) in `recovered`; the model has $(traj.model.compartments)"))
        total += traj.counts[traj.model.index_of[s], end]
    end
    return total / n
end

final_size(ens::OutbreakEnsemble; recovered = nothing) =
    Float64[final_size(traj; recovered = recovered) for traj in ens.trajectories]

"""
    reinfection_histogram(traj; L = nothing) -> Vector{Int}

Histogram of per-node infection counts at the end of `traj`: index `p + 1` holds the number of nodes infected `p`
times. A node's count is incremented each time it enters an infected compartment from a non-infected one, and a
node that starts infected (a seed, latent or infectious) counts once. The vector has length `L + 1` (or
`maximum + 1` if `L` is `nothing`); counts above `L` are saturated into the top bucket, matching the convention of
`with_reinfection_counting`.
"""
function reinfection_histogram(traj::OutbreakTrajectory;
                               L::Union{Integer, Nothing} = nothing)
    counts = traj.final_infection_counts
    Lmax = isnothing(L) ? maximum(counts; init = 0) : Int(L)
    Lmax >= 0 || throw(ArgumentError("L must be non-negative; got $(L)"))
    h = zeros(Int, Lmax + 1)
    @inbounds for c in counts
        idx = min(c, Lmax)
        h[idx + 1] += 1
    end
    return h
end

# --- internals ---

_time_grid(ens::OutbreakEnsemble, tgrid) =
    isnothing(tgrid) ? collect(range(ens.spec.tspan[1], ens.spec.tspan[2]; length = 200)) :
                       collect(Float64.(tgrid))

# Number of nodes that start the run in an infected compartment (the seeds counted by `final_size`).
function _initially_infected(traj::OutbreakTrajectory)
    infected = _infected_mask(traj.model)
    return sum(traj.counts[c, 1] for c in eachindex(infected) if infected[c]; init = 0)
end

# Right-continuous piecewise-constant lookup: the value after every snapshot at or before `t` (so several snapshots
# at the same time, e.g. an intervention applied at t0, resolve to the last one).
function _interp(times::AbstractVector{<:Real},
                 vals::AbstractVector{<:Real}, t::Real)
    t < times[1] && return Float64(vals[1])
    return Float64(vals[searchsortedlast(times, Float64(t))])
end

function quantile(v::AbstractVector{<:Real}, q::Real)
    # local minimal quantile (type-7, as Statistics.quantile) to avoid a dependency for one call
    n = length(v)
    sorted = sort(collect(v))
    h = (n - 1) * q + 1
    lo = floor(Int, h)
    hi = ceil(Int, h)
    lo == hi && return sorted[lo]
    return sorted[lo] + (h - lo) * (sorted[hi] - sorted[lo])
end
