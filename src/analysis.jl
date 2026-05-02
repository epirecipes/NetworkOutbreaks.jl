#=
analysis.jl

Common observables computed from one or many trajectories.

Each trajectory has its own irregular event time grid. To average across
runs we resample onto a common grid via piecewise-constant interpolation.
=#

"""
    mean_curve(ens, sym; tgrid = nothing)

Return `(t, μ)` where `t` is a common time grid and `μ[k]` is the mean
count in compartment `sym` at time `t[k]` across the ensemble.

If `tgrid` is `nothing`, a 200-point linear grid covering the spec's
`tspan` is used.
"""
function mean_curve(ens::OutbreakEnsemble, sym::Symbol;
                    tgrid::Union{Nothing, AbstractVector{<:Real}} = nothing)
    t = isnothing(tgrid) ?
        collect(range(ens.spec.tspan[1], ens.spec.tspan[2]; length = 200)) :
        collect(Float64.(tgrid))
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
    quantile_band(ens, sym; q = (0.025, 0.975), tgrid = nothing)

Return `(t, lo, hi)` where `lo[k]` and `hi[k]` are the `q[1]` and `q[2]`
quantiles of compartment `sym` at time `t[k]` across the ensemble.
"""
function quantile_band(ens::OutbreakEnsemble, sym::Symbol;
                       q::Tuple{<:Real, <:Real} = (0.025, 0.975),
                       tgrid::Union{Nothing, AbstractVector{<:Real}} = nothing)
    t = isnothing(tgrid) ?
        collect(range(ens.spec.tspan[1], ens.spec.tspan[2]; length = 200)) :
        collect(Float64.(tgrid))
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
    final_size(traj; recovered = nothing)

Total fraction of nodes that have been infected at least once during the
trajectory. Computed from `final_infection_counts`. The `recovered`
keyword is accepted for API symmetry but ignored — the per-node infection
counter is the more direct quantity.
"""
function final_size(traj::OutbreakTrajectory; recovered = nothing)
    return count(>(0), traj.final_infection_counts) / length(traj.final_infection_counts)
end

"""
    reinfection_histogram(traj; L = nothing)

Histogram of per-node infection counts at the end of `traj`. Returns a
`Vector{Int}` of length `L+1` (or `maximum + 1` if `L` is `nothing`)
where index `p+1` holds the number of nodes infected `p` times. Counts
above `L` are saturated into the top bucket — matching the convention of
`with_reinfection_counting`.
"""
function reinfection_histogram(traj::OutbreakTrajectory;
                               L::Union{Integer, Nothing} = nothing)
    counts = traj.final_infection_counts
    Lmax = isnothing(L) ? maximum(counts; init = 0) : Int(L)
    h = zeros(Int, Lmax + 1)
    @inbounds for c in counts
        idx = min(c, Lmax)
        h[idx + 1] += 1
    end
    return h
end

# --- internals ---

function _interp(times::AbstractVector{<:Real},
                 vals::AbstractVector{<:Real}, t::Real)
    if t <= times[1]
        return Float64(vals[1])
    elseif t >= times[end]
        return Float64(vals[end])
    end
    k = searchsortedlast(times, Float64(t))
    return Float64(vals[k])
end

function quantile(v::AbstractVector{<:Real}, q::Real)
    # local minimal quantile to avoid pulling in Statistics for one call
    n = length(v)
    sorted = sort(collect(v))
    h = (n - 1) * q + 1
    lo = floor(Int, h)
    hi = ceil(Int, h)
    lo == hi && return sorted[lo]
    return sorted[lo] + (h - lo) * (sorted[hi] - sorted[lo])
end
