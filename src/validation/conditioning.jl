# Owner: WP27 (DESIGN_NetworkEpiCore.md §G.2 WP27, §E.2, §E.4).
#=
validation/conditioning.jl

Conditioning, time alignment and the Wilson interval of the scenario runner (design §E.2; verified issue N02, whose
skeptic's corrected fix this follows).

Definitions used by the whole runner (runner.jl, summarise.jl), for a run of a Scenario on N nodes:

- An *infection* is an event that moves a node from a non-infected into an infected compartment, with the structural
  infected compartments of `_infected_mask` (design §J.8; the rule of `final_size` and of the edge-based `:cumulative`
  accumulator). The *seeds* are the nodes that start infected.
- `:cumulative(t)` = (seeds + infections up to t)/N: the fraction ever infected for T_EB models (it ends at
  `final_size`), cumulative incidence for SIS and SIRS. *Cumulative incidence excluding the seeds* is
  (infections up to t)/N.
- `MajorOutbreak(c)`: the run is major if its infections by t_end, as a fraction of N, are ≥ c (N02 corrected fix
  (1): the seeds are never counted, so the rule is not vacuous when the seed fraction exceeds c).
  `Survival()`: prevalence (the `:infectious` observable) at t_end is positive. `Unconditioned()`: every run.
- `CumulativeCrossing(ℓ)`: run i *crosses* at t_i, the time of its m-th infection, where m is the smallest integer
  with m/N ≥ ℓ (N02 corrected fix (2): alignment on cumulative incidence, which is monotone, never on a compartment
  level with an inferred direction). The reference time t* is the grid time nearest the median crossing time of the
  conditioned runs that cross (of all crossing runs if none of them does), and run i is shifted by
  s_i = t_i − t*: its aligned value at grid time t_k is its value at its own time t_i + (k − k*)·Δt (Δt the grid
  step, t* = t_{k*}), i.e. x_i(t_k + s_i). Runs that never cross are not shifted (s_i = 0).
- Masking (N02 corrected fix (3)): an aligned sample before the run's start t0, or after its end t_end while the run
  can still change, is *missing*; after the end of an absorbed run (no transition can fire any more) the final value
  is exact and is used. Statistics at each grid time are over the runs with a sample there, and that number n(t) is
  stored with the summary.

The reference time t* is a statistic of the ensemble, not the deterministic curve's crossing time (NetworkOutbreaks
does not depend on the deterministic back ends): a deterministic curve is compared with an aligned summary after it
is shifted to cross ℓ at t* ([`aligned_curves`](@ref)), which is the same comparison up to a common time translation.
=#

import Statistics          # median, quantile! (the validation files; Statistics is a dependency)

export wilson_interval, unaligned_scenario, aligned_curves

"""
    NetworkOutbreaks.WILSON_Z

The two-sided 95% standard normal quantile Φ⁻¹(0.975) = 1.959963984540054 used by [`wilson_interval`](@ref).
"""
const WILSON_Z = 1.959963984540054

"""
    wilson_interval(k::Integer, n::Integer; z = NetworkOutbreaks.WILSON_Z) -> (lo, hi)

The Wilson score interval for a binomial proportion with `k` successes in `n` trials (95% by default; `z` is the
standard normal quantile of the level):

    (p̂ + z²/2n ± z·√(p̂(1 − p̂)/n + z²/4n²)) / (1 + z²/n),   p̂ = k/n,

clamped to [0, 1] and exactly 0 (1) at k = 0 (k = n). The scenario summaries report P(major) = n_major/nsims with
this interval (design §E.2). Unlike the Wald interval it has good coverage near 0 and 1, where P(major) of the
canonical scenarios lies.

```julia
wilson_interval(81, 263)       # (0.25529, 0.36621), Newcombe (1998), Stat. Med. 17:857, example (c)
wilson_interval(0, 10)         # (0.0, 0.27753)
```
"""
function wilson_interval(k::Integer, n::Integer; z::Real = WILSON_Z)
    n >= 1 || throw(ArgumentError("wilson_interval: n must be ≥ 1; got $(n)"))
    0 <= k <= n || throw(ArgumentError("wilson_interval: k must be in 0:n = 0:$(n); got $(k)"))
    (isfinite(z) && z > 0) || throw(ArgumentError("wilson_interval: z must be finite and > 0; got $(z)"))
    p = k / n
    z2 = Float64(z)^2
    denom = 1 + z2 / n
    centre = (p + z2 / (2n)) / denom
    half = Float64(z) / denom * sqrt(p * (1 - p) / n + z2 / (4 * Float64(n)^2))
    lo = k == 0 ? 0.0 : clamp(centre - half, 0.0, 1.0)
    hi = k == n ? 1.0 : clamp(centre + half, 0.0, 1.0)
    return (lo, hi)
end

# ---------------------------------------------------------------------------------------------------------------
# Conditioning (a run is summarised as a `ScenarioRun`, runner.jl)
# ---------------------------------------------------------------------------------------------------------------

# Is the run kept by the conditioning rule? (N nodes)
_selected(c::MajorOutbreak, run, N::Int) = run.new_infections / N >= c.threshold
_selected(::Survival, run, N::Int) = run.infectious_end > 0
_selected(::Unconditioned, run, N::Int) = true

# The quantity the rule selects on, as a fraction of N, with its description (the `[extras.conditioning]` of a
# summary records its largest value over the discarded runs and its smallest over the kept ones). Unconditioned keeps
# every run, so its new-infection fractions only describe the spread of the runs.
_selection_measure(::Union{MajorOutbreak, Unconditioned}, run, N::Int) = run.new_infections / N
_selection_measure(::Survival, run, N::Int) = run.infectious_end / N
_selection_measure_text(::MajorOutbreak) = "new infections by t_end (excluding the seeds) / N"
_selection_measure_text(::Survival) = "prevalence (infectious nodes) at t_end / N"
_selection_measure_text(::Unconditioned) = "new infections by t_end (excluding the seeds) / N (every run is kept)"

_rule_text(c) = sprint(show, c)

# ---------------------------------------------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------------------------------------------

# The number m of infections at which a run crosses the level ℓ on N nodes: the smallest m ≥ 1 with m/N ≥ ℓ (the
# comparison is made on the fraction, exactly as the rule is stated).
function _crossing_count(level::Float64, N::Int)
    m = max(1, ceil(Int, level * N))
    while m > 1 && (m - 1) / N >= level
        m -= 1
    end
    while m / N < level
        m += 1
    end
    return m
end

_crossing_count(::NoAlignment, N::Int) = 0
_crossing_count(a::CumulativeCrossing, N::Int) = _crossing_count(a.level, N)

# The reference grid index k* (t* = tgrid[k*]): the grid time nearest the median crossing time of the selected runs
# that cross, else of every crossing run; 0 when no run crosses. `tgrid` is uniform (a scenario's range); a one-point
# grid has k* = 1.
function _reference_index(runs, selected::AbstractVector{Bool}, tgrid::AbstractVector{Float64})
    ts = Float64[runs[i].crossing for i in eachindex(runs) if selected[i] && !isnan(runs[i].crossing)]
    isempty(ts) && (ts = Float64[r.crossing for r in runs if !isnan(r.crossing)])
    isempty(ts) && return 0
    length(tgrid) == 1 && return 1
    m = Statistics.median(ts)
    Δ = tgrid[2] - tgrid[1]
    return clamp(round(Int, (m - tgrid[1]) / Δ) + 1, 1, length(tgrid))
end

"""
    unaligned_scenario(sc::Scenario) -> Scenario

The companion of a time-aligned scenario (`sc.sim.align` a `CumulativeCrossing`, the small-seed scenarios of §E.2)
without alignment: `derive(sc; id = Symbol(sc.id, "_unaligned"), align = NoAlignment())`, with the tag `:unaligned`.
Alignment is applied when an ensemble is summarised, so the two scenarios have the same runs (same streams, same
graphs) and different hashes; `regenerate_scenarios` writes both summaries from one ensemble, and
`scenario_summary(unaligned_scenario(sc))` loads the unaligned one, for the "aligned vs unaligned" comparison of
design §E.2.
"""
function unaligned_scenario(sc::Scenario)
    sc.sim.align isa CumulativeCrossing || throw(ArgumentError(
        "unaligned_scenario(:$(sc.id)): the scenario is not time-aligned (align = $(sc.sim.align))"))
    return derive(sc; id = Symbol(sc.id, "_unaligned"), align = NoAlignment(),
                  title = string(sc.title, " (unaligned)"),
                  tags = unique!(vcat(sc.tags, [:derived, :unaligned])),
                  notes = string("The unaligned companion of :", sc.id, " (same runs). ", sc.notes))
end

"""
    aligned_curves(curves::ModelCurves, ref::EnsembleSummary) -> ModelCurves

Deterministic `curves` shifted in time to compare with a time-aligned summary `ref` (a scenario with
`align = CumulativeCrossing(ℓ)`). The runs of `ref` are aligned so that each crosses ℓ of cumulative incidence
excluding the seeds at the reference time t* (`ref.extras[:alignment]`, see [`summarise`](@ref)); the curves are
shifted so that theirs does too:

- t_det is the first time at which `curves[:cumulative](t) − curves[:cumulative](t₀)` reaches ℓ (linear
  interpolation between grid points); the curves need a `:cumulative` observable;
- the result, on `ref.t`, is x(t + t_det − t*) for every observable, linearly interpolated, and held at its first
  (last) value before (after) the range of `curves.t`. Holding the first value is an approximation (a deterministic
  small-seed curve changes by O(ρ) there); holding the last is exact once the curve has converged.

The shift t_det − t* is stored in `metadata[:alignment_shift]`. Then `compare(ref, aligned_curves(det, ref))` is the
aligned comparison of design §E.2.
"""
function aligned_curves(c::ModelCurves, ref::EnsembleSummary)
    haskey(ref.extras, :alignment) || throw(ArgumentError(
        "aligned_curves: the summary :$(ref.id) is not time-aligned (it has no `alignment` extras); compare the " *
        "curves with it directly"))
    al = ref.extras[:alignment]
    level = Float64(al["level"])
    tstar = Float64(al["reference_time"])
    isnan(tstar) && throw(ArgumentError(
        "aligned_curves: no run of the summary :$(ref.id) crossed the alignment level, so it has no reference time"))
    haskey(c, :cumulative) || throw(ArgumentError(
        "aligned_curves: the curves $(repr(c.label)) have no :cumulative observable, which the alignment needs"))
    t = c.t
    y = c[:cumulative]
    y0 = y[1]
    j = findfirst(v -> v - y0 >= level, y)
    j === nothing && throw(ArgumentError(
        "aligned_curves: the curves $(repr(c.label)) never reach $(level) of cumulative incidence excluding the " *
        "seeds on t ∈ [$(first(t)), $(last(t))]"))
    tdet = if j == 1
        t[1]
    else
        a, b = y[j - 1] - y0, y[j] - y0
        t[j - 1] + (level - a) / (b - a) * (t[j] - t[j - 1])
    end
    δ = tdet - tstar
    vals = Dict{Symbol, Vector{Float64}}(X => _interp_held(t, v, ref.t .+ δ) for (X, v) in c.values)
    meta = merge(Dict{Symbol, Any}(c.metadata), Dict{Symbol, Any}(:alignment_shift => δ))
    return ModelCurves(c.label, ref.t, vals, c.representation, meta)
end

# Linear interpolation of (t, y) at the points s, held at the end values outside [t[1], t[end]].
function _interp_held(t::Vector{Float64}, y::Vector{Float64}, s::AbstractVector{Float64})
    out = similar(s, Float64)
    for (i, x) in pairs(s)
        if x <= t[1]
            out[i] = y[1]
        elseif x >= t[end]
            out[i] = y[end]
        else
            j = searchsortedlast(t, x)
            w = (x - t[j]) / (t[j + 1] - t[j])
            out[i] = (1 - w) * y[j] + w * y[j + 1]
        end
    end
    return out
end
