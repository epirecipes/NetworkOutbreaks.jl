# Owner: WP27 (DESIGN_NetworkEpiCore.md §G.2 WP27, §E.2, §E.4).
#=
validation/summarise.jl

`summarise(ens)`: a ScenarioEnsemble to a NetworkEpiCore EnsembleSummary (design §E.4): P(major) with its Wilson
interval, the pointwise statistics of every observable over the conditioned runs and over all runs (time-aligned for
a `CumulativeCrossing` scenario), and the per-run values (final size, prevalence peak, major flag, shifts, realised
graph statistics). The rules are those of validation/conditioning.jl.

Stored precision. Every number in a summary is a population fraction (or a time, or a graph statistic), and the
comparisons that read summaries (NetworkEpiCore `compare`) work in absolute units with a standard-error floor of
1e-4 and tolerances of 5e-3 (design §E.2). Each statistic is therefore rounded to a fixed number of decimal places,
far below its own Monte Carlo error (`SUMMARY_DIGITS`): the mean and the standard error to 6 (resolution 1e-6, 1% of
the se floor), the standard deviation and the quantiles to 5 (their sampling errors are ≳ 5e-4 at 200 runs, and the
quantiles of counts/N are resolved to 1/N anyway), peak times and alignment shifts to 6; the realised graph statistics
to 6 significant digits; final sizes and peak values (counts/N) are stored exactly. This keeps the committed files
near or below 200 kB (design §E.4). The rounding is deterministic, so a summary is still a pure function of the
ensemble and regeneration is bit-identical.
=#

export summarise

"""
    NetworkOutbreaks.SUMMARY_DIGITS

The number of decimal places to which [`summarise`](@ref) rounds what it stores: `mean` and `se` (6) and `sd` and
`quantiles` (5) of the pointwise statistics, and `times` (6: peak times and time-alignment shifts). Each is far below
the Monte Carlo error of the statistic (see src/validation/summarise.jl).
"""
const SUMMARY_DIGITS = (mean = 6, se = 6, sd = 5, quantiles = 5, times = 6)

"""
    NetworkOutbreaks.SUMMARY_REVISION

Revision of the summarising code (conditioning, alignment, statistics, rounding), recorded in the provenance of every
summary and checked when one is loaded: [`scenario_summary`](@ref) and [`missing_scenario_summaries`](@ref) treat a
summary of another revision as stale. Bump it whenever a change alters what [`summarise`](@ref) computes from a
given ensemble, then regenerate the committed summaries.

History: "1" the first; "2" `[extras.conditioning]` of `Survival` scenarios describes the prevalence at t_end, the
quantity the rule selects on (it described new infections).
"""
const SUMMARY_REVISION = "2"

const _REALISED_SIGDIGITS = 6
const _QUANTILES = (0.025, 0.25, 0.5, 0.75, 0.975)

_round_to(x::Float64, d::Int) = round(x; digits = d)
_round_time(x::Float64) = _round_to(x, SUMMARY_DIGITS.times)

"""
    summarise(ens::ScenarioEnsemble; scenario = ens.scenario) -> EnsembleSummary

Summarise the reference ensemble of a scenario (design §E.2, §E.4) as a NetworkEpiCore `EnsembleSummary`, the object
that `save_summary` commits and that EdgeBasedModels and NodeBasedModels compare their curves with.

**Conditioning** (`sc.sim.condition`, see src/validation/conditioning.jl): `MajorOutbreak(c)` keeps the runs whose
infections by t_end, excluding the seeds, are at least c·N; `Survival()` the runs with positive prevalence at t_end;
`Unconditioned()` every run. `n_major` runs are kept, `p_major = n_major/nsims` with its 95% Wilson interval
([`wilson_interval`](@ref)).

**Alignment** (`sc.sim.align`): with `CumulativeCrossing(ℓ)` every run that reaches ℓ of cumulative incidence
excluding the seeds is shifted to reach it at the reference time t*, the grid time nearest the median crossing time
of the conditioned runs; `shifts[i]` = t_i − t* (aligned value at grid time t = the run's value at t + shifts[i]; 0
for a run that never crosses). Samples outside a run's observation window are missing unless the run is absorbed.

**Statistics.** `cond` (over the conditioned runs) and `uncond` (over all runs) hold, for every observable of
`sc.observables` (population fractions: species, `:infectious`, `:cumulative`), the pointwise mean, standard
deviation, standard error of the mean and the 2.5, 25, 50, 75 and 97.5% quantiles (type 7) on `sc.tgrid`. The
standard error is sd/√n for independent runs (`graphs = :per_run`; with `:fixed` it is conditional on the one graph)
and the between-graph (cluster) standard error with `(:pool, G)`. Without alignment every run has a sample at every
grid time; with alignment n(t) is stored in `extras[:alignment]`. A selection without runs gives `NaN` statistics and
a warning. Values are rounded far below their Monte Carlo errors ([`NetworkOutbreaks.SUMMARY_DIGITS`](@ref)).

**Per run:** `final_size` (fraction ever infected, including the seeds: `final_size(traj)`), `peak` (time on the
run's own clock and value of the prevalence maximum), `major`, `shifts` (aligned scenarios only) and `realised`
(the run's graph: `:mean_degree`, `:excess_degree`, `:clustering`, `:erased_fraction`, where they apply).

**Extras** (TOML values): `:conditioning` (the rule, `n_selected`, and the `measure` the rule selects on with its
largest value over the discarded runs and its smallest over the kept ones, `largest_unselected` <
`smallest_selected`: for `MajorOutbreak` the new-infection fraction, which shows whether the threshold sits in the
gap of the bimodal final-size distribution; for `Survival` the prevalence at t_end), `:sampling` (graphs and standard-error method),
`:alignment` (aligned scenarios: rule, level, `reference_time` t*, `reference_index`, `n_crossed`, `cond_n` and
`uncond_n` = n(t)) and, for models with arrows back into the susceptible class (SIS, SIRS),
`:reinfection_histogram` (`cond`, `uncond`: the mean fraction of nodes infected p = 0, 1, … times). **Provenance**:
Julia and package versions, the algorithm, `ALGORITHM_REVISION` and `SUMMARY_REVISION`; no date or wall time, so the
files of a regenerated summary are bit-identical.

`scenario` may replace the ensemble's scenario by one that differs only in `sim.condition` or `sim.align` (the runs
do not depend on either); the alignment must then be `NoAlignment()` or the ensemble's own, whose aligned samples the
ensemble holds (see [`unaligned_scenario`](@ref)). The summary carries the id and hash of `scenario`.
"""
function summarise(ens::ScenarioEnsemble; scenario::Scenario = ens.scenario)
    sc = scenario
    _check_summarisable(ens, sc)
    N = sc.sim.N
    runs = ens.runs
    n = length(runs)
    sel = BitVector([_selected(sc.sim.condition, run, N) for run in runs])
    n_major = count(sel)
    tgrid = collect(Float64, sc.tgrid)
    nt = length(tgrid)
    aligned = sc.sim.align isa CumulativeCrossing
    kstar = aligned ? _reference_index(runs, sel, tgrid) : 0
    use_aligned = kstar > 0
    cluster = sc.sim.graphs isa Tuple && !(sc.network isa WellMixed)
    all_idx = collect(1:n)
    sel_idx = findall(sel)
    n_major == 0 && @warn "summarise(:$(sc.id)): no run satisfies the conditioning rule $(_rule_text(sc.sim.condition)) (0 of $(n) runs); the conditioned statistics are NaN"
    cond = Dict{Symbol, SummaryStats}()
    uncond = Dict{Symbol, SummaryStats}()
    cond_n = zeros(Int, nt)
    uncond_n = zeros(Int, nt)
    for (o, X) in enumerate(sc.observables)
        cond[X] = _pointwise(runs, sel_idx, o, kstar, use_aligned, nt, N, cluster, cond_n)
        uncond[X] = sel_idx == all_idx ? cond[X] : _pointwise(runs, all_idx, o, kstar, use_aligned, nt, N, cluster,
                                                               uncond_n)
    end
    sel_idx == all_idx && (uncond_n .= cond_n)
    shifts = if aligned
        tstar = use_aligned ? tgrid[kstar] : NaN
        Float64[isnan(r.crossing) || !use_aligned ? 0.0 : _round_time(r.crossing - tstar) for r in runs]
    else
        Float64[]
    end
    realised = Dict{Symbol, Vector{Float64}}(
        k => Float64[round(r.realised[i]; sigdigits = _REALISED_SIGDIGITS) for r in runs]
        for (i, k) in enumerate(ens.realised_names))
    extras = Dict{Symbol, Any}(:conditioning => _conditioning_extras(sc, runs, sel, N),
                               :sampling => _sampling_extras(sc, cluster))
    if aligned
        extras[:alignment] = Dict{String, Any}(
            "rule" => _rule_text(sc.sim.align), "level" => sc.sim.align.level,
            "reference_time" => use_aligned ? tgrid[kstar] : NaN, "reference_index" => kstar,
            "reference" => "the grid time nearest the median crossing time of the conditioned runs that cross",
            "n_crossed" => count(r -> !isnan(r.crossing), runs), "cond_n" => cond_n, "uncond_n" => uncond_n)
    end
    any(r -> !isempty(r.histogram), runs) && (extras[:reinfection_histogram] = _histogram_extras(runs, sel, N))
    return EnsembleSummary(; id = sc.id, scenario_hash = scenario_hash(sc), algorithm_revision = ALGORITHM_REVISION,
                           N, nsims = n, n_major, p_major = n_major / n, p_major_ci = wilson_interval(n_major, n),
                           t = tgrid, observables = copy(sc.observables), cond, uncond,
                           final_size = Float64[r.final_infected / N for r in runs],
                           peak = NTuple{2, Float64}[(_round_time(r.peak_time), r.peak_count / N) for r in runs],
                           major = sel, shifts, realised, extras, provenance = _provenance(sc))
end

# The canonical text of a scenario without the conditioning and alignment lines: what the simulation reads.
function _simulation_text(sc::Scenario)
    lines = split(canonical_text(sc), '\n')
    return join(filter(l -> !(startswith(l, "sim.condition = ") || startswith(l, "sim.align = ")), lines), '\n')
end

function _check_summarisable(ens::ScenarioEnsemble, sc::Scenario)
    sc === ens.scenario && return nothing
    _simulation_text(sc) == _simulation_text(ens.scenario) || throw(ArgumentError(
        "summarise: the scenario :$(sc.id) does not describe the simulation of the ensemble (:$(ens.scenario.id)); " *
        "only the conditioning rule and the alignment may differ"))
    a = sc.sim.align
    (a isa NoAlignment || a == ens.scenario.sim.align) || throw(ArgumentError(
        "summarise: the ensemble of :$(ens.scenario.id) holds aligned samples for $(ens.scenario.sim.align), not " *
        "for $(a); run scenario_ensemble on the scenario with that alignment (its runs are the same)"))
    return nothing
end

# The sample of observable o of run `run` at grid index k (−1 when missing).
@inline function _sample_at(run::ScenarioRun, o::Int, k::Int, kstar::Int, use_aligned::Bool, nt::Int)
    (use_aligned && !isempty(run.aligned)) && return run.aligned[o, k - kstar + nt]
    return run.grid[o, k]
end

# Pointwise statistics of observable o over the runs `which`; n(t) is written into `nvec`.
function _pointwise(runs::Vector{ScenarioRun}, which::Vector{Int}, o::Int, kstar::Int, use_aligned::Bool, nt::Int,
                    N::Int, cluster::Bool, nvec::Vector{Int})
    stats = ntuple(_ -> fill(NaN, nt), 8)
    buf = Float64[]
    gbuf = Int[]
    for k in 1:nt
        empty!(buf)
        empty!(gbuf)
        for i in which
            v = _sample_at(runs[i], o, k, kstar, use_aligned, nt)
            v == _MISSING_SAMPLE && continue
            push!(buf, v / N)
            push!(gbuf, runs[i].graph)
        end
        m = length(buf)
        nvec[k] = m
        m == 0 && continue
        μ = sum(buf) / m
        sd = m > 1 ? sqrt(sum(x -> (x - μ)^2, buf) / (m - 1)) : NaN
        se = cluster ? _cluster_se(buf, gbuf, μ) : sd / sqrt(m)
        qs = Statistics.quantile!(buf, _QUANTILES)
        stats[1][k] = _round_to(μ, SUMMARY_DIGITS.mean)
        stats[2][k] = _round_to(sd, SUMMARY_DIGITS.sd)
        stats[3][k] = _round_to(se, SUMMARY_DIGITS.se)
        for q in 1:5
            stats[3 + q][k] = _round_to(Float64(qs[q]), SUMMARY_DIGITS.quantiles)
        end
    end
    return SummaryStats(stats)
end

# Between-graph standard error of the mean μ of the values v on graphs g: the cluster-robust (sandwich) variance
# G/(G − 1)·Σ_g e_g²/n² with e_g = Σ_{i ∈ g}(v_i − μ), over the G graphs that have values.
function _cluster_se(v::Vector{Float64}, g::Vector{Int}, μ::Float64)
    e = zeros(maximum(g))
    for (x, j) in zip(v, g)
        e[j] += x - μ
    end
    used = falses(length(e))
    for j in g
        used[j] = true
    end
    G = count(used)
    G >= 2 || return NaN
    s = 0.0
    for j in eachindex(e)
        used[j] && (s += e[j]^2)
    end
    return sqrt(G / (G - 1) * s) / length(v)
end

function _conditioning_extras(sc::Scenario, runs, sel::BitVector, N::Int)
    c = sc.sim.condition
    fr = Float64[_selection_measure(c, r, N) for r in runs]
    kept = fr[sel]
    dropped = fr[.!sel]
    return Dict{String, Any}("rule" => _rule_text(c), "n_selected" => count(sel),
                             "largest_unselected" => isempty(dropped) ? NaN : maximum(dropped),
                             "smallest_selected" => isempty(kept) ? NaN : minimum(kept),
                             "measure" => _selection_measure_text(c))
end

function _sampling_extras(sc::Scenario, cluster::Bool)
    g = sc.sim.graphs
    graphs = sc.network isa WellMixed ? (sc.sim.algorithm === :mass_action ? "none" : "complete") :
             g isa Tuple ? "pool" : String(g)
    d = Dict{String, Any}("graphs" => graphs, "se" => cluster ? "cluster (between-graph)" : "iid (sd/sqrt(n))",
                          "algorithm" => string(nameof(typeof(_algorithm(sc.sim.algorithm)))),
                          "base_seed" => string(sc.sim.base_seed))
    g isa Tuple && (d["pool_size"] = last(g))
    return d
end

function _histogram_extras(runs, sel::BitVector, N::Int)
    L = maximum(r -> length(r.histogram), runs)
    mean_hist(which) = begin
        h = zeros(L)
        isempty(which) && return fill(NaN, L)
        for i in which
            hi = runs[i].histogram
            for p in eachindex(hi)
                h[p] += hi[p] / N
            end
        end
        _round_to.(h ./ length(which), SUMMARY_DIGITS.mean)
    end
    return Dict{String, Any}("cond" => mean_hist(findall(sel)), "uncond" => mean_hist(collect(eachindex(runs))),
                             "index" => "entry p + 1 is the mean fraction of nodes infected p times")
end

function _provenance(sc::Scenario)
    return Dict{String, String}(
        "julia_version" => string(VERSION),
        "NetworkOutbreaks" => string(pkgversion(@__MODULE__)),
        "NetworkEpiCore" => string(pkgversion(NetworkEpiCore)),
        "Graphs" => string(pkgversion(Graphs)),
        "StableRNGs" => string(pkgversion(parentmodule(StableRNG))),
        "algorithm" => string(nameof(typeof(_algorithm(sc.sim.algorithm)))),
        "algorithm_revision" => ALGORITHM_REVISION,
        "summary_revision" => SUMMARY_REVISION,
        "summary_digits" => join(("$(k)=$(v)" for (k, v) in pairs(SUMMARY_DIGITS)), ","),
        "generator" => "NetworkOutbreaks.summarise(scenario_ensemble(sc))")
end
