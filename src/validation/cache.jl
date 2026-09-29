# Owner: WP27 (DESIGN_NetworkEpiCore.md §G.2 WP27, §E.1, §E.4).
#=
validation/cache.jl

The committed reference summaries and their cache (design §E.4):

- committed summaries live in NetworkOutbreaks' `data/scenarios/` (`scenario_data_dir()`), three files per scenario
  named `<id>__<first 8 hex digits of scenario_hash(sc)>` (NetworkEpiCore `save_summary`);
- the cache key is (`scenario_hash(sc)`, `ALGORITHM_REVISION`), and the summary must also have been made by the
  current summarising code (`SUMMARY_REVISION`, in its provenance): `load_summary` refuses a file whose stored hash
  differs from the requested one, `scenario_summary` passes `algorithm_revision = ALGORITHM_REVISION` and checks the
  summary revision, so a stale summary is never loaded;
- `scenario_summary(sc; policy)` loads (`:committed`), loads or computes (`:auto`, into a user cache directory, never
  into the package) or computes (`:recompute`). With `NETEPI_STRICT_CACHE=1` (set in CI) `:auto` uses the committed
  summaries only: it neither reads the user cache (which depot caching can carry into CI) nor computes, so a missing
  or stale committed summary is an error;
- `regenerate_scenarios(ids)` (and scripts/regenerate_scenarios.jl) rewrites the committed files, together with the
  unaligned companions of the time-aligned scenarios, and removes the files of the same ids with other hashes.
=#

export scenario_summary, regenerate_scenarios, scenario_data_dir, scenario_cache_dir, missing_scenario_summaries

"""
    scenario_data_dir() -> String

The directory of the committed reference summaries: `data/scenarios/` of the NetworkOutbreaks package (see its
README.md).
"""
scenario_data_dir() = normpath(joinpath(@__DIR__, "..", "..", "data", "scenarios"))

"""
    scenario_cache_dir() -> String

Where `scenario_summary(sc; policy = :auto)` and `policy = :recompute` store summaries they compute: the directory
named by the environment variable `NETEPI_CACHE_DIR` if it is set, otherwise
`<first depot>/scratchspaces/<NetworkOutbreaks UUID>/scenarios`. Never the package directory.
"""
function scenario_cache_dir()
    d = get(ENV, "NETEPI_CACHE_DIR", "")
    isempty(d) || return d
    return joinpath(first(DEPOT_PATH), "scratchspaces", string(Base.PkgId(@__MODULE__).uuid), "scenarios")
end

# NETEPI_STRICT_CACHE=1 (or true/yes): committed summaries only (no user cache, no computation).
_strict_cache() = lowercase(strip(get(ENV, "NETEPI_STRICT_CACHE", ""))) in ("1", "true", "yes")

# The valid summary of `sc` in `dir`, or `nothing` with the reason pushed onto `reasons`. Valid: the stored scenario
# hash is `scenario_hash(sc)`, the algorithm revision is `ALGORITHM_REVISION` (both checked by `load_summary`) and
# the summary revision in the provenance is `SUMMARY_REVISION`.
function _try_load(dir::AbstractString, sc::Scenario, reasons::Vector{String})
    s = try
        load_summary(dir, sc; algorithm_revision = ALGORITHM_REVISION)
    catch err
        err isa ArgumentError || rethrow()
        push!(reasons, sprint(showerror, err))
        return nothing
    end
    rev = get(s.provenance, "summary_revision", "none")
    if rev != SUMMARY_REVISION
        push!(reasons, "the summary $(summary_basename(sc.id, s.scenario_hash)) in $(dir) was made with summary " *
                       "revision $(rev), not $(SUMMARY_REVISION) (a stale summary); regenerate it")
        return nothing
    end
    return s
end

"""
    scenario_summary(id_or_sc; policy = :committed, dir = scenario_data_dir(), cache_dir = scenario_cache_dir(),
                     parallel = false) -> EnsembleSummary

The reference summary of a scenario (a registered id such as `:sir_pois5`, or a `Scenario`, e.g. one made with
`derive`), keyed by (`scenario_hash(sc)`, `ALGORITHM_REVISION`) (design §E.4) and made by the current summarising
code (`NetworkOutbreaks.SUMMARY_REVISION`). It never returns a stale summary:

- `policy = :committed` (the default): load the committed summary from `dir`. A missing summary, one with another
  scenario hash, or one from another `ALGORITHM_REVISION` or `SUMMARY_REVISION` is an `ArgumentError` that says how
  to regenerate it.
- `policy = :auto`: the committed summary if valid, else a valid one in `cache_dir`, else compute it
  ([`scenario_ensemble`](@ref) and [`summarise`](@ref), `parallel` threads), save it in `cache_dir` (unless
  `cache_dir = nothing`) and return it; the reason (missing or stale) is logged. With the environment variable
  `NETEPI_STRICT_CACHE=1` (set in CI) only the committed summary is used: `cache_dir` is not read and nothing is
  computed, so a missing or stale committed summary is an `ArgumentError`, whatever the user cache holds.
- `policy = :recompute`: compute it, save it in `cache_dir` (unless `nothing`) and return it (an explicit request,
  also under `NETEPI_STRICT_CACHE`).

Summaries are computed with exactly the streams of the committed ones, so a recomputed summary equals the committed
one of the same hash and revision.

```julia
ref = scenario_summary(:sir_pois5)                  # N = 10⁴, 200 runs, fresh graph per run
ref.p_major, ref.cond[:I].mean                      # P(major), mean prevalence of the major runs
```
"""
function scenario_summary(x::Union{Symbol, Scenario}; policy::Symbol = :committed,
                          dir::AbstractString = scenario_data_dir(),
                          cache_dir::Union{Nothing, AbstractString} = scenario_cache_dir(),
                          parallel::Bool = false)
    policy in (:committed, :auto, :recompute) || throw(ArgumentError(
        "scenario_summary: policy must be :committed, :auto or :recompute; got $(repr(policy))"))
    sc = scenario(x)
    policy === :recompute && return _compute_summary(sc, cache_dir, parallel)
    reasons = String[]
    s = _try_load(dir, sc, reasons)
    s === nothing || return s
    key = "hash $(first(scenario_hash(sc), 8)), algorithm revision $(ALGORITHM_REVISION), summary revision " *
          "$(SUMMARY_REVISION)"
    if policy === :committed
        throw(ArgumentError(
            "scenario_summary(:$(sc.id)): no valid committed summary ($(key)) in $(dir): $(only(reasons)). " *
            "Regenerate it with NetworkOutbreaks' scripts/regenerate_scenarios.jl, or pass policy = :auto to " *
            "compute it"))
    end
    # strict mode: the committed summaries only; a user cache (possibly restored from a depot cache) is not read
    _strict_cache() && throw(ArgumentError(
        "scenario_summary(:$(sc.id)): no valid committed summary ($(key)) in $(dir), and NETEPI_STRICT_CACHE is " *
        "set, so neither the user cache nor a simulation is used: $(only(reasons)). Regenerate it with " *
        "NetworkOutbreaks' scripts/regenerate_scenarios.jl and commit it"))
    if cache_dir !== nothing
        s = _try_load(cache_dir, sc, reasons)
        s === nothing || return s
    end
    @info "scenario_summary(:$(sc.id)): no valid summary ($(join(reasons, "; "))); simulating $(sc.sim.nsims) runs on N = $(sc.sim.N) nodes"
    return _compute_summary(sc, cache_dir, parallel)
end

function _compute_summary(sc::Scenario, cache_dir, parallel::Bool)
    s = summarise(scenario_ensemble(sc; parallel))
    cache_dir === nothing || save_summary(cache_dir, s)
    return s
end

# The summary files of `id` in `dir` with another hash than `hash`.
function _stale_files(dir::AbstractString, id::Symbol, hash::AbstractString)
    isdir(dir) || return String[]
    pat = Regex("^" * string(id) * "__([0-9a-f]{8})\\.(toml|curves\\.csv|runs\\.csv)\$")
    out = String[]
    for f in readdir(dir)
        m = match(pat, f)
        m === nothing && continue
        m.captures[1] == first(hash, 8) || push!(out, joinpath(dir, f))
    end
    return out
end

# The scenarios whose summaries a regeneration writes for `sc`: itself and, when time-aligned, its unaligned companion.
_summary_targets(sc::Scenario, companions::Bool) =
    companions && sc.sim.align isa CumulativeCrossing ? Scenario[sc, unaligned_scenario(sc)] : Scenario[sc]

_summary_key(sc::Scenario) = (sc.id, scenario_hash(sc))

# The ensembles that a regeneration of `ids` simulates, in order, each with the scenarios it is summarised as
# (`_summary_targets`). Every summary appears once: a repeated id is simulated once, and a scenario given in `ids` that
# is also the unaligned companion of another one given (the same id and hash, e.g. a registered `<id>_unaligned`) is
# not simulated on its own, since the ensemble of the aligned scenario has the same runs.
function _regeneration_plan(ids, companions::Bool)
    groups = [sc => _summary_targets(sc, companions) for sc in (scenario(x) for x in ids)]
    as_companion = Set(_summary_key(t) for (_, ts) in groups for t in ts[2:end])
    seen = Set{Tuple{Symbol, String}}()
    plan = Pair{Scenario, Vector{Scenario}}[]
    for (sc, ts) in groups
        k = _summary_key(sc)
        (k in seen || k in as_companion) && continue
        new = Scenario[t for t in ts if !(_summary_key(t) in seen)]
        foreach(t -> push!(seen, _summary_key(t)), new)
        push!(plan, sc => new)
    end
    return plan
end

_default_ids() = scenario_ids(; exclude = [:deferred])

"""
    regenerate_scenarios(ids = scenario_ids(; exclude = [:deferred]); dir = scenario_data_dir(),
                         parallel = Threads.nthreads() > 1, companions = true, prune = true, io = stdout)
        -> Vector{NamedTuple}

Simulate the reference ensembles of the scenarios `ids` (registered ids or `Scenario`s) and write their summaries to
`dir` (by default the committed `data/scenarios/`) with `save_summary`: the files `<id>__<hash8>.{toml,curves.csv,
runs.csv}` of design §E.4. By default every registered scenario that NetworkOutbreaks can simulate (those not tagged
`:deferred`), including the N-scaling and derived variants.

- `companions = true`: a time-aligned scenario (`CumulativeCrossing`) also gets the summary of its
  [`unaligned_scenario`](@ref), from the same ensemble. A companion that is also in `ids` (e.g. a registered
  `<id>_unaligned` with the same hash) is written once, from that ensemble, and not simulated again; repeated ids
  are simulated once;
- `prune = true`: files of the same ids with another hash (stale summaries) are removed from `dir`;
- `parallel`: runs of each ensemble on threads (the result does not depend on it).

The files are a pure function of the scenario, `ALGORITHM_REVISION`, `SUMMARY_REVISION` and the package versions
(no date or wall time), so regenerating gives bit-identical files. One line per summary (id, hash, runs, P(major),
wall time) is printed to `io` (`nothing` for none), and the same data is returned as `(id, hash, files, nsims,
n_major, p_major, seconds)` NamedTuples. A scenario that cannot be simulated is an error, raised after the others
have been written (with every failure listed).
"""
function regenerate_scenarios(ids = _default_ids(); dir::AbstractString = scenario_data_dir(),
                              parallel::Bool = Threads.nthreads() > 1, companions::Bool = true,
                              prune::Bool = true, io::Union{Nothing, IO} = stdout)
    results = NamedTuple[]
    failures = Pair{Symbol, String}[]
    for (sc, targets) in _regeneration_plan(ids, companions)
        start = time_ns()
        ens = try
            scenario_ensemble(sc; parallel)
        catch err
            err isa InterruptException && rethrow()
            push!(failures, sc.id => sprint(showerror, err))
            io === nothing || println(io, rpad(string(sc.id), 28), " FAILED: ", last(failures[end]))
            nothing
        end
        ens === nothing && continue
        t = (time_ns() - start) / 1e9
        for target in targets
            s = summarise(ens; scenario = target)
            paths = save_summary(dir, s)
            if prune
                foreach(rm, _stale_files(dir, target.id, s.scenario_hash))
            end
            io === nothing || println(io, rpad(string(target.id), 28), " ", first(s.scenario_hash, 8), "  N = ",
                                      rpad(string(s.N), 7), " runs ", rpad(string(s.nsims), 5), " P(major) ",
                                      rpad(string(round(s.p_major; digits = 4)), 7), " ",
                                      round(t; digits = 1), " s")
            push!(results, (id = target.id, hash = s.scenario_hash, files = paths, nsims = s.nsims,
                            n_major = s.n_major, p_major = s.p_major, seconds = t))
        end
    end
    isempty(failures) || throw(ErrorException(
        "regenerate_scenarios: $(length(failures)) scenario(s) could not be simulated:\n" *
        join(("  :$(k): $(v)" for (k, v) in failures), "\n")))
    return results
end

"""
    missing_scenario_summaries(ids = scenario_ids(; exclude = [:deferred]); dir = scenario_data_dir(),
                               companions = true) -> Vector{Pair{Symbol,String}}

The summaries that [`regenerate_scenarios`](@ref)`(ids; companions)` writes (the scenarios `ids` and, with
`companions`, the unaligned companions of the time-aligned ones, each once) that have no valid summary in `dir`,
each with the reason: missing, another scenario hash, another `ALGORITHM_REVISION` or another `SUMMARY_REVISION`.
Empty when every summary that `scenario_summary(...; policy = :committed)` would load is present; this is the check
of `scripts/regenerate_scenarios.jl --check`.
"""
function missing_scenario_summaries(ids = _default_ids(); dir::AbstractString = scenario_data_dir(),
                                    companions::Bool = true)
    out = Pair{Symbol, String}[]
    for (_, targets) in _regeneration_plan(ids, companions), sc in targets
        reasons = String[]
        _try_load(dir, sc, reasons) === nothing && push!(out, sc.id => only(reasons))
    end
    return out
end
