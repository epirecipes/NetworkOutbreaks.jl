# Owner: WP27 (DESIGN_NetworkEpiCore.md §G.2 WP27, §E.1, §E.2).
#=
validation/runner.jl

The reference ensembles of the NetworkEpiCore scenarios: `scenario_ensemble(sc)` runs the scenario's `nsims` runs and
keeps, per run, what the summaries need (`ScenarioRun`: the observables on the time grid, aligned samples, the seeds,
infections, peak, crossing time and realised graph statistics); `scenario_graph(sc, r)` and `scenario_run(sc, r)`
regenerate the graph and the trajectory of one run.

Streams (design §E.2 as amended by §J.7): with b = `sc.sim.base_seed`, graph j is drawn with
`sample_graph(sc.network, N; rng = NetworkOutbreaks.stable_rng(b + j))` and run r is simulated with
`seed = b + 2^32 + r` (i.e. `stable_rng(b + 2^32 + r)`), arithmetic mod 2^64; run r uses graph j = r (`graphs =
:per_run`, a fresh graph per run), j = 1 (`:fixed`) or j = mod1(r, G) (`(:pool, G)`). These are exactly the streams
of `simulate(sc)` (src/convenience.jl), so the runs of `scenario_ensemble(sc)` are the runs of `simulate(sc)`.
A dynamic network starts each run from graph j of its base and evolves it inside the sampler; a well-mixed scenario
with `algorithm = :mass_action` has no graph.

Each run is simulated with `keep = :events` (the event log gives the infections, hence `:cumulative`, the crossing
time and the major-outbreak rule) and reduced to a `ScenarioRun` at once, so the memory is that of the grid samples,
not of the trajectories. Runs are independent, so `parallel = true` (threads) gives the same ensemble.
=#

export ScenarioEnsemble, scenario_ensemble, scenario_graph, scenario_run

# Missing aligned sample (outside the run's observation window).
const _MISSING_SAMPLE = Int32(-1)

"""
    NetworkOutbreaks.ScenarioRun

What [`scenario_ensemble`](@ref) keeps of one run r of a scenario on N nodes (counts are numbers of nodes):

- `run`, `graph`: the run index r and the index j of its graph (see [`scenario_graph`](@ref));
- `seeds`: nodes that start in an infected compartment; `new_infections`: infections during the run (entries into an
  infected compartment from a non-infected one), so `:cumulative`(t_end)·N = `seeds + new_infections`;
- `final_infected`: nodes ever infected (`final_size`·N); `infectious_end`: nodes in the scenario's infectious
  compartments at t_end; `absorbed`: no transition can fire in the final state;
- `peak_time`, `peak_count`: the first time at which prevalence (the `:infectious` observable) is largest, on the
  run's own clock, and that count;
- `crossing`: with a `CumulativeCrossing(ℓ)` scenario, the time of the infection at which cumulative incidence
  excluding the seeds first reaches ℓ (`NaN` if it never does, or without alignment);
- `grid`: the counts of the scenario's observables (rows, in `sc.observables` order) at the grid times `sc.tgrid`
  (right-continuous);
- `aligned`: for a crossing run of an aligned scenario, the counts at the times `crossing + j·Δt`,
  j = −(n−1), …, n−1 (n grid points of step Δt), with −1 outside the observation window (see
  `validation/conditioning.jl`); empty otherwise;
- `realised`: realised statistics of the run's graph, in the order of the ensemble's `realised_names`;
- `histogram`: the reinfection histogram (`reinfection_histogram`) for models with arrows back into the susceptible
  class (SIS, SIRS); empty otherwise.
"""
struct ScenarioRun
    run::Int
    graph::Int
    seeds::Int
    new_infections::Int
    final_infected::Int
    infectious_end::Int
    absorbed::Bool
    peak_time::Float64
    peak_count::Int
    crossing::Float64
    grid::Matrix{Int32}
    aligned::Matrix{Int32}
    realised::Vector{Float64}
    histogram::Vector{Int}
end

"""
    ScenarioEnsemble

The reference ensemble of a scenario, returned by [`scenario_ensemble`](@ref) and summarised by
[`summarise`](@ref): `scenario`, its `hash` (`scenario_hash(scenario)`), `realised_names` (the realised graph
statistics recorded per run: `:mean_degree`, `:excess_degree`, `:clustering` (global transitivity) and
`:erased_fraction` where they apply) and `runs::Vector{ScenarioRun}` in run order. `length(ens)` is the number of
runs and `ens[r]` run r.
"""
struct ScenarioEnsemble
    scenario::Scenario
    hash::String
    realised_names::Vector{Symbol}
    runs::Vector{ScenarioRun}
end

Base.length(e::ScenarioEnsemble) = length(e.runs)
Base.getindex(e::ScenarioEnsemble, r::Integer) = e.runs[r]
Base.iterate(e::ScenarioEnsemble, args...) = iterate(e.runs, args...)
Base.show(io::IO, e::ScenarioEnsemble) =
    print(io, "ScenarioEnsemble(:", e.scenario.id, ", ", first(e.hash, 8), ", N = ", e.scenario.sim.N, ", ",
          length(e.runs), " runs)")

# ---------------------------------------------------------------------------------------------------------------
# The per-scenario plan
# ---------------------------------------------------------------------------------------------------------------

struct _RunPlan
    model::OutbreakModel
    algorithm::OutbreakAlgorithm
    N::Int
    t0::Float64
    t1::Float64
    tgrid::Vector{Float64}
    rel::Vector{Float64}              # alignment offsets j·Δt, j = −(n−1):(n−1); empty without alignment
    crossing_count::Int               # m (0 without alignment)
    rows::Vector{Vector{Int}}         # per observable: the compartments summed (empty for :cumulative)
    cumulative::BitVector             # per observable: is it :cumulative?
    infectious_idx::Vector{Int}
    infected::BitVector               # per compartment (`_infected_mask`)
    enters::BitVector                 # per transition: non-infected → infected
    spont_out::Vector{Float64}        # per compartment: total rate of its spontaneous transitions
    contacts::Vector{Tuple{Int, Vector{Int}}}   # (recipient, catalysts) of the contacts with a positive rate
    realised_names::Vector{Symbol}
    histogram::Bool
end

function _run_plan(sc::Scenario)
    om = OutbreakModel(sc.model, sc.params; network = sc.network)
    idx = om.index_of
    infectious = Int[idx[X] for X in infectious_species(sc.model)]
    rows = Vector{Int}[]
    cumul = falses(length(sc.observables))
    for (k, X) in enumerate(sc.observables)
        if X === :cumulative
            push!(rows, Int[])
            cumul[k] = true
        elseif X === :infectious
            push!(rows, infectious)
        else
            haskey(idx, X) || throw(ArgumentError(
                "scenario_ensemble(:$(sc.id)): the observable $(X) is not a compartment of the simulated model"))
            push!(rows, [idx[X]])
        end
    end
    infected = BitVector(_infected_mask(om))
    enters = BitVector([!infected[idx[tr.from]] && infected[idx[tr.to]] for tr in om.transitions])
    C = length(om.compartments)
    spont = zeros(C)
    cts = Tuple{Int, Vector{Int}}[]
    for tr in om.transitions
        tr.rate > 0 || continue
        if tr.type === :spontaneous
            spont[idx[tr.from]] += tr.rate
        else
            push!(cts, (idx[tr.from], Int[idx[c] for c in _effective_via(tr, om)]))
        end
    end
    tgrid = collect(Float64, sc.tgrid)
    n = length(tgrid)
    m = _crossing_count(sc.sim.align, sc.sim.N)
    Δ = Float64(step(sc.tgrid))
    rel = m == 0 ? Float64[] : Float64[j * Δ for j in -(n - 1):(n - 1)]
    return _RunPlan(om, _algorithm(sc.sim.algorithm), sc.sim.N, sc.tspan[1], sc.tspan[2], tgrid, rel, m, rows,
                    cumul, infectious, infected, enters, spont, cts, _realised_names(sc),
                    typing(sc.model).theory !== :T_EB)
end

# The realised graph statistics recorded for a scenario's network (sorted, the order of the runs file).
function _realised_names(sc::Scenario)
    net = sc.network
    if net isa WellMixed
        return sc.sim.algorithm === :mass_action ? Symbol[] : [:erased_fraction, :excess_degree, :mean_degree]
    end
    net isa MultiplexNetwork && return [:erased_fraction, :excess_degree, :mean_degree]
    net isa DynamicNetwork && return [:clustering, :excess_degree, :mean_degree]
    # fleeting contacts have stub counts but no graph (sample_graph returns FleetingContacts): no clustering
    net isa MFSHNetwork && return [:excess_degree, :mean_degree]
    return [:clustering, :erased_fraction, :excess_degree, :mean_degree]
end

# ---------------------------------------------------------------------------------------------------------------
# Graphs and streams
# ---------------------------------------------------------------------------------------------------------------

# The graph index of run r.
function _graph_index(sc::Scenario, r::Integer)
    sc.network isa WellMixed && return 1
    g = sc.sim.graphs
    g === :per_run && return Int(r)
    g === :fixed && return 1
    return mod1(Int(r), last(g))
end

# The contact network of graph j, as `simulate(sc)` builds it, and its GraphInfo (or `nothing`).
function _scenario_network(sc::Scenario, j::Integer)
    net = sc.network
    N = sc.sim.N
    if net isa WellMixed
        sc.sim.algorithm === :mass_action && return SampledNetwork(net, N), nothing
        return sample_graph(net, N)               # K_N (no random draws), as simulate(model, ::WellMixed)
    end
    rng = stable_rng(_graph_seed(sc.sim.base_seed, j))
    net isa DynamicNetwork && return _initial_network(net, N, rng), nothing
    return sample_graph(net, N; rng)
end

_underlying_graph(g::TypedGraph) = g.graph
_underlying_graph(g::AbstractGraph) = g
_underlying_graph(g::DynamicGraph) = g.graph
_underlying_graph(g::StaticNetwork) = g.graph

function _degree_stats(g::AbstractGraph)
    ds = degree(g)
    s = sum(ds; init = 0)
    return (mean = nv(g) == 0 ? 0.0 : s / nv(g), excess = s == 0 ? 0.0 : sum(k -> k * (k - 1), ds; init = 0) / s)
end

# The realised statistics of a run's network, in the order of `names`.
function _realised_values(names::Vector{Symbol}, network, info)
    isempty(names) && return Float64[]
    vals = Dict{Symbol, Float64}()
    if info !== nothing
        vals[:mean_degree] = info.mean_degree
        vals[:excess_degree] = info.excess_degree
        vals[:erased_fraction] = info.erased_fraction
    end
    if :clustering in names || info === nothing
        g = _underlying_graph(network)
        if info === nothing
            st = _degree_stats(g)
            vals[:mean_degree] = st.mean
            vals[:excess_degree] = st.excess
        end
        :clustering in names && (vals[:clustering] = Float64(global_clustering_coefficient(g)))
    end
    return Float64[vals[k] for k in names]
end

# The shared networks of a scenario whose runs do not each draw their own (:fixed, (:pool, G), well mixed).
function _shared_networks(sc::Scenario, plan::_RunPlan)
    (sc.network isa WellMixed || sc.sim.graphs !== :per_run) || return nothing
    G = sc.network isa WellMixed || sc.sim.graphs === :fixed ? 1 : last(sc.sim.graphs)
    out = Vector{Tuple{Any, Vector{Float64}}}(undef, G)
    for j in 1:G
        net, info = _scenario_network(sc, j)
        out[j] = (net, _realised_values(plan.realised_names, net, info))
    end
    return out
end

function _check_run_index(sc::Scenario, r::Integer, where)
    1 <= r <= sc.sim.nsims || throw(ArgumentError(
        "$(where)(:$(sc.id), $(r)): the scenario has runs 1:$(sc.sim.nsims)"))
    return Int(r)
end

"""
    scenario_graph(sc::Scenario, r::Integer) -> graph
    scenario_graph(sc::Scenario) -> graph

The exact contact graph of run `r` of the reference ensemble of `sc` (design §E.4): graph j of the streams of
[`scenario_ensemble`](@ref), with j = r for `graphs = :per_run`, j = 1 for `:fixed` and j = mod1(r, G) for
`(:pool, G)`, i.e. `first(sample_graph(sc.network, sc.sim.N; rng = NetworkOutbreaks.stable_rng(sc.sim.base_seed + j)))`.
It is a `Graphs.SimpleGraph`, a [`TypedGraph`](@ref) (multitype networks), a [`MultiplexGraph`](@ref) (multiplex
networks) or, for a dynamic network, the graph the run **starts** from (the run then rewires its own copy). So the
individual- and pair-based models of NodeBasedModels can run on the same graph as the simulation.

`scenario_graph(sc)` is the graph of every run of a quenched scenario (`graphs = :fixed`, e.g. `:sir_reg6_fixed`).
A well-mixed scenario simulated with `MassActionSSA` has no graph (an `ArgumentError`).
"""
function scenario_graph(sc::Scenario, r::Integer)
    r = _check_run_index(sc, r, "scenario_graph")
    (sc.network isa WellMixed && sc.sim.algorithm === :mass_action) && throw(ArgumentError(
        "scenario_graph(:$(sc.id)): the scenario is well mixed and simulated with MassActionSSA, which has no " *
        "contact graph"))
    net, _ = _scenario_network(sc, _graph_index(sc, r))
    return net isa DynamicGraph ? net.graph : net
end

function scenario_graph(sc::Scenario)
    (sc.sim.graphs === :fixed || sc.network isa WellMixed) || throw(ArgumentError(
        "scenario_graph(:$(sc.id)): graphs = $(repr(sc.sim.graphs)), so the runs do not share one graph; pass the " *
        "run index, scenario_graph(sc, r)"))
    return scenario_graph(sc, 1)
end

"""
    scenario_run(sc::Scenario, r::Integer; keep = :events) -> OutbreakTrajectory

Regenerate run `r` of the reference ensemble of `sc` alone, at full resolution: the simulation of the scenario's model
on [`scenario_graph`](@ref)`(sc, r)` (or its dynamic network, or the well-mixed population) with
`seed = sc.sim.base_seed + 2^32 + r` and the scenario's algorithm. It is the trajectory that `scenario_ensemble(sc)`
reduced to `ens[r]`. `keep` is `:events` (the event log too) or `:counts`.
"""
function scenario_run(sc::Scenario, r::Integer; keep::Symbol = :events)
    r = _check_run_index(sc, r, "scenario_run")
    plan = _run_plan(sc)
    net, _ = _scenario_network(sc, _graph_index(sc, r))
    return _simulate_run(sc, plan, net, r; keep)
end

_simulate_run(sc::Scenario, plan::_RunPlan, net, r::Integer; keep::Symbol = :events) =
    simulate(OutbreakSpec(plan.model, net, sc.initial, sc.tspan); algorithm = plan.algorithm,
             seed = _run_seed(sc.sim.base_seed, r), keep)

# ---------------------------------------------------------------------------------------------------------------
# scenario_ensemble
# ---------------------------------------------------------------------------------------------------------------

"""
    scenario_ensemble(sc::Scenario; parallel = false) -> ScenarioEnsemble
    scenario_ensemble(id::Symbol; kw...)

Run the reference ensemble of a NetworkEpiCore scenario (design §E.1, §E.2): `sc.sim.nsims` runs of `sc.model` with
`sc.params`, seeded by `sc.initial` (exactly ρ_X·N nodes per compartment, disjoint and uniform; on a multitype network
each stratum's seeds on its own node type, design §J.6), on `sc.sim.N` nodes of `sc.network`, over `sc.tspan`, with
`sc.sim.algorithm` (`:next_reaction` → `NextReaction`, `:direct` → `DirectSSA`, `:composition_rejection`, `:has`,
`:mass_action` → `MassActionSSA`).

Graphs and random streams are those of `simulate(sc)` (design §J.7): graph j from
`sample_graph(sc.network, N; rng = NetworkOutbreaks.stable_rng(b + j))` and run r from `stable_rng(b + 2^32 + r)`,
b = `sc.sim.base_seed`, with a fresh graph per run by default (`graphs = :per_run`, iid runs). Each run is reduced at
once to a [`NetworkOutbreaks.ScenarioRun`](@ref); conditioning and alignment are applied by [`summarise`](@ref).
With `parallel = true` runs are distributed over threads; the ensemble is the same.

`scenario_ensemble(sc, trajectories)` builds an ensemble from given trajectories instead (see its docstring).
"""
function scenario_ensemble(sc::Scenario; parallel::Bool = false)
    plan = _run_plan(sc)
    shared = _shared_networks(sc, plan)
    function one(r)
        j = _graph_index(sc, r)
        net, realised = if shared === nothing
            n_, info = _scenario_network(sc, j)
            (n_, _realised_values(plan.realised_names, n_, info))
        else
            shared[j]
        end
        traj = _simulate_run(sc, plan, net, r)
        return _run_record(plan, traj, r, j, realised)
    end
    runs = _map_runs(one, sc.sim.nsims, parallel)
    return ScenarioEnsemble(sc, scenario_hash(sc), plan.realised_names, runs)
end
scenario_ensemble(id::Symbol; kw...) = scenario_ensemble(scenario(id); kw...)

"""
    scenario_ensemble(sc::Scenario, trajectories::AbstractVector{OutbreakTrajectory};
                      graphs = 1:length(trajectories)) -> ScenarioEnsemble

An ensemble of `sc` made from given trajectories (run r is `trajectories[r]`, on graph `graphs[r]`, which only the
between-graph standard error of `(:pool, G)` scenarios uses), for conditioning and alignment on constructed or
externally simulated runs. The trajectories must be of the scenario's simulated model
(`OutbreakModel(sc.model, sc.params; network = sc.network)`), span `sc.tspan` and carry their event log (simulated
with `keep = :events`). No realised graph statistics are recorded. The number of trajectories may differ from
`sc.sim.nsims`.
"""
function scenario_ensemble(sc::Scenario, trajectories::AbstractVector{OutbreakTrajectory};
                           graphs::AbstractVector{<:Integer} = 1:length(trajectories))
    isempty(trajectories) && throw(ArgumentError("scenario_ensemble(:$(sc.id)): no trajectories"))
    length(graphs) == length(trajectories) || throw(ArgumentError(
        "scenario_ensemble(:$(sc.id)): $(length(graphs)) graph indices for $(length(trajectories)) trajectories"))
    plan = _run_plan(sc)
    plan = _RunPlan(plan.model, plan.algorithm, plan.N, plan.t0, plan.t1, plan.tgrid, plan.rel, plan.crossing_count,
                    plan.rows, plan.cumulative, plan.infectious_idx, plan.infected, plan.enters, plan.spont_out,
                    plan.contacts, Symbol[], plan.histogram)
    runs = ScenarioRun[]
    for (r, traj) in enumerate(trajectories)
        traj.model.compartments == plan.model.compartments || throw(ArgumentError(
            "scenario_ensemble(:$(sc.id)): trajectory $(r) has compartments $(traj.model.compartments), not those " *
            "of the scenario's model $(plan.model.compartments)"))
        length(traj.model.transitions) == length(plan.model.transitions) || throw(ArgumentError(
            "scenario_ensemble(:$(sc.id)): trajectory $(r) is of a model with other transitions"))
        size(traj.counts, 2) == length(traj.times) && sum(view(traj.counts, :, 1)) == plan.N || throw(ArgumentError(
            "scenario_ensemble(:$(sc.id)): trajectory $(r) is not a run on N = $(plan.N) nodes"))
        (traj.times[1] == plan.t0 && traj.times[end] == plan.t1) || throw(ArgumentError(
            "scenario_ensemble(:$(sc.id)): trajectory $(r) spans [$(traj.times[1]), $(traj.times[end])], not the " *
            "scenario's tspan $(sc.tspan)"))
        (isempty(traj.events) && length(traj.times) > 2) && throw(ArgumentError(
            "scenario_ensemble(:$(sc.id)): trajectory $(r) has no event log; simulate with keep = :events"))
        push!(runs, _run_record(plan, traj, r, Int(graphs[r]), Float64[]))
    end
    return ScenarioEnsemble(sc, scenario_hash(sc), Symbol[], runs)
end

function _map_runs(f, n::Int, parallel::Bool)
    parallel || return ScenarioRun[f(r) for r in 1:n]
    tasks = [Threads.@spawn f(r) for r in 1:n]
    out = Vector{ScenarioRun}(undef, n)
    err = nothing
    for r in 1:n
        try
            out[r] = fetch(tasks[r])
        catch e
            err === nothing && (err = e isa TaskFailedException ? tasks[r].exception : e)
        end
    end
    err === nothing || throw(err)
    return out
end

# ---------------------------------------------------------------------------------------------------------------
# One run → ScenarioRun
# ---------------------------------------------------------------------------------------------------------------

# Can any transition fire in the state `counts`? (A contact is taken to be possible whenever its recipient and one of
# its catalysts are occupied, whatever the graph, so an "absorbed" run is certainly absorbed.)
function _absorbed(plan::_RunPlan, counts::AbstractVector{<:Integer})
    for c in eachindex(counts)
        counts[c] > 0 && plan.spont_out[c] > 0 && return false
    end
    for (s, via) in plan.contacts
        counts[s] > 0 && any(v -> counts[v] > 0, via) && return false
    end
    return true
end

function _run_record(plan::_RunPlan, traj::OutbreakTrajectory, r::Int, j::Int, realised::Vector{Float64})
    counts = traj.counts
    times = traj.times
    C = size(counts, 1)
    seeds = 0
    @inbounds for c in 1:C
        plan.infected[c] && (seeds += counts[c, 1])
    end
    inf_times = Float64[e.time for e in traj.events if plan.enters[e.transition_index]]
    new = length(inf_times)
    final_infected = count(>(0), traj.final_infection_counts)
    last_col = @view counts[:, end]
    inf_end = sum(c -> last_col[c], plan.infectious_idx; init = 0)
    absorbed = _absorbed(plan, last_col)
    # prevalence peak (first maximum), on the run's own clock
    peak_t, peak_v = times[1], sum(c -> counts[c, 1], plan.infectious_idx; init = 0)
    @inbounds for k in 2:length(times)
        v = 0
        for c in plan.infectious_idx
            v += counts[c, k]
        end
        if v > peak_v
            peak_v, peak_t = v, times[k]
        end
    end
    grid = _samples(plan, times, counts, inf_times, seeds, plan.tgrid, false, absorbed)
    crossing = NaN
    aligned = Matrix{Int32}(undef, 0, 0)
    if plan.crossing_count > 0 && new >= plan.crossing_count
        crossing = inf_times[plan.crossing_count]
        aligned = _samples(plan, times, counts, inf_times, seeds, crossing .+ plan.rel, true, absorbed)
    end
    hist = plan.histogram ? reinfection_histogram(traj) : Int[]
    return ScenarioRun(r, j, seeds, new, final_infected, inf_end, absorbed, peak_t, peak_v, crossing, grid, aligned,
                       realised, hist)
end

# The observables at the sorted times `q` (right-continuous). With `masked`, a time before t0, or after t_end while
# the run is not absorbed, gives the missing value.
function _samples(plan::_RunPlan, times::Vector{Float64}, counts::Matrix{Int}, inf_times::Vector{Float64},
                  seeds::Int, q::AbstractVector{Float64}, masked::Bool, absorbed::Bool)
    nobs = length(plan.rows)
    out = Matrix{Int32}(undef, nobs, length(q))
    @inbounds for (k, t) in pairs(q)
        if masked && (t < plan.t0 || (t > plan.t1 && !absorbed))
            out[:, k] .= _MISSING_SAMPLE
            continue
        end
        col = t < times[1] ? 1 : searchsortedlast(times, t)
        for o in 1:nobs
            if plan.cumulative[o]
                out[o, k] = Int32(seeds + searchsortedlast(inf_times, t))
            else
                v = 0
                for c in plan.rows[o]
                    v += counts[c, col]
                end
                out[o, k] = Int32(v)
            end
        end
    end
    return out
end
