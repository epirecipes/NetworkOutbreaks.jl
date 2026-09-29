#=
convenience.jl (owner: WP16)

The high-level API of design §A.5:

    simulate(model, net::NetworkDescriptor; N, p, initial, tspan, nsims, graphs, algorithm, seed, …) -> OutbreakEnsemble
    simulate(model, g::Graphs.AbstractGraph; p, initial, tspan, nsims, …)                            # fixed graph
    simulate(sc::Scenario; …)                                                                         # uses sc.sim

`model` is anything `OutbreakModel(model, p)` accepts (a ContactModel, a model `contact_model` converts, or an
OutbreakModel). Random numbers follow design §J.7: with base seed b, graph r is drawn with `stable_rng(b + r)` and
SSA run r with `stable_rng(b + 2^32 + r)` (arithmetic mod 2^64), so a scenario's runs and graphs can be regenerated
one by one.

It also declares the generic `sample_graph(desc, N; rng)`, whose methods for each descriptor live in
src/generators/*.jl (WP25, WP36a) and src/processes/*.jl (the dynamic and fleeting-contact descriptors); the fallback
here is the error for a descriptor without a sampler.
=#

# ---------------------------------------------------------------------------------------------------------------
# Graph sampling
# ---------------------------------------------------------------------------------------------------------------

"""
    sample_graph(net::NetworkDescriptor, N::Integer; rng::AbstractRNG = Random.default_rng()) -> (graph, info::GraphInfo)

Draw one contact network with `N` nodes from the NetworkEpiCore descriptor `net`, using `rng`; `info` is a
[`GraphInfo`](@ref) describing the realised graph (its `N`, `edges`, `mean_degree`, `excess_degree`, the fraction of
stub pairs erased as self-loops or multiple edges `erased_fraction`, the construction `method`, and per-descriptor
details). Reproducible for a given `rng` state (use `NetworkOutbreaks.stable_rng(s)`; `simulate(model, net; N, seed =
b)` draws graph r from `stable_rng(b + r)`, design §J.7).

Methods (design §C.3):
- `ConfigurationNetwork`: `random_regular_graph(N, k)` for `RegularDegree(k)`, the Erdős–Rényi graph for
  `PoissonDegree`, and the erased configuration model otherwise;
- `ExplicitGraph`: the graph itself (N must equal its number of nodes);
- `MultitypeNetwork`: a [`TypedGraph`](@ref) (node types recorded, typed seeding of design §J.6);
- `ClusteredNetwork`: the Newman–Miller single-edge and triangle construction;
- `MultiplexNetwork`: independent layer samples as a [`MultiplexGraph`](@ref);
- `WellMixed(κ)`: the complete graph as a one-layer `MultiplexGraph` with layer rate κ/(N − 1) (the lumping test,
  N ≤ 5000; simulate well-mixed populations with [`MassActionSSA`](@ref));
- `DegreeCorrelatedNetwork`: the joint-degree (2K) construction;
- `DynamicNetwork`: the initial graph, a sample of its base network;
- `MFSHNetwork`: the stub counts of a run of fleeting contacts ([`FleetingContacts`](@ref)), not a graph.

A descriptor without a sampler is an `ArgumentError`.
"""
function sample_graph end

sample_graph(net::NetworkDescriptor, N::Integer; rng::AbstractRNG = Random.default_rng()) = throw(ArgumentError(
    "sample_graph: NetworkOutbreaks has no graph generator for $(nameof(typeof(net))) (design §C.3); pass an " *
    "explicit graph (a Graphs.AbstractGraph or an ExplicitGraph)"))

# ---------------------------------------------------------------------------------------------------------------
# Random-number streams (design §J.7)
# ---------------------------------------------------------------------------------------------------------------

const _RUN_OFFSET = UInt64(1) << 32

_graph_seed(base::UInt64, r::Integer) = base + UInt64(r)                 # mod 2^64
_run_seed(base::UInt64, r::Integer) = base + _RUN_OFFSET + UInt64(r)     # mod 2^64

function _base_seed(seed, rng)
    if rng !== nothing
        seed === nothing || throw(ArgumentError("pass either `seed` or `rng` to simulate, not both"))
        return rand(rng, UInt64)
    end
    return seed === nothing ? rand(UInt64) : _seed_u64(seed)
end

# ---------------------------------------------------------------------------------------------------------------
# simulate(model, network; …)
# ---------------------------------------------------------------------------------------------------------------

_outbreak_model(model::OutbreakModel, p, net) = OutbreakModel(model, p)
_outbreak_model(model, p, net) = OutbreakModel(model, p; network = net)

function _grid(tgrid, tspan, keep)
    keep in (:grid, :counts, :events) || throw(ArgumentError("keep must be :grid, :counts or :events; got $(repr(keep))"))
    keep === :grid || return nothing
    if tgrid === nothing
        isfinite(tspan[2]) || throw(ArgumentError("keep = :grid with an infinite time span needs an explicit `tgrid`"))
        return collect(range(Float64(tspan[1]), Float64(tspan[2]); length = 201))
    end
    grid = collect(Float64, tgrid)
    isempty(grid) && throw(ArgumentError("tgrid is empty"))
    issorted(grid) || throw(ArgumentError("tgrid must be sorted"))
    (first(grid) >= tspan[1] && last(grid) <= tspan[2]) ||
        throw(ArgumentError("tgrid must lie within tspan = $(tspan)"))
    return grid
end

# The trajectory resampled on `grid` (right-continuous, as `state_at`), without its event log.
function _on_grid(traj::OutbreakTrajectory, grid::Vector{Float64})
    C = size(traj.counts, 1)
    counts = Matrix{Int}(undef, C, length(grid))
    for (k, t) in pairs(grid)
        j = t < traj.times[1] ? 1 : searchsortedlast(traj.times, t)
        counts[:, k] .= @view traj.counts[:, j]
    end
    return OutbreakTrajectory(traj.model, copy(grid), counts, traj.final_infection_counts, OutbreakEvent[],
                              traj.seed, traj.algorithm)
end

function _one_run(om, graph, initial, tspan, algorithm, seed, keep, grid, interventions)
    spec = OutbreakSpec(om, graph, initial, tspan)
    traj = simulate(spec; algorithm, seed, keep = keep === :grid ? :counts : keep, interventions)
    return grid === nothing ? traj : _on_grid(traj, grid)
end

function _run_all(run, nsims::Int, parallel::Bool)
    trajs = Vector{OutbreakTrajectory}(undef, nsims)
    if parallel
        tasks = [Threads.@spawn run(r) for r in 1:nsims]
        # wait for every run, then raise the error of the first failed run as itself (an ArgumentError, say), as
        # the sequential loop does, rather than as a TaskFailedException
        foreach(_wait_quietly, tasks)
        for r in 1:nsims
            istaskfailed(tasks[r]) && throw(tasks[r].exception)
            trajs[r] = fetch(tasks[r])
        end
    else
        for r in 1:nsims
            trajs[r] = run(r)
        end
    end
    return trajs
end

function _wait_quietly(t::Task)
    try
        wait(t)
    catch err
        err isa TaskFailedException || rethrow()
    end
    return nothing
end

"""
    simulate(model, net::NetworkDescriptor; N, p = Dict(), initial, tspan, nsims = 1, graphs = :per_run,
             algorithm = NextReaction(), seed = nothing, rng = nothing, tgrid = nothing, keep = :grid,
             interventions = InterventionPlan(), parallel = false) -> OutbreakEnsemble

Simulate `nsims` runs of `model` on graphs of `N` nodes drawn from the network descriptor `net` (design §A.5).

- `model`: a `ContactModel` (e.g. `sir_model()`), any model `contact_model` accepts, or an `OutbreakModel`; it is
  converted by `OutbreakModel(model, p; network = net)`, so the rate convention uses the nominal
  `mean_degree(net)` and τ is the per-contact rate of every back end. `p` gives the parameter values.
- `initial`: a `SeedSpec` (e.g. `SeedFraction(:I => 0.01)`; see [`OutbreakSpec`](@ref) for the rounding and the
  background compartment). The keyword `seed` is the random seed, not the seeding.
- `graphs`: `:per_run` (a fresh graph for every run, so the runs are iid; the default), `:fixed` (one graph for every
  run) or `(:pool, G)` (G graphs used in turn).
- Random numbers (design §J.7): with base seed `b` (`seed`, or drawn from `rng`, or random), graph r comes from
  [`sample_graph`](@ref)`(net, N; rng = NetworkOutbreaks.stable_rng(b + r))` (graph 1 for `:fixed`, graph
  `mod1(r, G)` of the pool) and run r is `simulate(spec_r; seed = b + 2^32 + r)`, arithmetic mod 2^64. So
  `traj.seed` of run r is `b + 2^32 + r`, and the result does not depend on `parallel`.
- `keep = :grid` (the default) stores each run on `tgrid` (default: 201 points over `tspan`), right-continuously;
  `:counts` keeps every event time and `:events` also the event log. `final_size` always refers to the end of
  `tspan`.
- `algorithm` and `interventions` are passed to `simulate(spec; …)`.

The returned ensemble's `spec` has the converted model, the seeding, the time span and, as its network, a
[`SampledNetwork`](@ref) (or the fixed graph for `graphs = :fixed`).

```julia
ens = simulate(sir_model(), ConfigurationNetwork(PoissonDegree(5)); N = 10_000,
               p = Dict(:τ => 1/6, :γ => 1/4), initial = SeedFraction(:I => 0.01), tspan = (0.0, 60.0),
               nsims = 50, seed = 20260926)
mean(final_size(ens))            # ≈ 0.80, the edge-based R∞
```
"""
function simulate(model, net::NetworkDescriptor;
                  N::Integer,
                  p::AbstractDict = Dict{Symbol, Float64}(),
                  initial::SeedSpec,
                  tspan::Tuple{<:Real, <:Real},
                  nsims::Integer = 1,
                  graphs = :per_run,
                  algorithm::OutbreakAlgorithm = NextReaction(),
                  seed::Union{Nothing, Integer} = nothing,
                  rng::Union{Nothing, AbstractRNG} = nothing,
                  tgrid = nothing,
                  keep::Symbol = :grid,
                  interventions::InterventionPlan = InterventionPlan(),
                  parallel::Bool = false)
    N >= 1 || throw(ArgumentError("simulate: N must be ≥ 1; got $(N)"))
    nsims >= 1 || throw(ArgumentError("simulate: nsims must be ≥ 1; got $(nsims)"))
    mode = _check_graphs_mode(graphs; allow_fixed = true)
    ts = (Float64(tspan[1]), Float64(tspan[2]))
    om = _outbreak_model(model, p, net)
    grid = _grid(tgrid, ts, keep)
    base = _base_seed(seed, rng)
    draw(j) = first(sample_graph(net, N; rng = stable_rng(_graph_seed(base, j))))
    if mode === :fixed
        g = draw(1)
        trajs = _run_all(r -> _one_run(om, g, initial, ts, algorithm, _run_seed(base, r), keep, grid, interventions),
                         Int(nsims), parallel)
        return OutbreakEnsemble(OutbreakSpec(om, g, initial, ts), trajs)
    end
    pool = mode === :per_run ? nothing : [draw(j) for j in 1:last(mode)]
    graph_of(r) = pool === nothing ? draw(r) : pool[mod1(r, length(pool))]
    trajs = _run_all(r -> _one_run(om, graph_of(r), initial, ts, algorithm, _run_seed(base, r), keep, grid,
                                   interventions),
                     Int(nsims), parallel)
    return OutbreakEnsemble(OutbreakSpec(om, SampledNetwork(net, N, mode), initial, ts), trajs)
end

"""
    simulate(model, g::Graphs.AbstractGraph; N = nothing, p = Dict(), initial, tspan, nsims = 1,
             algorithm = NextReaction(), seed = nothing, rng = nothing, tgrid = nothing, keep = :grid,
             interventions = InterventionPlan(), parallel = false) -> OutbreakEnsemble
    simulate(model, net::ExplicitGraph; kw...)

Simulate `nsims` runs of `model` on the fixed graph `g` (quenched). The model is converted with
`OutbreakModel(model, p; network = g)` (a rate convention then uses the realised mean degree of `g`), and run r is
`simulate(spec; seed = b + 2^32 + r)` for the base seed `b` (design §J.7). The other keywords are those of
`simulate(model, net::NetworkDescriptor; …)` (without `graphs`). The number of nodes is `nv(g)`: `N` may be omitted,
and if it is given it must equal `nv(g)`.
"""
function simulate(model, g::AbstractGraph;
                  N::Union{Nothing, Integer} = nothing,
                  p::AbstractDict = Dict{Symbol, Float64}(),
                  initial::SeedSpec,
                  tspan::Tuple{<:Real, <:Real},
                  nsims::Integer = 1,
                  algorithm::OutbreakAlgorithm = NextReaction(),
                  seed::Union{Nothing, Integer} = nothing,
                  rng::Union{Nothing, AbstractRNG} = nothing,
                  tgrid = nothing,
                  keep::Symbol = :grid,
                  interventions::InterventionPlan = InterventionPlan(),
                  parallel::Bool = false)
    N === nothing || N == nv(g) || throw(ArgumentError("simulate: the graph has $(nv(g)) nodes, not N = $(N)"))
    nsims >= 1 || throw(ArgumentError("simulate: nsims must be ≥ 1; got $(nsims)"))
    ts = (Float64(tspan[1]), Float64(tspan[2]))
    om = _outbreak_model(model, p, g)
    grid = _grid(tgrid, ts, keep)
    base = _base_seed(seed, rng)
    trajs = _run_all(r -> _one_run(om, g, initial, ts, algorithm, _run_seed(base, r), keep, grid, interventions),
                     Int(nsims), parallel)
    return OutbreakEnsemble(OutbreakSpec(om, g, initial, ts), trajs)
end

function simulate(model, net::ExplicitGraph; kw...)
    g = net.graph
    g isa AbstractGraph || throw(ArgumentError("simulate: the ExplicitGraph does not hold a Graphs.AbstractGraph"))
    return simulate(model, g; kw...)
end

# SimConfig.algorithm symbols (NetworkEpiCore SIM_ALGORITHMS) → samplers. `:fleeting` is the count-level sampler of
# `MFSHNetwork` (design §K WP36c).
function _algorithm(sym::Symbol)
    sym === :next_reaction && return NextReaction()
    sym === :direct && return DirectSSA()
    sym === :composition_rejection && return CompositionRejection()
    sym === :has && return HAS()
    sym === :mass_action && return MassActionSSA()
    sym === :fleeting && return FleetingContactSSA()
    known = union(SIM_ALGORITHMS, (:fleeting,))
    throw(ArgumentError("unknown simulation algorithm :$(sym); expected one of $(Tuple(known))"))
end

"""
    simulate(sc::Scenario; nsims = sc.sim.nsims, seed = sc.sim.base_seed, algorithm, keep = :grid,
             parallel = false) -> OutbreakEnsemble

The reference ensemble of a NetworkEpiCore `Scenario`: `simulate(sc.model, sc.network; N = sc.sim.N,
p = sc.params, initial = sc.initial, tspan = sc.tspan, tgrid = sc.tgrid, graphs = sc.sim.graphs, …)` with the
scenario's algorithm (`sc.sim.algorithm`) and base seed, so run r and its graph use the streams of design §J.7.
Conditioning and alignment (`sc.sim.condition`, `sc.sim.align`) are applied when the ensemble is summarised, not
here. Change other settings with `derive(sc; …)`, which gives a scenario with its own hash.
"""
function simulate(sc::Scenario;
                  nsims::Integer = sc.sim.nsims,
                  seed::Integer = sc.sim.base_seed,
                  algorithm::OutbreakAlgorithm = _algorithm(sc.sim.algorithm),
                  keep::Symbol = :grid,
                  parallel::Bool = false)
    return simulate(sc.model, sc.network; N = sc.sim.N, p = sc.params, initial = sc.initial, tspan = sc.tspan,
                    nsims, graphs = sc.sim.graphs, algorithm, seed, tgrid = sc.tgrid, keep, parallel)
end

# ---------------------------------------------------------------------------------------------------------------
# NetworkEpiCore observables on trajectories
# ---------------------------------------------------------------------------------------------------------------

"""
    compartment(traj::OutbreakTrajectory, X::Symbol) -> Vector{Float64}

The fraction of nodes in compartment `X` at each snapshot time `times(traj)` (the NetworkEpiCore generic, with the
`(result, X)` argument order of result objects).
"""
function compartment(traj::OutbreakTrajectory, X::Symbol)
    n = length(traj.final_infection_counts)
    return compartment_series(traj, X) ./ n
end

"""
    compartments(traj::OutbreakTrajectory, Xs::AbstractVector{Symbol}) -> Dict{Symbol,Vector{Float64}}

[`compartment`](@ref) for several compartments at once.
"""
compartments(traj::OutbreakTrajectory, Xs::AbstractVector{Symbol}) =
    Dict{Symbol, Vector{Float64}}(X => compartment(traj, X) for X in Xs)
