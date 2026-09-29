# Owner: WP26 (DESIGN_NetworkEpiCore.md §G.2 WP26, §C.3).
#=
processes/common.jl

Graph processes: contact networks whose edges change during a run, driven by a stochastic process that the samplers
run alongside the epidemic ("swap events inside NextReaction/HAS, drawn with the trajectory RNG", design §C.3).

- `GraphProcess`: the supertype of the processes (`NeighbourExchangeProcess` in neighbour_exchange.jl; the stretch
  dormant-contact process in dormant.jl).
- `DynamicGraph(graph, process)`: the contact network (an `AbstractContactNetwork`) that starts as `graph` and
  evolves by `process`. Every run works on its own copy of the graph, so a spec can be simulated repeatedly and the
  runs of an ensemble are independent.
- `evolve_graph!(g, process, Δt; rng)`: the process alone, without an epidemic (for inspection and tests).
- `simulate(model, net::DynamicNetwork; N, …)`: the route from a NetworkEpiCore descriptor (fresh initial graph per
  run, design §J.7 streams).

The per-run protocol that a `GraphProcess` implements (internal; a new process adds methods for its own types):

    _process_state(p, g::SimpleGraph, rng) -> ps          # per-run state on the run's own copy `g` of the graph
    _process_rate(ps) -> Float64                           # total rate of process events in the current state
    _process_fire!(ps, g, node_state, rng) -> touched      # one event: mutate `g`, return the nodes whose
                                                           # neighbourhoods changed (a reused vector)
    _process_summary(ps) -> NamedTuple                     # event counts, for `evolve_graph!`

The samplers re-read `_process_rate` after every event (epidemic, process or intervention), so the process clock
stays exact if the rate changes (NextReaction rescales it with the Gibson–Bruck identity, HAS redraws its waiting
time from the total rate); for neighbour exchange it is the constant ηE/2. The contact hazards of the touched nodes
are refreshed after each process event, and only theirs: a process event changes no compartment.

Samplers: `NextReaction` and `HAS` run graph processes. `DirectSSA` and `CompositionRejection` would silently treat a
`DynamicGraph` as its initial, static graph, so `_outbreak_graph(::DynamicGraph)` (the graph those samplers read)
throws instead.
=#

export GraphProcess, DynamicGraph, evolve_graph!

"""
    GraphProcess

Supertype of the stochastic processes that rewire a contact graph during a run of `NextReaction` or `HAS`:
[`NeighbourExchangeProcess`](@ref). A process is attached to a graph with [`DynamicGraph`](@ref); run it alone with
[`evolve_graph!`](@ref).
"""
abstract type GraphProcess end

"""
    DynamicGraph(graph, process::GraphProcess)
    DynamicGraph(graph, process::NetworkEpiCore.NetworkProcess)

A contact network that starts as `graph` (an undirected `Graphs.AbstractGraph` without self-loops) and whose edges
change during a run by `process`, e.g. `DynamicGraph(g, NeighbourExchangeProcess(η))`. A NetworkEpiCore process
(`NeighbourExchange(η)`) is converted to the NetworkOutbreaks one.

Each run copies `graph` (as a `SimpleGraph{Int}`) and evolves the copy with the run's own random-number generator,
so `simulate(spec; seed)` is reproducible and `graph` is never mutated. The process events do not appear in the
trajectory (they change no compartment); the epidemic sees the graph as it is at each event.

Supported by `NextReaction` and `HAS` (with interventions); `DirectSSA` and `CompositionRejection` raise an
`ArgumentError`. `simulate(model, DynamicNetwork(base, NeighbourExchange(η)); N, …)` builds one per run from a fresh
sample of `base`.
"""
struct DynamicGraph{G <: AbstractGraph, P <: GraphProcess} <: AbstractContactNetwork
    graph::G
    process::P
    function DynamicGraph{G, P}(graph::G, process::P) where {G <: AbstractGraph, P <: GraphProcess}
        _check_contact_graph(graph)
        return new{G, P}(graph, process)
    end
end
DynamicGraph(graph::G, process::P) where {G <: AbstractGraph, P <: GraphProcess} =
    DynamicGraph{G, P}(graph, process)
DynamicGraph(graph::AbstractGraph, process::NetworkProcess) = DynamicGraph(graph, _graph_process(process))

# The NetworkOutbreaks process of a NetworkEpiCore process (methods per process type; neighbour_exchange.jl).
_graph_process(p::NetworkProcess) = throw(ArgumentError(
    "NetworkOutbreaks has no graph process for $(nameof(typeof(p))) yet (design §C.3, §K WP36); the neighbour " *
    "exchange process is NeighbourExchange(η)"))

# DirectSSA and CompositionRejection read the graph of a network through `_outbreak_graph`; they do not run graph
# processes, and treating a DynamicGraph as its initial graph would silently simulate a static network.
_outbreak_graph(net::DynamicGraph) = throw(ArgumentError(
    "a DynamicGraph (the graph process $(nameof(typeof(net.process)))) runs only under NextReaction or HAS; " *
    "DirectSSA and CompositionRejection do not simulate graph processes"))

Base.show(io::IO, net::DynamicGraph) =
    print(io, "DynamicGraph(", nv(net.graph), " nodes, ", ne(net.graph), " edges, ", net.process, ")")

# A rate convention takes the mean degree of the initial graph (see `OutbreakModel(cm, p; network)`).
_rate_network(net::DynamicGraph) = ExplicitGraph(net.graph)

# ---------------------------------------------------------------------------------------------------------------
# The per-run protocol (defaults)
# ---------------------------------------------------------------------------------------------------------------

_process_summary(ps) = NamedTuple()

# The run's own mutable copy of the initial graph, and the process state on it.
_run_graph(g::SimpleGraph{Int}) = copy(g)
function _run_graph(g::AbstractGraph)
    h = SimpleGraph{Int}(nv(g))
    for e in edges(g)
        add_edge!(h, Int(src(e)), Int(dst(e)))
    end
    return h
end

function _prepare_dynamic(net::DynamicGraph, rng::AbstractRNG)
    g = _run_graph(net.graph)
    return g, _process_state(net.process, g, rng)
end

# The samplers call these for every run; `nothing` is "no graph process" (static, time-varying and multiplex
# networks), for which they compile to nothing.
_proc_rate(::Nothing) = 0.0
_proc_rate(ps) = _process_rate(ps)

# True when the epidemic state can never change again, whatever the graph does: no scheduled intervention pending
# (threshold interventions fire only on state changes), no spontaneous transition with a positive rate out of an
# occupied compartment, and no contact transition with a positive rate that has a node in its recipient compartment
# and a catalyst node other than that recipient (a node is not its own contact), so every contact hazard stays 0 on
# any graph. (An SI run after everyone is infected stops here, although catalysts remain.)
function _epidemic_absorbing(rm::_RunModel, counts::Vector{Int}, sched::_Schedule)
    _interventions_pending(sched) && return false
    @inbounds for c in 1:rm.C
        counts[c] > 0 && rm.spont_total[c] > 0 && return false
    end
    @inbounds for s in 1:rm.C
        counts[s] > 0 || continue
        for j in rm.contact_by_src[s]
            rm.rates[j] > 0 || continue
            mask = rm.via_mask[j]
            partners = 0
            for c in 1:rm.C
                mask[c] && (partners += counts[c] - (c == s ? 1 : 0))
            end
            partners > 0 && return false
        end
    end
    return true
end

# ---------------------------------------------------------------------------------------------------------------
# The process alone
# ---------------------------------------------------------------------------------------------------------------

"""
    evolve_graph!(g::SimpleGraph, process::GraphProcess, Δt; rng = Random.default_rng()) -> NamedTuple

Run `process` alone (no epidemic) on the graph `g` for the time `Δt ≥ 0`, mutating `g`, and return the process's
event counts (for [`NeighbourExchangeProcess`](@ref): `(events, rewired)`, the attempted swaps and those that were
not rejected). It uses the same events as a run of `NextReaction` or `HAS` on `DynamicGraph(g, process)`.

```julia
g = random_regular_graph(1000, 6; rng = StableRNG(1))
evolve_graph!(g, NeighbourExchangeProcess(1.0), 2.0; rng = StableRNG(2))   # ≈ 3000 swaps (ηE/2 per unit time)
all(==(6), degree(g))                                                       # true: degrees are preserved
```
"""
function evolve_graph!(g::SimpleGraph{Int}, process::GraphProcess, Δt::Real;
                       rng::AbstractRNG = Random.default_rng())
    (isfinite(Δt) && Δt >= 0) || throw(ArgumentError("evolve_graph!: Δt must be finite and ≥ 0; got $(Δt)"))
    _check_contact_graph(g)
    ps = _process_state(process, g, rng)
    t = 0.0
    node_state = Int[]           # no epidemic: the process sees no node states
    while true
        r = _process_rate(ps)
        r > 0 || break
        t += randexp(rng) / r
        t <= Δt || break
        _process_fire!(ps, g, node_state, rng)
    end
    return _process_summary(ps)
end
evolve_graph!(g::AbstractGraph, process::GraphProcess, Δt::Real; kw...) = throw(ArgumentError(
    "evolve_graph!: the graph must be a SimpleGraph{Int} (it is mutated in place); got $(typeof(g))"))

# ---------------------------------------------------------------------------------------------------------------
# simulate(model, net::DynamicNetwork; …)
# ---------------------------------------------------------------------------------------------------------------

"""
    sample_graph(net::DynamicNetwork, N::Integer; rng = Random.default_rng()) -> (DynamicGraph, info::GraphInfo)

The contact network a run of `simulate(model, net; N, …)` starts from: a sample `g` of the base network
(`sample_graph(net.base, N; rng)`, whose [`GraphInfo`](@ref) is returned) with the process attached,
`DynamicGraph(g, process)` (`NeighbourExchange(η)` → [`NeighbourExchangeProcess`](@ref)). The run then rewires its
own copy of `g`. A process whose initial state is not a sample of the base (dormant contacts) has its own method.
"""
function sample_graph(net::DynamicNetwork, N::Integer; rng::AbstractRNG = Random.default_rng())
    g, info = sample_graph(net.base, N; rng)
    return DynamicGraph(g, _graph_process(net.process)), info
end

# The contact network of one run (and of the scenario runner): `sample_graph(net, N; rng)`, i.e. the process on a
# fresh sample of the base network, or the method of a process with its own initial state.
_initial_network(net::DynamicNetwork, N::Int, rng::AbstractRNG) = first(sample_graph(net, N; rng))

"""
    simulate(model, net::DynamicNetwork; N, p = Dict(), initial, tspan, nsims = 1, graphs = :per_run,
             algorithm = NextReaction(), seed = nothing, rng = nothing, tgrid = nothing, keep = :grid,
             interventions = InterventionPlan(), parallel = false) -> OutbreakEnsemble

Simulate `model` on the dynamic network `net = DynamicNetwork(base, process)` (design §C.3): run r starts from the
graph `sample_graph(net.base, N; rng = NetworkOutbreaks.stable_rng(b + r))` (graph 1 for every run with
`graphs = :fixed`, graph `mod1(r, G)` with `(:pool, G)`) and evolves it by the process
(`NeighbourExchange(η)` → [`NeighbourExchangeProcess`](@ref)) inside the sampler, with the run's own stream
`stable_rng(b + 2^32 + r)` (design §J.7). So every run has its own network path. `algorithm` must be `NextReaction`
(the default) or `HAS`; the other keywords are those of `simulate(model, net::NetworkDescriptor; …)`, and the rate
convention uses the nominal `mean_degree(net)`.

```julia
ens = simulate(sir_model(), DynamicNetwork(RegularDegree(6), NeighbourExchange(1.0)); N = 5000,
               p = Dict(:τ => 1/12, :γ => 1/4), initial = SeedFraction(:I => 0.01), tspan = (0.0, 100.0),
               nsims = 10, seed = 1)
```
"""
function simulate(model, net::DynamicNetwork;
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
    algorithm isa Union{NextReaction, HAS} || throw(ArgumentError(
        "simulate: a DynamicNetwork runs under NextReaction or HAS; got $(nameof(typeof(algorithm)))"))
    ts = (Float64(tspan[1]), Float64(tspan[2]))
    om = _outbreak_model(model, p, net)
    grid = _grid(tgrid, ts, keep)
    base = _base_seed(seed, rng)
    draw(j) = _initial_network(net, Int(N), stable_rng(_graph_seed(base, j)))
    mode = _check_graphs_mode(graphs; allow_fixed = true)
    pool = mode === :per_run ? nothing : mode === :fixed ? [draw(1)] : [draw(j) for j in 1:last(mode)]
    network_of(r) = pool === nothing ? draw(r) : pool[mod1(r, length(pool))]
    trajs = _run_all(r -> _one_run(om, network_of(r), initial, ts, algorithm, _run_seed(base, r), keep, grid,
                                   interventions),
                     Int(nsims), parallel)
    spec_net = mode === :fixed ? only(pool) : SampledNetwork(net, N, mode)
    return OutbreakEnsemble(OutbreakSpec(om, spec_net, initial, ts), trajs)
end
