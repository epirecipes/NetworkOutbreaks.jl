#=
network.jl (owner: WP16)

Contact-network containers. `OutbreakSpec` accepts an `AbstractGraph` (auto-wrapped as `StaticNetwork`) or an
`AbstractContactNetwork`:

- `StaticNetwork`: a fixed graph;
- `TimeVaryingNetwork`: a base graph plus a time-sorted list of edge updates (strict add/remove semantics,
  validated when the `OutbreakSpec` is built);
- `MultiplexGraph`: several layers on one node set with per-layer rate multipliers (NetworkOutbreaks 0.1 called it
  `MultiplexNetwork`; that name is now NetworkEpiCore's multiplex *descriptor*, re-exported);
- `SampledNetwork`: the network of an ensemble whose runs draw fresh graphs from a NetworkEpiCore descriptor.

Other containers are defined next to the processes that use them: `DynamicGraph` (a graph rewired by a graph process
during a run; src/processes/common.jl), `FleetingContacts` (the stub counts of mean-field social heterogeneity;
src/processes/mfsh.jl) and `TypedGraph` (a graph that records node types; src/generators/multitype.jl, a
`Graphs.AbstractGraph` rather than a container).

Graphs are never mutated: samplers copy a time-varying or dynamic graph at the start of every run.
=#

"""
    AbstractContactNetwork

Supertype of the contact-network containers an [`OutbreakSpec`](@ref) accepts: [`StaticNetwork`](@ref),
[`TimeVaryingNetwork`](@ref), [`MultiplexGraph`](@ref) and [`SampledNetwork`](@ref). A plain
`Graphs.AbstractGraph` is wrapped as a `StaticNetwork`.
"""
abstract type AbstractContactNetwork end

"""
    StaticNetwork{G<:AbstractGraph}

Wraps a fixed contact graph. Created automatically when a plain `AbstractGraph` is passed to `OutbreakSpec`.
Contact graphs must be undirected and free of self-loops (checked when a simulation starts).
"""
struct StaticNetwork{G <: AbstractGraph} <: AbstractContactNetwork
    graph::G
end

const _TVN_Update = NamedTuple{(:t, :src, :dst, :action),
                               Tuple{Float64, Int, Int, Symbol}}

"""
    TimeVaryingNetwork(graph, updates)

A contact network whose edges change over time. `graph` is the base graph: the network at the start `tspan[1]` of
a run. `updates` is a collection of edge updates, each with fields

- `t`: the time at which the update applies;
- `src`, `dst`: the end nodes (1-based; an edge is an unordered pair);
- `action`: `:add` (the edge is present from `t` on) or `:remove` (absent from `t` on).

The updates are sorted by time (stably, so simultaneous updates keep their order). Updates before `tspan[1]`
describe the network's past and are skipped; the others are applied in order, before any event at the same time.

The semantics are strict, because a simple graph has no edge multiplicities: when the `OutbreakSpec` is built, an
`:add` of an edge that is already present, or a `:remove` of an edge that is absent, at the time of the update is an
`ArgumentError` (overlapping contacts on the same pair must be merged into one interval), as are end nodes outside
`1:nv(graph)`, self-loops and unknown actions. The graph is copied at the start of every run, so `simulate` and
`simulate_ensemble` can be called repeatedly on the same spec.

Supported by `DirectSSA`, `NextReaction` and `HAS`; `CompositionRejection` raises an `ArgumentError`.

```julia
g = path_graph(4)
tvn = TimeVaryingNetwork(g, [(t = 5.0, src = 1, dst = 4, action = :add),
                            (t = 8.0, src = 1, dst = 2, action = :remove)])
```
"""
struct TimeVaryingNetwork{G <: AbstractGraph} <: AbstractContactNetwork
    graph::G
    updates::Vector{_TVN_Update}
end

# Accepts heterogeneous iterables of NamedTuples (t may be an Integer, action a String).
function TimeVaryingNetwork(graph::G, updates) where {G <: AbstractGraph}
    typed = _TVN_Update[(t = Float64(u.t), src = Int(u.src),
                         dst = Int(u.dst), action = Symbol(u.action))
                        for u in updates]
    issorted(typed; by = u -> u.t) || sort!(typed; by = u -> u.t)
    return TimeVaryingNetwork{G}(graph, typed)
end

# Validate the updates that a run starting at `t0` applies (design decision on the NO 0.1 no-op semantics, WP3
# review: strict add/remove). Applies them in order to a copy of the base graph.
function _check_tvn_updates(net::TimeVaryingNetwork, t0::Float64)
    g = copy(net.graph)
    n = nv(g)
    for u in net.updates
        isnan(u.t) && throw(ArgumentError("TimeVaryingNetwork update $(u) has time NaN"))
        (1 <= u.src <= n && 1 <= u.dst <= n) ||
            throw(ArgumentError("TimeVaryingNetwork update $(u) refers to a node outside 1:$(n)"))
        u.src == u.dst && throw(ArgumentError("TimeVaryingNetwork update $(u) would add a self-loop"))
        u.action in (:add, :remove) ||
            throw(ArgumentError("TimeVaryingNetwork update action must be :add or :remove; got $(u.action)"))
        u.t < t0 && continue            # the network's past: skipped by the samplers
        if u.action === :add
            add_edge!(g, u.src, u.dst) || throw(ArgumentError(
                "TimeVaryingNetwork update $(u): the edge $(u.src)–$(u.dst) is already present at t = $(u.t); " *
                "an :add needs an absent edge (merge overlapping contacts into one interval)"))
        else
            rem_edge!(g, u.src, u.dst) || throw(ArgumentError(
                "TimeVaryingNetwork update $(u): the edge $(u.src)–$(u.dst) is absent at t = $(u.t); a :remove " *
                "needs a present edge"))
        end
    end
    return nothing
end

"""
    MultiplexGraph(layers, layer_rates; names = [:layer1, :layer2, …])

A multiplex contact network: several layers (graphs) on the same node set, each with a non-negative rate multiplier
and a name. The contact hazard contributed by a catalyst neighbour reached through layer ℓ is
`tr.rate * layer_rates[ℓ]`, where `tr.rate` is the rate of the model's contact transition (a neighbour linked in
several layers counts once per layer). A contact transition with `layer = :all` acts on every layer; one with
`layer = name` (see [`OutbreakTransition`](@ref); a NetworkEpiCore `Contact` on a named layer keeps it) acts only on
the layer called `name`, so hazard_j(v) = rate_j · Σ_ℓ [layer_j ∈ (:all, names[ℓ])] · layer_rates[ℓ] ·
#{u ∈ N_ℓ(v) : state(u) ∈ via_j}.

All layers must have the same number of nodes; the names must be unique and must not be `:all`. Supported by
`DirectSSA`, `NextReaction` and `HAS`; `CompositionRejection` raises an `ArgumentError`.
`sample_graph(net::MultiplexNetwork, N)` draws one from a NetworkEpiCore multiplex descriptor, named after its
layers.

NetworkOutbreaks 0.1 called this type `MultiplexNetwork`; that name now belongs to NetworkEpiCore's multiplex
*descriptor* (`MultiplexNetwork(:home => RegularDegree(3), …)`), which NetworkOutbreaks re-exports. The 0.1 call
`MultiplexNetwork(layers, layer_rates)` is a `MethodError` whose message points here.

```julia
households = erdos_renyi(N, 4 / N)
schools    = erdos_renyi(N, 8 / N)
net  = MultiplexGraph([households, schools], [2.0, 1.0]; names = [:home, :school])
spec = OutbreakSpec(model = model, network = net, initial = SeedFraction(:I => 0.01), tspan = (0.0, 40.0))
```
"""
struct MultiplexGraph{G <: AbstractGraph} <: AbstractContactNetwork
    layers::Vector{G}
    layer_rates::Vector{Float64}
    names::Vector{Symbol}

    function MultiplexGraph{G}(layers::Vector{G}, layer_rates::Vector{Float64},
                               names::Vector{Symbol} = _default_layer_names(length(layers))) where {G <: AbstractGraph}
        isempty(layers) && throw(ArgumentError("MultiplexGraph requires at least one layer"))
        length(layers) == length(layer_rates) ||
            throw(ArgumentError("layers and layer_rates must have the same length"))
        length(names) == length(layers) ||
            throw(ArgumentError("MultiplexGraph: $(length(names)) layer names for $(length(layers)) layers"))
        allunique(names) || throw(ArgumentError("MultiplexGraph: the layer names must be unique; got $(names)"))
        :all in names && throw(ArgumentError(
            "MultiplexGraph: the layer name :all is reserved (a transition with layer = :all acts on every layer)"))
        all(r -> isfinite(r) && r >= 0, layer_rates) ||
            throw(ArgumentError("layer_rates must be finite and non-negative"))
        n = nv(layers[1])
        all(g -> nv(g) == n, layers) ||
            throw(ArgumentError("all layers must have the same number of nodes"))
        return new{G}(layers, layer_rates, names)
    end
end

_default_layer_names(L::Integer) = Symbol[Symbol(:layer, l) for l in 1:L]

function MultiplexGraph(layers::AbstractVector{<:AbstractGraph},
                        layer_rates::AbstractVector{<:Real};
                        names = _default_layer_names(length(layers)))
    isempty(layers) && throw(ArgumentError("MultiplexGraph requires at least one layer"))
    G = typeof(first(layers))
    all(g -> g isa G, layers) ||
        throw(ArgumentError("all layers of a MultiplexGraph must have the same graph type; got $(unique(typeof.(layers)))"))
    return MultiplexGraph{G}(collect(G, layers), Vector{Float64}(layer_rates), collect(Symbol, names))
end

"""
    layer_names(g::MultiplexGraph) -> Vector{Symbol}

The layer names of a [`MultiplexGraph`](@ref), in layer order (a method of NetworkEpiCore's `layer_names`).
"""
NetworkEpiCore.layer_names(g::MultiplexGraph) = copy(g.names)

# The migration hint for the NetworkOutbreaks 0.1 container call `MultiplexNetwork(layers, layer_rates)`, which now
# reaches NetworkEpiCore's descriptor (registered for MethodError in `__init__`, src/NetworkOutbreaks.jl).
const _MULTIPLEX_MIGRATION_HINT =
    "NetworkOutbreaks 0.2: the multiplex graph container of NetworkOutbreaks 0.1, " *
    "MultiplexNetwork(layers, layer_rates), is now MultiplexGraph(layers, layer_rates). The name MultiplexNetwork " *
    "belongs to NetworkEpiCore's multiplex descriptor, MultiplexNetwork(:home => RegularDegree(3), …), whose " *
    "graphs sample_graph(net, N) draws as MultiplexGraphs."

# Is `exc` the 0.1 container call MultiplexNetwork(graphs, rates)?
_is_multiplex_container_call(exc::MethodError) =
    exc.f === NetworkEpiCore.MultiplexNetwork && !isempty(exc.args) &&
    (first(exc.args) isa AbstractVector{<:AbstractGraph} || first(exc.args) isa AbstractGraph)

function _multiplex_migration_hint(io::IO, exc::MethodError, argtypes, kwargs)
    _is_multiplex_container_call(exc) && print(io, "\n\n", _MULTIPLEX_MIGRATION_HINT)
    return nothing
end

"""
    SampledNetwork(descriptor::NetworkDescriptor, N::Integer, graphs = :per_run)

The contact network of an ensemble whose runs draw their graphs of `N` nodes from a NetworkEpiCore `descriptor`
(`graphs = :per_run`: a fresh graph per run; `(:pool, G)`: a pool of G graphs shared by the runs). It is the
`network` of the `spec` of an ensemble returned by `simulate(model, net::NetworkDescriptor; N, …)`, which records
the ensemble's settings; it is not a single graph, so `simulate(spec)` refuses it (use `simulate(model, net; …)`,
or [`sample_graph`](@ref) for one graph).
"""
struct SampledNetwork{D <: NetworkDescriptor} <: AbstractContactNetwork
    descriptor::D
    N::Int
    graphs::Union{Symbol, Tuple{Symbol, Int}}
    function SampledNetwork{D}(descriptor::D, N::Integer, graphs) where {D <: NetworkDescriptor}
        N >= 1 || throw(ArgumentError("SampledNetwork: N must be ≥ 1; got $(N)"))
        return new{D}(descriptor, Int(N), _check_graphs_mode(graphs))
    end
end
SampledNetwork(descriptor::D, N::Integer, graphs = :per_run) where {D <: NetworkDescriptor} =
    SampledNetwork{D}(descriptor, N, graphs)

# The graph mode of an ensemble: `:per_run` or `(:pool, G)`, and `:fixed` where one graph for every run is allowed
# (the `graphs` keyword of `simulate`; a `SampledNetwork` never holds `:fixed`, whose ensemble has the graph itself).
function _check_graphs_mode(g; allow_fixed::Bool = false)
    g === :per_run && return g
    allow_fixed && g === :fixed && return g
    g isa Tuple{Symbol, Integer} && first(g) === :pool && last(g) >= 1 && return (:pool, Int(last(g)))
    valid = allow_fixed ? ":per_run, :fixed or (:pool, G)" : ":per_run or (:pool, G)"
    throw(ArgumentError("graphs must be $(valid) with G ≥ 1; got $(repr(g))"))
end

# --- Graph interface delegation ---

_outbreak_graph(n::AbstractContactNetwork) = n.graph
_outbreak_graph(n::AbstractGraph)          = n          # safety fallback
# For multiplex, the "primary" graph is the first layer (used only as a node-count or edge-iteration target by callers
# that are not multiplex-aware).
_outbreak_graph(n::MultiplexGraph)         = n.layers[1]
_outbreak_graph(n::SampledNetwork) = throw(ArgumentError(
    "the network of this spec is a SampledNetwork (fresh graphs drawn from $(n.descriptor) with N = $(n.N)), not " *
    "one graph; run it with simulate(model, net; N, …), or draw a graph with sample_graph(net, N; rng)"))

Graphs.nv(n::AbstractContactNetwork) = nv(n.graph)
Graphs.ne(n::AbstractContactNetwork) = ne(n.graph)
Graphs.nv(n::MultiplexGraph) = nv(n.layers[1])
Graphs.ne(n::MultiplexGraph) = sum(ne, n.layers)
Graphs.nv(n::SampledNetwork) = n.N
Graphs.ne(n::SampledNetwork) = _outbreak_graph(n)

# The descriptor whose nominal mean degree a rate convention uses (see `OutbreakModel(cm, p; network)`).
_rate_network(net::SampledNetwork) = net.descriptor
_rate_network(net::Union{StaticNetwork, TimeVaryingNetwork}) = ExplicitGraph(net.graph)
_layer_names_of(net::MultiplexGraph) = net.names
_layer_names_of(net::SampledNetwork) = _layer_names_of(net.descriptor)
_rate_network(net::MultiplexGraph) = throw(ArgumentError(
    "a frequency- or density-dependent rate convention needs a NetworkDescriptor to take the mean degree from; " *
    "for a MultiplexGraph pass the multiplex descriptor (MultiplexNetwork) or give τ explicitly"))
