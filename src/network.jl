#=
network.jl

Contact network wrapper types. `OutbreakSpec` accepts either an
`AbstractGraph` (auto-wrapped as `StaticNetwork`) or an
`AbstractContactNetwork` directly.

`TimeVaryingNetwork` holds a mutable base graph and a sorted vector of
edge-level updates `(t, src, dst, action)` where `action ∈ (:add, :remove)`.
Algorithms that support time-varying networks copy the mutable graph at
simulation start so the spec is not mutated between runs.
=#

abstract type AbstractContactNetwork end

"""
    StaticNetwork{G<:AbstractGraph}

Wraps a fixed contact graph. Created automatically when a plain
`AbstractGraph` is passed to `OutbreakSpec`.
"""
struct StaticNetwork{G <: AbstractGraph} <: AbstractContactNetwork
    graph::G
end

"""
    TimeVaryingNetwork{G<:AbstractGraph}

A contact network whose topology changes over time. `graph` is the mutable
base graph (state at `tspan[1]`). `updates` is a *sorted* (by `.t`) vector of
edge updates applied during a simulation.

Each update is a `NamedTuple` with fields:
- `t`      :: Float64  — simulation time at which the update fires
- `src`    :: Int      — source node
- `dst`    :: Int      — destination node
- `action` :: Symbol   — `:add` or `:remove`

The graph is `deepcopy`-ed at the start of each trajectory so `simulate` and
`simulate_ensemble` are safe to call repeatedly on the same spec.

Currently supported algorithms: `DirectSSA`, `NextReaction`, and `HAS`.
`CompositionRejection` raises an `ArgumentError` if combined with a
`TimeVaryingNetwork`.
"""
const _TVN_Update = NamedTuple{(:t, :src, :dst, :action),
                               Tuple{Float64, Int, Int, Symbol}}

struct TimeVaryingNetwork{G <: AbstractGraph} <: AbstractContactNetwork
    graph::G
    updates::Vector{_TVN_Update}
end

# Convenience constructor: accepts heterogeneous iterables of NamedTuples.
function TimeVaryingNetwork(graph::G, updates) where {G <: AbstractGraph}
    typed = _TVN_Update[(t = Float64(u.t), src = Int(u.src),
                         dst = Int(u.dst), action = Symbol(u.action))
                        for u in updates]
    issorted(typed; by = u -> u.t) ||
        sort!(typed; by = u -> u.t)
    return TimeVaryingNetwork{G}(graph, typed)
end

"""
    MultiplexNetwork{G<:AbstractGraph}

A multiplex contact network: several layers (sub-graphs) over the same node
set, each carrying its own per-edge transmission-rate multiplier. The
infection hazard contributed by an infectious neighbour reached via layer
`ℓ` is `tr.rate * layer_rates[ℓ]`, where `tr.rate` comes from the
`OutbreakModel` infection transition.

All layers must share the same number of nodes. Currently supported by
`DirectSSA`; combining a `MultiplexNetwork` with `TimeVaryingNetwork`
semantics or with `CompositionRejection` is not supported.

# Example
```julia
households = erdos_renyi(N, 4 / N)
schools    = erdos_renyi(N, 8 / N)
net        = MultiplexNetwork([households, schools], [2.0, 1.0])
spec       = OutbreakSpec(model, net, SeedFraction(0.01), (0.0, 40.0))
```
"""
struct MultiplexNetwork{G <: AbstractGraph} <: AbstractContactNetwork
    layers::Vector{G}
    layer_rates::Vector{Float64}

    function MultiplexNetwork{G}(layers::Vector{G},
                                 layer_rates::Vector{Float64}) where {G <: AbstractGraph}
        isempty(layers) && throw(ArgumentError("MultiplexNetwork requires at least one layer"))
        length(layers) == length(layer_rates) ||
            throw(ArgumentError("layers and layer_rates must have the same length"))
        all(>=(0), layer_rates) ||
            throw(ArgumentError("layer_rates must be non-negative"))
        n = nv(layers[1])
        all(g -> nv(g) == n, layers) ||
            throw(ArgumentError("all layers must have the same number of nodes"))
        return new{G}(layers, layer_rates)
    end
end

function MultiplexNetwork(layers::AbstractVector{<:AbstractGraph},
                          layer_rates::AbstractVector{<:Real})
    G = typeof(layers[1])
    layers_v = Vector{G}(undef, length(layers))
    @inbounds for (i, g) in pairs(layers)
        layers_v[i] = g
    end
    return MultiplexNetwork{G}(layers_v, Vector{Float64}(layer_rates))
end

# --- Graph interface delegation ---

_outbreak_graph(n::AbstractContactNetwork) = n.graph
_outbreak_graph(n::AbstractGraph)          = n          # safety fallback
# For multiplex, the "primary" graph is the first layer (used only as a
# convenient nv/edge-iteration target by callers that don't multiplex-aware).
_outbreak_graph(n::MultiplexNetwork)       = n.layers[1]

Graphs.nv(n::AbstractContactNetwork) = nv(n.graph)
Graphs.ne(n::AbstractContactNetwork) = ne(n.graph)
Graphs.nv(n::MultiplexNetwork) = nv(n.layers[1])
Graphs.ne(n::MultiplexNetwork) = sum(ne, n.layers)
