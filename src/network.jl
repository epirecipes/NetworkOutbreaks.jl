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

Currently supported algorithms: `DirectSSA`, `NextReaction`.
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

# --- Graph interface delegation ---

_outbreak_graph(n::AbstractContactNetwork) = n.graph
_outbreak_graph(n::AbstractGraph)          = n          # safety fallback

Graphs.nv(n::AbstractContactNetwork) = nv(n.graph)
Graphs.ne(n::AbstractContactNetwork) = ne(n.graph)
