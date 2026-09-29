# Owner: WP26 (DESIGN_NetworkEpiCore.md §G.2 WP26, §C.2, §C.3).
#=
processes/neighbour_exchange.jl

NeighbourExchangeProcess: degree-preserving double-edge swaps at total rate ηE/2, the process of
DynamicNetwork(base, NeighbourExchange(η)) (Miller–Slim–Volz dynamic fixed degree, DFD).

Each edge breaks at the per-edge rate η and its two stubs rejoin the stubs of another breaking edge (design §C.2:
"NO realises it as degree-preserving double-edge swaps at total rate ηE/2 (two edges per swap), so each edge rewires
at rate η"). A swap picks an ordered pair of distinct edges uniformly, ({a, b}, {c, d}) with a random orientation of
the second, and proposes {a, d}, {c, b}; over the two orientations this is each of the two re-pairings of the four
stubs with probability 1/2. A proposal that would create a self-loop or a multiple edge is rejected: the event
happens but the graph is unchanged. The proposal is symmetric, so the chain stays on the simple graphs with the
initial degree sequence and the uniform distribution on them is stationary; an edge is rewired at rate η times the
acceptance probability, 1 − O(⟨k²⟩/(N⟨k⟩)).

The edge list is kept alongside the graph so that a uniform edge is drawn in O(1); a swap costs O(k) (sorted
adjacency lists) and changes the neighbourhoods of at most four nodes, whose contact hazards the samplers refresh.
=#

export NeighbourExchangeProcess

"""
    NeighbourExchangeProcess(η)
    NeighbourExchangeProcess(p::NeighbourExchange)

Neighbour exchange on a contact graph: every edge breaks at the **per-edge** rate η ≥ 0 and its two stubs rejoin
the stubs of another breaking edge, so no degree ever changes (the Miller–Slim–Volz dynamic fixed-degree model, the
process of NetworkEpiCore's `DynamicNetwork(base, NeighbourExchange(η))`).

It is realised as degree-preserving double-edge swaps at total rate ηE/2 (E edges, two edges per swap, so each
edge takes part in swaps at rate η): a swap picks two distinct edges {a, b} and {c, d} uniformly at random and
re-pairs their ends as {a, d}, {c, b} or as {a, c}, {b, d}, with probability 1/2 each. A swap that would create a
self-loop or a multiple edge is rejected and leaves the graph unchanged, so the graph stays simple, the uniform
distribution over simple graphs with the initial degrees is stationary, and an edge is rewired at rate η up to the
rejection probability O(⟨k²⟩/(N⟨k⟩)) (about 10⁻³ for 6-regular graphs on 10⁴ nodes).

Attach it to a graph with [`DynamicGraph`](@ref)`(g, NeighbourExchangeProcess(η))` and run `NextReaction` or `HAS`;
the swaps are drawn from the run's random-number generator. η = 0 is the static graph: the run is then identical,
event by event, to the run on the static graph with the same seed. [`evolve_graph!`](@ref) runs the swaps alone.
"""
struct NeighbourExchangeProcess <: GraphProcess
    η::Float64
    function NeighbourExchangeProcess(η)
        η isa Union{AbstractFloat, Integer, Rational, AbstractIrrational} || throw(ArgumentError(
            "NeighbourExchangeProcess: η must be a number; got $(η) (a symbolic η needs a value)"))
        (isfinite(η) && η >= 0) ||
            throw(ArgumentError("NeighbourExchangeProcess: η must be finite and ≥ 0; got $(η)"))
        return new(Float64(η))
    end
end
NeighbourExchangeProcess(p::NeighbourExchange) = NeighbourExchangeProcess(p.η)

_graph_process(p::NeighbourExchange) = NeighbourExchangeProcess(p)

# Per-run state: the edge list (src[i] < dst[i] initially; the orientation of a stored edge carries no meaning), the
# touched nodes of the last swap, and the event counts.
mutable struct _NEState
    η::Float64
    src::Vector{Int}
    dst::Vector{Int}
    touched::Vector{Int}
    attempted::Int
    accepted::Int
end

function _process_state(p::NeighbourExchangeProcess, g::SimpleGraph{Int}, ::AbstractRNG)
    E = ne(g)
    src_ = Vector{Int}(undef, E)
    dst_ = Vector{Int}(undef, E)
    for (i, e) in enumerate(edges(g))
        src_[i] = src(e)
        dst_[i] = dst(e)
    end
    return _NEState(p.η, src_, dst_, sizehint!(Int[], 4), 0, 0)
end

# Swaps need two edges.
_process_rate(ps::_NEState) = length(ps.src) >= 2 ? ps.η * length(ps.src) / 2 : 0.0

function _process_fire!(ps::_NEState, g::SimpleGraph{Int}, node_state, rng::AbstractRNG)
    empty!(ps.touched)
    E = length(ps.src)
    E >= 2 || return ps.touched
    ps.attempted += 1
    i = rand(rng, 1:E)
    j = rand(rng, 1:(E - 1))
    j >= i && (j += 1)
    a, b = ps.src[i], ps.dst[i]
    c, d = ps.src[j], ps.dst[j]
    rand(rng, Bool) && ((c, d) = (d, c))
    # Proposed {a, d}, {c, b}. Edges sharing a node give a self-loop or re-propose an existing edge, so every
    # degenerate case is rejected by these four checks.
    (a == d || c == b || has_edge(g, a, d) || has_edge(g, c, b)) && return ps.touched
    rem_edge!(g, a, b)
    rem_edge!(g, c, d)
    add_edge!(g, a, d)
    add_edge!(g, c, b)
    ps.src[i], ps.dst[i] = a, d
    ps.src[j], ps.dst[j] = c, b
    ps.accepted += 1
    push!(ps.touched, a, b, c, d)
    return ps.touched
end

_process_summary(ps::_NEState) = (events = ps.attempted, rewired = ps.accepted)

Base.show(io::IO, p::NeighbourExchangeProcess) = print(io, "NeighbourExchangeProcess(", p.η, ")")
