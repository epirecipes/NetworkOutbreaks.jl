# Owner: WP36b (DESIGN_NetworkEpiCore.md §C.3, §J.7, §K WP36b).
#=
processes/dormant.jl

The dormant-contact (DC) stub process of Miller, Slim & Volz (2012), Part II §3.2.4, the stochastic model behind
NetworkEpiCore's `DynamicNetwork(base, DormantContacts(η_form, η_break))`:

- `DormantContactProcess(η_form, η_break, max_degrees)`: node v has `max_degrees[v]` = k_m stubs; a stub is active
  (half of an edge of the contact graph) or dormant. Every active edge breaks at rate η_break (MSV's η₂) and both of
  its stubs become dormant; every dormant stub becomes active at rate η_form (MSV's η₁) by pairing with another
  activating stub, i.e. unordered pairs of dormant stubs form edges at rate η_form/(D − 1) each, a total rate
  η_form·D/2 with D dormant stubs. So each stub alternates between exponential active (rate η_break) and dormant
  (rate η_form) periods, independently of the other stubs, and a new partner is drawn in proportion to dormant
  stubs, as the edge-based model assumes.
- `sample_graph(net::DynamicNetwork{<:DormantContacts}, N; rng)`: the stationary initial state of one run: N maximum
  degrees drawn from the base law, each stub active with probability A = η_form/(η_form + η_break) independently,
  and the active stubs paired uniformly at random (erased configuration model: a self-loop or a repeated edge
  leaves its two stubs dormant, as does an odd last stub).
- `simulate(model, net::DynamicNetwork{<:DormantContacts}; N, …)`: the route of processes/common.jl, with a fresh
  initial state per run (the graph stream of design §J.7) and the process in NextReaction or HAS.

A pairing that would create a self-loop or repeat an active edge is rejected: the event happens and both stubs stay
dormant. The contact graph stays simple and the per-stub activation rate is η_form up to the rejection probability,
O(k_m/(N A⟨k_m⟩)) per pairing. A process event changes the neighbourhoods of two nodes, whose contact hazards the
samplers refresh; its cost is O(k) (sorted adjacency lists), and a uniform active edge or dormant stub is drawn in
O(1) from the lists kept alongside the graph.
=#

export DormantContactProcess

"""
    DormantContactProcess(η_form, η_break, max_degrees)
    DormantContactProcess(p::DormantContacts, max_degrees)

The dormant-contact stub process of Miller, Slim & Volz (2012, Part II §3.2.4), the process of NetworkEpiCore's
`DynamicNetwork(base, DormantContacts(η_form, η_break))`. Node v has `max_degrees[v]` stubs; each is active (half of
an edge of the contact graph) or dormant. Every active edge breaks at rate `η_break` (MSV's η₂) and its two stubs
become dormant; every dormant stub becomes active at rate `η_form` (MSV's η₁) by pairing with another activating
stub (unordered pairs of dormant stubs form edges at rate η_form/(D − 1) each, D the number of dormant stubs). At
stationarity each stub is active with probability η_form/(η_form + η_break), independently of the others.

Attach it to a graph whose degrees do not exceed `max_degrees` with [`DynamicGraph`](@ref)`(g, process)` (the
dormant stubs of v are then `max_degrees[v] − degree(g, v)`) and run `NextReaction` or `HAS`; the events are drawn
from the run's random-number generator. A pairing that would create a self-loop or repeat an active edge is
rejected (the stubs stay dormant), so the graph stays simple. `simulate(model, DynamicNetwork(base,
DormantContacts(η_form, η_break)); N, …)` draws the stationary initial state of every run with
[`sample_graph`](@ref). [`evolve_graph!`](@ref) runs the process alone and returns
`(events, broken, formed, rejected)`. Both rates must be finite numbers ≥ 0; `max_degrees` is copied.

```julia
g, info = sample_graph(DynamicNetwork(EmpiricalDegree(2 => 0.5, 8 => 0.5), DormantContacts(1.0, 1.0)), 10_000;
                       rng = NetworkOutbreaks.stable_rng(1))
g isa DynamicGraph && g.process isa DormantContactProcess     # true
info.active_fraction                                           # ≈ 0.5
```
"""
struct DormantContactProcess <: GraphProcess
    η_form::Float64
    η_break::Float64
    max_degrees::Vector{Int}
    function DormantContactProcess(η_form, η_break, max_degrees::AbstractVector{<:Integer})
        for (name, η) in (("η_form", η_form), ("η_break", η_break))
            η isa Union{AbstractFloat, Integer, Rational, AbstractIrrational} || throw(ArgumentError(
                "DormantContactProcess: $(name) must be a number; got $(η) (a symbolic rate needs a value)"))
            (isfinite(η) && η >= 0) ||
                throw(ArgumentError("DormantContactProcess: $(name) must be finite and ≥ 0; got $(η)"))
        end
        all(>=(0), max_degrees) || throw(ArgumentError(
            "DormantContactProcess: maximum degrees must be ≥ 0; got $(minimum(max_degrees))"))
        return new(Float64(η_form), Float64(η_break), collect(Int, max_degrees))
    end
end
DormantContactProcess(p::DormantContacts, max_degrees::AbstractVector{<:Integer}) =
    DormantContactProcess(p.η_form, p.η_break, max_degrees)

Base.:(==)(a::DormantContactProcess, b::DormantContactProcess) =
    a.η_form == b.η_form && a.η_break == b.η_break && a.max_degrees == b.max_degrees
Base.hash(p::DormantContactProcess, h::UInt) =
    hash(p.max_degrees, hash(p.η_break, hash(p.η_form, hash(:DormantContactProcess, h))))

function Base.show(io::IO, p::DormantContactProcess)
    n = length(p.max_degrees)
    print(io, "DormantContactProcess(η_form = ", p.η_form, ", η_break = ", p.η_break, ", ", n, " nodes, ",
          sum(p.max_degrees; init = 0), " stubs)")
end

# A NetworkEpiCore DormantContacts descriptor does not say how many stubs each node of a given graph has.
_graph_process(p::DormantContacts) = throw(ArgumentError(
    "DormantContacts($(p.η_form), $(p.η_break)) needs the maximum degree (number of stubs) of every node: attach " *
    "DormantContactProcess(η_form, η_break, max_degrees) to the graph of active edges, " *
    "DynamicGraph(g, DormantContactProcess(p, max_degrees)), or let simulate(model, DynamicNetwork(base, " *
    "DormantContacts(η_form, η_break)); N, …) draw the stationary initial state of each run"))

# The graph must fit the stubs: one maximum degree per node, and no node with more edges than stubs.
function _check_dormant_graph(g::AbstractGraph, p::DormantContactProcess)
    length(p.max_degrees) == nv(g) || throw(ArgumentError(
        "DormantContactProcess: $(length(p.max_degrees)) maximum degrees for a graph of $(nv(g)) nodes"))
    for v in vertices(g)
        degree(g, v) <= p.max_degrees[v] || throw(ArgumentError(
            "DormantContactProcess: node $(v) has $(degree(g, v)) edges but only $(p.max_degrees[v]) stubs"))
    end
    return nothing
end

function DynamicGraph(graph::AbstractGraph, process::DormantContactProcess)
    _check_contact_graph(graph)
    _check_dormant_graph(graph, process)
    return DynamicGraph{typeof(graph), DormantContactProcess}(graph, process)
end

# Per-run state: the active edge list (src[i], dst[i]; its order carries no meaning), the dormant-stub pool (one entry
# per dormant stub: its node), the touched nodes of the last event, and the event counts.
mutable struct _DCState
    η_form::Float64
    η_break::Float64
    src::Vector{Int}
    dst::Vector{Int}
    pool::Vector{Int}
    touched::Vector{Int}
    broken::Int
    formed::Int
    rejected::Int
end

function _process_state(p::DormantContactProcess, g::SimpleGraph{Int}, ::AbstractRNG)
    _check_dormant_graph(g, p)
    E = ne(g)
    src_ = Vector{Int}(undef, E)
    dst_ = Vector{Int}(undef, E)
    for (i, e) in enumerate(edges(g))
        src_[i] = src(e)
        dst_[i] = dst(e)
    end
    pool = Int[]
    sizehint!(pool, sum(p.max_degrees; init = 0) - 2E)
    for v in vertices(g), _ in 1:(p.max_degrees[v] - degree(g, v))
        push!(pool, v)
    end
    return _DCState(p.η_form, p.η_break, src_, dst_, pool, sizehint!(Int[], 2), 0, 0, 0)
end

_dc_break_rate(ps::_DCState) = ps.η_break * length(ps.src)
_dc_form_rate(ps::_DCState) = (D = length(ps.pool); D >= 2 ? ps.η_form * D / 2 : 0.0)
_process_rate(ps::_DCState) = _dc_break_rate(ps) + _dc_form_rate(ps)

# Remove entry i of v by moving the last entry into its place.
@inline function _dc_swap_remove!(v::Vector{Int}, i::Int)
    @inbounds v[i] = v[end]
    pop!(v)
    return v
end

function _process_fire!(ps::_DCState, g::SimpleGraph{Int}, node_state, rng::AbstractRNG)
    empty!(ps.touched)
    rb = _dc_break_rate(ps)
    rf = _dc_form_rate(ps)
    rb + rf > 0 || return ps.touched
    if rand(rng) * (rb + rf) < rb
        # an active edge breaks: both stubs become dormant
        i = rand(rng, 1:length(ps.src))
        u, v = ps.src[i], ps.dst[i]
        rem_edge!(g, u, v)
        _dc_swap_remove!(ps.src, i)
        _dc_swap_remove!(ps.dst, i)
        push!(ps.pool, u, v)
        ps.broken += 1
    else
        # two distinct dormant stubs pair (uniform unordered pair)
        D = length(ps.pool)
        i = rand(rng, 1:D)
        j = rand(rng, 1:(D - 1))
        j >= i && (j += 1)
        u, v = ps.pool[i], ps.pool[j]
        if u == v || has_edge(g, u, v)
            ps.rejected += 1
            return ps.touched
        end
        add_edge!(g, u, v)
        push!(ps.src, u)
        push!(ps.dst, v)
        _dc_swap_remove!(ps.pool, max(i, j))
        _dc_swap_remove!(ps.pool, min(i, j))
        ps.formed += 1
    end
    push!(ps.touched, u, v)
    return ps.touched
end

_process_summary(ps::_DCState) =
    (events = ps.broken + ps.formed + ps.rejected, broken = ps.broken, formed = ps.formed, rejected = ps.rejected)

# ---------------------------------------------------------------------------------------------------------------
# The stationary initial state of a run
# ---------------------------------------------------------------------------------------------------------------

function _dc_rates(p::DormantContacts)
    for (name, η) in (("η_form", p.η_form), ("η_break", p.η_break))
        η isa Union{AbstractFloat, Integer, Rational} || throw(ArgumentError(
            "sample_graph: DormantContacts needs numeric rates; $(name) = $(η) is symbolic"))
    end
    a, b = Float64(p.η_form), Float64(p.η_break)
    a + b > 0 || throw(ArgumentError("sample_graph: DormantContacts needs η_form + η_break > 0"))
    return a, b
end

"""
    sample_graph(net::DynamicNetwork{<:DormantContacts}, N::Integer; rng = Random.default_rng())
        -> (DynamicGraph, info::GraphInfo)

The stationary initial state of one run on the dormant-contact network `net = DynamicNetwork(base,
DormantContacts(η_form, η_break))` (Miller, Slim & Volz 2012, Part II §3.2.4): N maximum degrees k_m drawn iid from
the base degree law, each stub active with probability A = η_form/(η_form + η_break) independently, and the active
stubs paired uniformly at random. A pair that would be a self-loop or repeat an edge is erased and its two stubs stay
dormant (`info.erased`), as does an odd last active stub. The result is the contact network of the run,
`DynamicGraph(g, DormantContactProcess(net.process, k_m))`, whose graph `g` holds the active edges.

`info.method == :dormant_contacts`; `info.mean_degree` and `info.excess_degree` describe the active graph (compare
`mean_degree(net)` = A⟨k_m⟩), and `info.details` has `mean_max_degree` (⟨k_m⟩ of the sample), `active_fraction`
(active stubs / all stubs), `target_active_fraction` (A) and `dormant_stubs`; the maximum degrees themselves are
`g.process.max_degrees`. Reproducible for a
given `rng` state, e.g. `rng = NetworkOutbreaks.stable_rng(s)` (the graph stream of design §J.7).
"""
function sample_graph(net::DynamicNetwork{<:DormantContacts}, N::Integer; rng::AbstractRNG = Random.default_rng())
    N = _gen_check_N(N)
    η_form, η_break = _dc_rates(net.process)
    A = η_form / (η_form + η_break)
    km = _gen_draw_degrees(rng, _gen_sampler(net.base.degrees), N)
    active = Int[]
    for v in 1:N, _ in 1:km[v]
        rand(rng) < A && push!(active, v)
    end
    _gen_shuffle!(rng, active)
    g = SimpleGraph(N)
    pairs, erased = _gen_pair_within!(g, active)
    M = sum(km; init = 0)
    details = (mean_max_degree = M / N, active_fraction = M == 0 ? 0.0 : 2 * ne(g) / M,
               target_active_fraction = A, dormant_stubs = M - 2 * ne(g))
    info = _gen_info(net, :dormant_contacts, g, pairs, erased; details)
    return DynamicGraph(g, DormantContactProcess(η_form, η_break, km)), info
end

# The contact network of one run of simulate(model, net::DynamicNetwork; …) and of the scenario runner.
_initial_network(net::DynamicNetwork{<:DormantContacts}, N::Int, rng::AbstractRNG) = first(sample_graph(net, N; rng))
