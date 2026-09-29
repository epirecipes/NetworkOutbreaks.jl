# Owner: WP25 (DESIGN_NetworkEpiCore.md §G.2 WP25, §C.3).
#
# GraphInfo and the helpers shared by the graph generators (sample_graph methods; the generic and its fallback are in
# src/convenience.jl).
#
# Reproducibility (design §J.7, w1 defect m11). Every generator draws from the `rng` it is given, in a fixed order,
# and never from the global RNG. It uses only `rand(rng)`, `rand(rng, 1:n)` and `rand(rng, d)` for NetworkEpiCore
# degree laws (whose samplers are built on the first two), the calls whose streams StableRNGs keeps stable across
# Julia versions. Stubs are shuffled by the Fisher–Yates shuffle below rather than `Random.shuffle!`, whose algorithm
# is not part of that guarantee; so `sample_graph(net, N; rng = NetworkOutbreaks.stable_rng(s))` gives the same graph
# on every Julia version for given NetworkEpiCore samplers. (Regular and Poisson configuration networks use Graphs.jl's
# `random_regular_graph` and `erdos_renyi`, which are reproducible for a given Graphs.jl version.)

export GraphInfo

"""
    GraphInfo

What [`sample_graph`](@ref) realised: returned next to the graph, `(g, info) = sample_graph(net, N; rng)`.

Fields (the same for every descriptor):

- `descriptor`: the network descriptor that was sampled;
- `method`: the construction, one of `:random_regular`, `:erdos_renyi`, `:erased_configuration` (a
  `ConfigurationNetwork`), `:explicit` (an `ExplicitGraph`), `:complete` (`WellMixed`), `:unstructured`,
  `:stochastic_block_model` and `:typed_configuration` (a `MultitypeNetwork`), `:newman_miller` (a
  `ClusteredNetwork`) and `:multiplex` (the NetworkEpiCore `MultiplexNetwork`);
- `N`, `edges`: the numbers of nodes and edges (summed over the layers of a multiplex);
- `mean_degree`, `excess_degree`: the realised ⟨k⟩ = 2E/N and ⟨k(k − 1)⟩/⟨k⟩ (with k a node's total degree over
  the layers of a multiplex), to compare with `mean_degree(net)` and `excess_degree(net)`;
- `candidate_edges`, `erased`: the numbers of stub pairs formed by stub matching and of those erased because they
  were self-loops or repeated an edge (0 for constructions without stub matching);
- `erased_fraction`: `erased / candidate_edges` (0 when no stubs were matched);
- `details`: a `NamedTuple` of construction-specific statistics, also readable as properties (`info.transitivity`).
  Stubs removed to make stub totals consistent (`trimmed_stubs`) are reported here and are not counted as erased.

The `details` of each method:

- `:erdos_renyi`: `edge_probability` = μ/(N − 1);
- `:erased_configuration`: `parity_redraws` (degrees redrawn to make the degree sum even), `max_degree` (the largest
  drawn degree), `structural_cutoff` = √(N⟨k⟩) and `cutoff_exceeded` (`true` if the largest degree of the law, or
  the largest drawn degree for an unbounded law, exceeds the cutoff: multi-edges are then no longer rare, and the
  erased graph is disassortative);
- `:complete`: `κ` and `rate_scale` = κ/(N − 1) (see `sample_graph(::WellMixed, N)`);
- the `MultitypeNetwork` methods: `types`, `type_sizes` (nodes per type), `stub_totals` (S[a, b]: stubs of type-a
  nodes towards type b after the conditioning of the block totals; for `:unstructured` and
  `:stochastic_block_model`, which match no typed stubs, the realised edge ends), `matched_stubs` (after trimming;
  symmetric), `trimmed_blocks` (the blocks (a, b) whose totals could not be conditioned and were trimmed),
  `trimmed_stubs`, `trimmed_fraction`, `block_edges` (E[a, b]: realised edges between types a and b, symmetric),
  `mean_contacts` (realised M[a, b] = edge ends from type-a nodes to type-b nodes per type-a node) and
  `target_mean_contacts`;
- `:newman_miller`: `triangles_placed` (groups of three corners), `degenerate_triangles` (groups with a repeated
  node), `trimmed_stubs`, `trimmed_corners`, `triangles` (triangles in the realised graph, including accidental
  ones), `connected_triples`, `transitivity` = 3·triangles/connected triples and `target_transitivity` =
  `clustering_coefficient(net)`;
- `:multiplex`: `layers`, a vector of `layer name => GraphInfo`.
"""
struct GraphInfo
    descriptor::NetworkDescriptor
    method::Symbol
    N::Int
    edges::Int
    mean_degree::Float64
    excess_degree::Float64
    candidate_edges::Int
    erased::Int
    erased_fraction::Float64
    details::NamedTuple
end

function Base.getproperty(info::GraphInfo, s::Symbol)
    hasfield(GraphInfo, s) && return getfield(info, s)
    d = getfield(info, :details)
    haskey(d, s) && return getfield(d, s)
    throw(ArgumentError("GraphInfo (method :$(getfield(info, :method))) has no property :$s; its properties are " *
                        join(propertynames(info), ", ")))
end
Base.propertynames(info::GraphInfo, private::Bool = false) =
    (fieldnames(GraphInfo)..., keys(getfield(info, :details))...)

Base.:(==)(a::GraphInfo, b::GraphInfo) = all(getfield(a, f) == getfield(b, f) for f in fieldnames(GraphInfo))
Base.hash(info::GraphInfo, h::UInt) = foldr((f, acc) -> hash(getfield(info, f), acc), fieldnames(GraphInfo);
                                            init = hash(:GraphInfo, h))

function Base.show(io::IO, info::GraphInfo)
    print(io, "GraphInfo(:", info.method, ", N = ", info.N, ", edges = ", info.edges, ", mean degree ",
          _gen_fmt4(info.mean_degree), ", excess degree ", _gen_fmt4(info.excess_degree), ", erased ",
          info.erased, "/", info.candidate_edges, ")")
end

function Base.show(io::IO, ::MIME"text/plain", info::GraphInfo)
    println(io, "GraphInfo: ", info.method, " sample of ", _gen_descriptor_name(info.descriptor))
    println(io, "  nodes ", info.N, ", edges ", info.edges)
    println(io, "  mean degree ", _gen_fmt4(info.mean_degree), ", excess degree ", _gen_fmt4(info.excess_degree))
    print(io, "  erased ", info.erased, " of ", info.candidate_edges, " stub pairs (",
          _gen_fmt4(100 * info.erased_fraction), "%)")
    for (k, v) in pairs(info.details)
        print(io, "\n  ", k, ": ")
        if v isa AbstractVector{<:Pair{Symbol, GraphInfo}}
            print(io, join(("$(first(p)) => $(last(p))" for p in v), "; "))
        else
            print(io, v)
        end
    end
end

_gen_fmt4(x::Real) = string(round(Float64(x); sigdigits = 4))
_gen_descriptor_name(d::NetworkDescriptor) = string(nameof(typeof(d)))

# GraphInfo with the realised degree statistics of `g` (or of the total degrees `ds`).
function _gen_info(net::NetworkDescriptor, method::Symbol, g::AbstractGraph, candidates::Integer, erased::Integer;
                    details::NamedTuple = NamedTuple())
    return _gen_info(net, method, nv(g), ne(g), degree(g), candidates, erased; details)
end
function _gen_info(net::NetworkDescriptor, method::Symbol, N::Integer, edges::Integer, ds::AbstractVector{<:Integer},
                    candidates::Integer, erased::Integer; details::NamedTuple = NamedTuple())
    s = sum(ds; init = 0)
    mean_k = N == 0 ? 0.0 : s / N
    excess = s == 0 ? 0.0 : sum(k -> k * (k - 1), ds; init = 0) / s
    frac = candidates == 0 ? 0.0 : erased / candidates
    return GraphInfo(net, method, Int(N), Int(edges), mean_k, excess, Int(candidates), Int(erased), frac, details)
end

function _gen_check_N(N::Integer)
    N >= 1 || throw(ArgumentError("sample_graph: N must be ≥ 1; got $(N)"))
    return Int(N)
end

# ---------------------------------------------------------------------------------------------------------------
# Random streams
# ---------------------------------------------------------------------------------------------------------------

# Fisher–Yates shuffle from `rand(rng, 1:i)` only (stable under StableRNG).
function _gen_shuffle!(rng::AbstractRNG, v::AbstractVector)
    @inbounds for i in lastindex(v):-1:(firstindex(v) + 1)
        j = rand(rng, firstindex(v):i)
        v[i], v[j] = v[j], v[i]
    end
    return v
end

# Degree samplers built once per graph (NetworkEpiCore's single-draw `rand(rng, d)` rebuilds the table of a
# finite-support law on every call). Laws with a closed-form sampler are used as they are.
struct _GenTableSampler
    cdf::Vector{Float64}       # cdf[k + 1] = P(K ≤ k)
end
# Built from probabilities by `_gen_table` (not by the default constructor, which takes the CDF itself).
function _gen_table(p::AbstractVector{<:Real})
    c = cumsum(Float64.(p))
    c[end] > 0 || throw(ArgumentError("sample_graph: a degree law has no probability mass"))
    c ./= c[end]
    c[end] = 1.0
    return _GenTableSampler(c)
end
# u ∈ (0, 1], so degrees of probability 0 are never drawn.
_gen_draw(rng::AbstractRNG, s::_GenTableSampler) = searchsortedfirst(s.cdf, 1.0 - rand(rng)) - 1

struct _GenMixtureSampler{S}
    choose::_GenTableSampler
    components::Vector{S}
end

_gen_sampler(d::DegreeDistribution) = d
_gen_sampler(d::Union{PowerLawDegree, EmpiricalDegree}) = _gen_table(degree_probabilities(d))
function _gen_sampler(d::MixtureDegree)
    parts = [_gen_sampler(c) for c in d.components]
    return _GenMixtureSampler{eltype(parts)}(_gen_table(d.weights), parts)
end

_gen_draw(rng::AbstractRNG, d::DegreeDistribution) = rand(rng, d)
_gen_draw(rng::AbstractRNG, s::_GenMixtureSampler) = _gen_draw(rng, s.components[_gen_draw(rng, s.choose) + 1])

function _gen_draw_degrees(rng::AbstractRNG, s, n::Integer)
    k = Vector{Int}(undef, n)
    for i in 1:n
        k[i] = _gen_draw(rng, s)
    end
    return k
end

# The largest degree a law can produce, or `nothing` for an unbounded law.
_gen_support_max(d::RegularDegree) = d.k
_gen_support_max(d::BinomialDegree) = d.n
_gen_support_max(d::PowerLawDegree) = d.kmax
_gen_support_max(d::EmpiricalDegree) = something(findlast(>(0), d.p), 1) - 1
function _gen_support_max(d::MixtureDegree)
    ms = [_gen_support_max(c) for (w, c) in zip(d.weights, d.components) if w > 0]
    return any(isnothing, ms) ? nothing : maximum(ms; init = 0)
end
_gen_support_max(::DegreeDistribution) = nothing

# ---------------------------------------------------------------------------------------------------------------
# Stub matching
# ---------------------------------------------------------------------------------------------------------------

# The stub list of `nodes` with `counts[i]` stubs for node `nodes[i]`, appended to `out`.
function _gen_append_stubs!(out::Vector{Int}, nodes, counts)
    for (v, c) in zip(nodes, counts)
        for _ in 1:c
            push!(out, v)
        end
    end
    return out
end

# Add the edge u–v unless it is a self-loop or already present; returns whether it was added.
@inline _gen_add_edge!(g::SimpleGraph, u::Integer, v::Integer) = u != v && add_edge!(g, u, v)

# Pair a shuffled stub list with itself, consecutive stubs forming a pair (an odd last stub is left over). Returns
# (pairs formed, pairs erased).
function _gen_pair_within!(g::SimpleGraph, stubs::Vector{Int})
    m = length(stubs) ÷ 2
    erased = 0
    @inbounds for q in 1:m
        _gen_add_edge!(g, stubs[2q - 1], stubs[2q]) || (erased += 1)
    end
    return m, erased
end

# Pair two shuffled stub lists position by position, up to the shorter length. Returns (pairs formed, erased).
function _gen_pair_across!(g::SimpleGraph, a::Vector{Int}, b::Vector{Int})
    m = min(length(a), length(b))
    erased = 0
    @inbounds for q in 1:m
        _gen_add_edge!(g, a[q], b[q]) || (erased += 1)
    end
    return m, erased
end

# ---------------------------------------------------------------------------------------------------------------
# Realised statistics
# ---------------------------------------------------------------------------------------------------------------

# (triangles, connected triples) of a simple graph: each triangle is counted once, each path of length two once.
function _gen_triangles_and_triples(g::AbstractGraph)
    closed = 0
    triples = 0
    for v in vertices(g)
        nb = neighbors(g, v)
        k = length(nb)
        triples += k * (k - 1) ÷ 2
        for i in 1:k, j in (i + 1):k
            has_edge(g, nb[i], nb[j]) && (closed += 1)
        end
    end
    return closed ÷ 3, triples
end
