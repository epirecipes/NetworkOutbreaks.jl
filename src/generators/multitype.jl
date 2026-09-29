# Owner: WP25 (DESIGN_NetworkEpiCore.md §G.2 WP25, §C.3, §J.6).
#
# sample_graph for MultitypeNetwork: typed stub matching per (a, b) block with exact reciprocal totals (block totals
# conditioned, or trimmed where that is impossible), with exact constructions for unstructured and Poisson-block laws;
# node types stored in a TypedGraph; per-stratum seeding (the `_seed_nodes!` hook of src/spec.jl, design §J.6).

export TypedGraph, node_types, nodes_of_type

# ---------------------------------------------------------------------------------------------------------------
# TypedGraph
# ---------------------------------------------------------------------------------------------------------------

"""
    TypedGraph(graph, types::Vector{Symbol}, type_of::Vector{Int})
    TypedGraph(graph, node_types::AbstractVector{Symbol})

An undirected graph whose nodes carry types: node v has type `types[type_of[v]]`. It is a `Graphs.AbstractGraph`
that forwards the graph interface to `graph`, so it can be used wherever a graph can (`simulate(model, g; …)`,
`OutbreakSpec`, Graphs.jl algorithms). The second form takes one type name per node, with the types in order of
first appearance.

[`sample_graph`](@ref) returns a `TypedGraph` for a `MultitypeNetwork`. When a stratified model
(`stratify(sir_model(), [:a, :b])`) is simulated on it, each stratum is seeded on the nodes of its own type
(design §J.6): `SeedFraction(:I_a => ρ)` puts ρN of the N nodes (a fraction of **all** nodes, as in
NetworkEpiCore's `final_size` and the edge-based model) into `I_a`, chosen uniformly among the type-a nodes, and the
other type-a nodes start in the susceptible compartment of stratum a. A node listed in `SeedNodes` must be assigned
to a compartment of its own type's stratum (or to a compartment without a stratum). The model's strata must be the
graph's types, and `default` may only name a compartment without a stratum. A model without strata is seeded as on
an untyped graph.

See [`node_types`](@ref) and [`nodes_of_type`](@ref).
"""
struct TypedGraph{T <: Integer, G <: AbstractGraph{T}} <: AbstractGraph{T}
    graph::G
    types::Vector{Symbol}
    type_of::Vector{Int}
    function TypedGraph{T, G}(graph::G, types::Vector{Symbol},
                              type_of::Vector{Int}) where {T <: Integer, G <: AbstractGraph{T}}
        is_directed(graph) && throw(ArgumentError("TypedGraph: the graph must be undirected"))
        allunique(types) || throw(ArgumentError("TypedGraph: the type names must be unique; got $(types)"))
        length(type_of) == nv(graph) ||
            throw(ArgumentError("TypedGraph: $(length(type_of)) node types for $(nv(graph)) nodes"))
        all(i -> 1 <= i <= length(types), type_of) ||
            throw(ArgumentError("TypedGraph: every node type must index one of the $(length(types)) types"))
        return new{T, G}(graph, types, type_of)
    end
end
function TypedGraph(graph::G, types::AbstractVector{Symbol},
                    type_of::AbstractVector{<:Integer}) where {T, G <: AbstractGraph{T}}
    return TypedGraph{T, G}(graph, collect(Symbol, types), collect(Int, type_of))
end
function TypedGraph(graph::AbstractGraph, node_types::AbstractVector{Symbol})
    types = unique(node_types)
    index = Dict(a => i for (i, a) in enumerate(types))
    return TypedGraph(graph, types, Int[index[a] for a in node_types])
end

"""
    node_types(g::TypedGraph) -> Vector{Symbol}

The type of each node of `g` (`node_types(g)[v]` is the type of node v).
"""
node_types(g::TypedGraph) = g.types[g.type_of]

"""
    nodes_of_type(g::TypedGraph, a::Symbol) -> Vector{Int}

The nodes of type `a`, in increasing order (an `ArgumentError` for an unknown type).
"""
function nodes_of_type(g::TypedGraph, a::Symbol)
    i = findfirst(==(a), g.types)
    i === nothing && throw(ArgumentError("nodes_of_type: unknown type :$a; the types are $(g.types)"))
    return findall(==(i), g.type_of)
end

# The Graphs.jl interface, forwarded to the underlying graph.
Graphs.nv(g::TypedGraph) = nv(g.graph)
Graphs.ne(g::TypedGraph) = ne(g.graph)
Graphs.vertices(g::TypedGraph) = vertices(g.graph)
Graphs.edges(g::TypedGraph) = edges(g.graph)
Graphs.edgetype(g::TypedGraph) = edgetype(g.graph)
Graphs.has_vertex(g::TypedGraph, v::Integer) = has_vertex(g.graph, v)
Graphs.has_edge(g::TypedGraph, s::Integer, d::Integer) = has_edge(g.graph, s, d)
Graphs.outneighbors(g::TypedGraph, v::Integer) = outneighbors(g.graph, v)
Graphs.inneighbors(g::TypedGraph, v::Integer) = inneighbors(g.graph, v)
Graphs.is_directed(::Type{<:TypedGraph{T, G}}) where {T, G} = is_directed(G)
Graphs.is_directed(g::TypedGraph) = is_directed(g.graph)
Graphs.add_edge!(g::TypedGraph, args...) = add_edge!(g.graph, args...)
Graphs.rem_edge!(g::TypedGraph, args...) = rem_edge!(g.graph, args...)
Base.zero(::Type{TypedGraph{T, G}}) where {T, G} = TypedGraph{T, G}(zero(G), Symbol[], Int[])
Base.copy(g::TypedGraph) = TypedGraph(copy(g.graph), copy(g.types), copy(g.type_of))
Base.:(==)(a::TypedGraph, b::TypedGraph) = a.graph == b.graph && a.types == b.types && a.type_of == b.type_of
Base.hash(g::TypedGraph, h::UInt) = hash(g.type_of, hash(g.types, hash(g.graph, hash(:TypedGraph, h))))

function Base.show(io::IO, g::TypedGraph)
    sizes = [count(==(i), g.type_of) for i in eachindex(g.types)]
    print(io, "TypedGraph{", eltype(g), "}(", nv(g), " nodes, ", ne(g), " edges; types ",
          join(("$a ($n)" for (a, n) in zip(g.types, sizes)), ", "), ")")
end

# ---------------------------------------------------------------------------------------------------------------
# The typed configuration model
# ---------------------------------------------------------------------------------------------------------------

"""
    sample_graph(net::MultitypeNetwork, N::Integer; rng = Random.default_rng()) -> (g::TypedGraph, info::GraphInfo)

A typed configuration-model graph with `N` nodes (design §C.3), returned as a [`TypedGraph`](@ref) that stores the
node types, so a stratified model seeds each stratum on its own type (design §J.6).

Type a gets N_a nodes, N·`net.sizes` rounded by largest remainders (so ΣN_a = N; an `ArgumentError` if a type would
get no node); nodes `1:N_1` have the first type, the next N_2 nodes the second, and so on. The edges are then drawn by
one of three constructions (`info.method`):

- `:unstructured`, when every type has the same `SplitDegrees(total, b => n_b, …)` law (types independent of the
  network, as built by `unstructured(net, strata)`): a graph from `sample_graph(ConfigurationNetwork(total), N)`
  (exact degrees: random regular, Erdős–Rényi or erased configuration), with the types assigned to the nodes. Every
  such graph is exchangeable, so this is exactly the configuration model with independent node types (the unit law
  M10).
- `:stochastic_block_model`, when every law is `IndependentDegrees` of `PoissonDegree` parts (`sbm_network`):
  independent Bernoulli edges, between a type-a and a type-b node with probability M[a, b]/N_b (averaged with
  M[b, a]/N_a when rounding makes the two differ) and within type a with probability M[a, a]/(N_a − 1), with
  M = `mean_contacts(net)`. The typed degrees are binomial, the binomial approximation of the Poisson laws (as for
  `ConfigurationNetwork(PoissonDegree(μ))`), and reciprocity holds exactly by construction.
- `:typed_configuration`, for any other law (typed stub matching): each type-a node draws its edge counts
  (k_{a→b})_b from `net.degrees[a]`, and the stubs of each unordered block {a, b} are paired uniformly at random.
  A block's stub totals S[a, b] = Σ_{v of type a} k_{a→b}(v) and S[b, a] agree only in expectation (reciprocity
  n_a E[k_{a→b}] = n_b E[k_{b→a}]), and a within-type total S[a, a] must be even. When both sides of a block have
  their own law (`IndependentDegrees` parts), the counts are **conditioned** on these constraints: the counts of
  uniformly chosen nodes on either side are redrawn, keeping a redraw only if it brings the totals closer, until the
  totals agree (a within-type total is made even by redrawing one node at a time, as for the configuration model),
  and then 10 Metropolis–Hastings moves per node (redraw k_{a→b}(v) from its law, move the difference to a uniformly
  chosen node u on the other side, accept with probability min(1, P(k'_{b→a}(u))/P(k_{b→a}(u)))) relax the counts to
  the product law conditioned on equal totals, so each node's law is exact up to O(1/N). Where this is impossible
  (degenerate laws whose totals cannot be changed, or `SplitDegrees` laws, whose split couples the blocks of a node),
  the |S[a, b] − S[b, a]| = O(√N) excess stubs of the larger side, chosen uniformly, are **trimmed** instead, as is
  the odd stub of a within-type block (`info.trimmed_blocks`, `info.trimmed_stubs`, `info.trimmed_fraction`).
  Trimming lowers the mean degrees of the block by O(1/√N) (at N = 10⁴ about 1% of the stubs of a block between two
  Poisson(2)-like laws), and the final size by as much as that shifts the edge-based limit. The matched totals are
  always exactly reciprocal (`info.matched_stubs` is symmetric).

In every case self-loops and repeated edges are erased (`info.erased`, `info.erased_fraction`), and `info` holds the
realised edges per block (`info.block_edges`) and mean contacts (`info.mean_contacts`), next to
`info.target_mean_contacts = mean_contacts(net)` (see [`GraphInfo`](@ref)).

```julia
net = sbm_network([:a, :b], [0.5, 0.5]; mean_contacts = [6 2; 2 4])
g, info = sample_graph(net, 10_000; rng = NetworkOutbreaks.stable_rng(1))
info.mean_contacts            # ≈ [6 2; 2 4]
nodes_of_type(g, :b)          # the nodes 5001, …, 10000
```
"""
function sample_graph(net::MultitypeNetwork, N::Integer; rng::AbstractRNG = Random.default_rng())
    N = _gen_check_N(N)
    K = length(net.types)
    sizes = _gen_type_sizes(net, N)
    type_of = Vector{Int}(undef, N)
    ranges = Vector{UnitRange{Int}}(undef, K)
    start = 1
    for i in 1:K
        ranges[i] = start:(start + sizes[i] - 1)
        type_of[ranges[i]] .= i
        start += sizes[i]
    end

    total = _gen_unstructured_total(net)
    if total !== nothing
        g, inner = sample_graph(ConfigurationNetwork(total), N; rng)
        method, candidates, erased = :unstructured, inner.candidate_edges, inner.erased
        totals = matched = nothing
        trimmed_blocks = Tuple{Symbol, Symbol}[]
    elseif _gen_is_poisson_sbm(net)
        g = _gen_sbm_graph(net, sizes, ranges, rng)
        method, candidates, erased = :stochastic_block_model, 0, 0
        totals = matched = nothing
        trimmed_blocks = Tuple{Symbol, Symbol}[]
    else
        g, candidates, erased, totals, matched, trimmed_blocks = _gen_typed_stub_graph(net, N, ranges, rng)
        method = :typed_configuration
    end

    block = zeros(Int, K, K)
    for e in edges(g)
        a, b = type_of[src(e)], type_of[dst(e)]
        block[a, b] += 1
        a == b || (block[b, a] += 1)
    end
    ends = [i == j ? 2block[i, i] : block[i, j] for i in 1:K, j in 1:K]      # edge ends at type i towards type j
    if totals === nothing                      # no stub matching per block: the stubs are the realised edge ends
        totals = matched = ends
    end
    trimmed = sum(totals) - sum(matched)
    details = (types = copy(net.types), type_sizes = sizes, stub_totals = totals, matched_stubs = matched,
               trimmed_blocks, trimmed_stubs = trimmed,
               trimmed_fraction = sum(totals) == 0 ? 0.0 : trimmed / sum(totals), block_edges = block,
               mean_contacts = ends ./ sizes, target_mean_contacts = Float64.(mean_contacts(net)))
    tg = TypedGraph(g, copy(net.types), type_of)
    return tg, _gen_info(net, method, g, candidates, erased; details)
end

# Typed stub matching. Returns (graph, stub pairs formed, erased, stub totals, matched stubs, trimmed blocks).
function _gen_typed_stub_graph(net::MultitypeNetwork, N::Int, ranges::Vector{UnitRange{Int}}, rng::AbstractRNG)
    K = length(net.types)
    index = Dict(a => i for (i, a) in enumerate(net.types))
    # counts[a, b][q]: edges from the q-th node of type a towards type b; law[a, b]: its own law when k_{a→b} is
    # drawn independently of the node's other counts (an IndependentDegrees part), else `nothing`.
    counts = [zeros(Int, length(ranges[i])) for i in 1:K, _ in 1:K]
    law = Matrix{Union{Nothing, DegreeDistribution}}(nothing, K, K)
    for i in 1:K
        m = net.degrees[i]
        partners = Int[index[b] for b in partner_types(m)]
        if m isa IndependentDegrees
            for (p, (_, d)) in zip(partners, m.parts)
                law[i, p] = d
            end
        end
        s = _gen_mv_sampler(m)
        c = zeros(Int, length(partners))
        for q in eachindex(ranges[i])
            _gen_draw_counts!(c, rng, s)
            for (p, j) in enumerate(partners)
                counts[i, j][q] = c[p]
            end
        end
    end

    # Condition each block on its constraint where both sides have their own law; trim otherwise.
    trimmed_blocks = Tuple{Symbol, Symbol}[]
    for i in 1:K, j in i:K
        if i == j
            ok = !isodd(sum(counts[i, i])) || (law[i, i] !== nothing && _gen_fix_parity!(counts[i, i], law[i, i], rng))
        else
            ok = sum(counts[i, j]) == sum(counts[j, i]) ||
                 (law[i, j] !== nothing && law[j, i] !== nothing &&
                  _gen_condition_totals!(counts[i, j], counts[j, i], law[i, j], law[j, i], rng))
        end
        ok || push!(trimmed_blocks, (net.types[i], net.types[j]))
    end
    totals = [sum(counts[i, j]) for i in 1:K, j in 1:K]

    # Match each block (trimming whatever excess is left).
    g = SimpleGraph(N)
    matched = zeros(Int, K, K)
    candidates = 0
    erased = 0
    stubs(i, j) = _gen_shuffle!(rng, _gen_append_stubs!(sizehint!(Int[], totals[i, j]), ranges[i], counts[i, j]))
    for i in 1:K, j in i:K
        if i == j
            pairs, e = _gen_pair_within!(g, stubs(i, i))
            matched[i, i] = 2pairs
        else
            pairs, e = _gen_pair_across!(g, stubs(i, j), stubs(j, i))
            matched[i, j] = matched[j, i] = pairs
        end
        candidates += pairs
        erased += e
    end
    return g, candidates, erased, totals, matched, trimmed_blocks
end

# Make the sum of the iid counts `x` (law `d`) even by redrawing one uniformly chosen count at a time (as the
# configuration model fixes the parity of its degree sum). Returns false if that fails (e.g. all mass on odd values).
function _gen_fix_parity!(x::Vector{Int}, d::DegreeDistribution, rng::AbstractRNG)
    isempty(x) && return iseven(sum(x))
    p = degree_probabilities(d)
    length(unique(isodd(k - 1) for k in eachindex(p) if p[k] > 0)) == 2 || return iseven(sum(x))  # one parity only
    s = _gen_sampler(d)
    total = sum(x)
    for _ in 1:_GEN_MAX_PARITY_REDRAWS
        iseven(total) && return true
        v = rand(rng, 1:length(x))
        new = _gen_draw(rng, s)
        total += new - x[v]
        x[v] = new
    end
    return iseven(total)
end

_gen_pmf(p::Vector{Float64}, k::Int) = 0 <= k < length(p) ? p[k + 1] : 0.0
_gen_support_gcd(p::Vector{Float64}) = reduce(gcd, diff(findall(>(0), p)); init = 0)

# Condition the iid counts x (law dx) and y (law dy) on sum(x) == sum(y): (1) redraw uniformly chosen counts on
# either side, keeping a redraw only if it brings the totals closer, until they agree; (2) 10 Metropolis–Hastings
# moves per count on the set {sum(x) == sum(y)} whose stationary law is the product law conditioned on equal totals.
# A move redraws one count from its law and moves the difference to a uniformly chosen count on the other side,
# accepted with the ratio of the other side's probabilities (the proposal is the law itself, so the laws of the
# redrawn count cancel). Returns false, leaving the counts to be trimmed, if the totals cannot be made equal.
function _gen_condition_totals!(x::Vector{Int}, y::Vector{Int}, dx::DegreeDistribution, dy::DegreeDistribution,
                                rng::AbstractRNG)
    (isempty(x) || isempty(y)) && return sum(x) == sum(y)
    px = Vector{Float64}(degree_probabilities(dx))
    py = Vector{Float64}(degree_probabilities(dy))
    D = sum(x) - sum(y)
    # A redraw changes D by a difference of two support points, so D stays in D + gℤ, g the gcd of those differences
    # (g = 0 when neither law can change its total).
    g = gcd(_gen_support_gcd(px), _gen_support_gcd(py))
    (g == 0 ? D == 0 : D % g == 0) || return false
    sx, sy = _gen_sampler(dx), _gen_sampler(dy)
    nx, ny = length(x), length(y)
    tries = 0
    while D != 0
        tries += 1
        tries <= 1000 * (nx + ny) || return false
        if rand(rng) < 0.5
            v = rand(rng, 1:nx)
            new = _gen_draw(rng, sx)
            D2 = D + new - x[v]
            abs(D2) < abs(D) && (x[v] = new; D = D2)
        else
            u = rand(rng, 1:ny)
            new = _gen_draw(rng, sy)
            D2 = D - (new - y[u])
            abs(D2) < abs(D) && (y[u] = new; D = D2)
        end
    end
    for _ in 1:(10 * (nx + ny))
        v = rand(rng, 1:nx)
        u = rand(rng, 1:ny)
        if rand(rng) < 0.5                    # redraw x[v] from its law; y[u] absorbs the difference
            new = _gen_draw(rng, sx)
            y2 = y[u] + new - x[v]
            y2 == y[u] && continue
            ratio = _gen_pmf(py, y2) / _gen_pmf(py, y[u])
            (ratio >= 1 || rand(rng) < ratio) && (x[v] = new; y[u] = y2)
        else                                  # redraw y[u] from its law; x[v] absorbs the difference
            new = _gen_draw(rng, sy)
            x2 = x[v] + new - y[u]
            x2 == x[v] && continue
            ratio = _gen_pmf(px, x2) / _gen_pmf(px, x[v])
            (ratio >= 1 || rand(rng) < ratio) && (y[u] = new; x[v] = x2)
        end
    end
    return true
end

# The common total-degree law if every type has SplitDegrees(total, b => n_b, …) with the same total and weights equal
# to the type sizes (types independent of the network); otherwise `nothing`.
function _gen_unstructured_total(net::MultitypeNetwork)
    all(m -> m isa SplitDegrees, net.degrees) || return nothing
    total = first(net.degrees).total
    sizes = Dict(zip(net.types, net.sizes))
    for m in net.degrees
        m.total == total || return nothing
        w = Dict(m.weights)
        keys(w) == keys(sizes) || return nothing
        all(isapprox(w[b], sizes[b]; rtol = 1e-12, atol = 1e-15) for b in keys(sizes)) || return nothing
    end
    return total
end

_gen_is_poisson_sbm(net::MultitypeNetwork) =
    all(m -> m isa IndependentDegrees && all(p -> last(p) isa PoissonDegree, m.parts), net.degrees)

# Independent Bernoulli edges per block (geometric skipping over the candidate pairs, O(N_a + N_b + edges)).
function _gen_sbm_graph(net::MultitypeNetwork, sizes::Vector{Int}, ranges::Vector{UnitRange{Int}}, rng::AbstractRNG)
    K = length(net.types)
    M = Float64.(mean_contacts(net))
    g = SimpleGraph(sum(sizes))
    for i in 1:K, j in i:K
        a, b = net.types[i], net.types[j]
        if i == j
            M[i, i] > 0 || continue
            sizes[i] >= 2 || throw(ArgumentError(
                "sample_graph: type :$a has $(sizes[i]) node, so it cannot have within-type edges (mean $(M[i, i]))"))
            p = M[i, i] / (sizes[i] - 1)
        else
            p = (M[i, j] / sizes[j] + M[j, i] / sizes[i]) / 2
        end
        p <= 1 || throw(ArgumentError(
            "sample_graph: the edge probability $(round(p; sigdigits = 4)) between types :$a and :$b exceeds 1 at " *
            "N = $(sum(sizes)); increase N"))
        p > 0 || continue
        i == j ? _gen_bernoulli_within!(g, ranges[i], p, rng) : _gen_bernoulli_across!(g, ranges[i], ranges[j], p, rng)
    end
    return g
end

# The number of failures before the first success of Bernoulli(p) trials, log1p(−p) = lq < 0, capped at `cap`.
function _gen_geometric(rng::AbstractRNG, lq::Float64, cap::Int)
    isinf(lq) && return 0                                   # p = 1
    x = log(1.0 - rand(rng)) / lq                           # 1 − rand ∈ (0, 1]
    return x >= cap ? cap : floor(Int, x)
end

function _gen_bernoulli_across!(g::SimpleGraph, ra::UnitRange{Int}, rb::UnitRange{Int}, p::Float64, rng::AbstractRNG)
    nb = length(rb)
    total = length(ra) * nb
    lq = log1p(-p)
    ℓ = -1
    while true
        ℓ += 1 + _gen_geometric(rng, lq, total)
        ℓ < total || break
        add_edge!(g, ra[ℓ ÷ nb + 1], rb[ℓ % nb + 1])
    end
    return g
end

# Pairs (u, v) with u > v of the range r, in the order (2,1), (3,1), (3,2), (4,1), …
function _gen_bernoulli_within!(g::SimpleGraph, r::UnitRange{Int}, p::Float64, rng::AbstractRNG)
    n = length(r)
    total = n * (n - 1) ÷ 2
    lq = log1p(-p)
    ℓ = -1
    row, first_of_row = 2, 0                               # pairs of row i are indices first_of_row .+ (0:i-2)
    while true
        ℓ += 1 + _gen_geometric(rng, lq, total)
        ℓ < total || break
        while ℓ >= first_of_row + row - 1
            first_of_row += row - 1
            row += 1
        end
        add_edge!(g, r[row], r[ℓ - first_of_row + 1])
    end
    return g
end

# Nodes per type: N·sizes rounded by largest remainders (ties to the earlier type), so the counts sum to N.
function _gen_type_sizes(net::MultitypeNetwork, N::Int)
    x = net.sizes .* N
    n = floor.(Int, x)
    order = sortperm(x .- n; rev = true, alg = Base.Sort.DEFAULT_STABLE)
    for q in 1:(N - sum(n))
        n[order[q]] += 1
    end
    for (a, na) in zip(net.types, n)
        na >= 1 || throw(ArgumentError(
            "sample_graph: type :$a of the MultitypeNetwork (size $(net.sizes[findfirst(==(a), net.types)])) gets " *
            "no node at N = $(N); increase N"))
    end
    return n
end

# Samplers for the joint law of (k_{a→b})_b, drawing counts aligned with partner_types(m).
struct _GenIndependentSampler{S}
    parts::Vector{S}
end
struct _GenSplitSampler{S}
    total::S
    weights::Vector{Float64}
end
function _gen_mv_sampler(m::IndependentDegrees)
    parts = [_gen_sampler(d) for (_, d) in m.parts]
    return _GenIndependentSampler{eltype(parts)}(parts)
end
_gen_mv_sampler(m::SplitDegrees) = _GenSplitSampler(_gen_sampler(m.total), Float64[w for (_, w) in m.weights])
_gen_mv_sampler(m::MultivariateDegree) = throw(ArgumentError(
    "sample_graph: no sampler for the multivariate degree law $(typeof(m))"))

function _gen_draw_counts!(counts::Vector{Int}, rng::AbstractRNG, s::_GenIndependentSampler)
    for p in eachindex(s.parts)
        counts[p] = _gen_draw(rng, s.parts[p])
    end
    return counts
end

# The total degree split multinomially: sequential binomial draws (each a sum of Bernoulli trials), so a
# zero-weight partner type never gets a stub.
function _gen_draw_counts!(counts::Vector{Int}, rng::AbstractRNG, s::_GenSplitSampler)
    fill!(counts, 0)
    isempty(counts) && return counts
    left = _gen_draw(rng, s.total)
    wleft = sum(s.weights)
    last_positive = findlast(>(0), s.weights)
    last_positive === nothing && return counts
    for p in eachindex(s.weights)
        left == 0 && break
        w = s.weights[p]
        w > 0 || continue
        if p == last_positive
            counts[p] = left
            break
        end
        q = min(1.0, w / wleft)
        c = 0
        for _ in 1:left
            rand(rng) < q && (c += 1)
        end
        counts[p] = c
        left -= c
        wleft -= w
    end
    return counts
end

# ---------------------------------------------------------------------------------------------------------------
# Seeding on typed graphs (design §J.6; the hook `_seed_nodes!` of src/spec.jl)
# ---------------------------------------------------------------------------------------------------------------

function _seed_nodes!(state::Vector{Int}, model::OutbreakModel, seed::SeedSpec, network::StaticNetwork{<:TypedGraph},
                      rng::AbstractRNG)
    tg = network.graph
    isempty(model.strata) && return _seed_nodes!(state, model, seed, tg.graph, rng)    # an unstratified model
    _check_seed_compartments(model, seed)
    return _gen_typed_seed!(state, model, seed, tg, rng)
end

# The compartment that the unseeded nodes of each type start in: the model's first susceptible compartment of that
# type's stratum, or `default` when it belongs to no stratum.
function _gen_typed_backgrounds(model::OutbreakModel, seed::SeedSpec, tg::TypedGraph)
    strata = Set(values(model.strata))
    strata == Set(tg.types) || throw(ArgumentError(
        "the strata of the model :$(model.name) ($(join(sort!(collect(strata)), ", "))) are not the node types " *
        "of the TypedGraph ($(join(tg.types, ", "))); stratify the model over the network's types"))
    if seed.default !== nothing
        haskey(model.strata, seed.default) && throw(ArgumentError(
            "$(nameof(typeof(seed))): on a TypedGraph the unseeded nodes of each type start in the susceptible " *
            "compartment of their own stratum; `default = :$(seed.default)` belongs to stratum " *
            ":$(model.strata[seed.default]) only. Omit `default`"))
        return fill(model.index_of[seed.default], length(tg.types))
    end
    bgs = Vector{Int}(undef, length(tg.types))
    for (i, a) in enumerate(tg.types)
        j = findfirst(X -> get(model.strata, X, nothing) === a, model.susceptible)
        j === nothing && throw(ArgumentError(
            "the model :$(model.name) has no susceptible compartment in stratum :$a, so the unseeded nodes of type " *
            ":$a have no compartment to start in; pass `default`"))
        bgs[i] = model.index_of[model.susceptible[j]]
    end
    return bgs
end

_gen_without_default(seed::SeedFraction) = SeedFraction(seed.fractions, nothing)
_gen_without_default(seed::SeedCount) = SeedCount(seed.counts, nothing)

function _gen_typed_seed!(state::Vector{Int}, model::OutbreakModel, seed::Union{SeedFraction, SeedCount},
                          tg::TypedGraph, rng::AbstractRNG)
    n = length(state)
    bgs = _gen_typed_backgrounds(model, seed, tg)
    # Counts are fractions of all n nodes (design §J.6), placed on the nodes of the compartment's stratum.
    counts = seed_counts(_gen_without_default(seed), n)
    fill!(state, 0)
    free = Vector{Vector{Int}}(undef, length(tg.types))
    for (i, a) in enumerate(tg.types)
        nodes = _gen_shuffle!(rng, findall(==(i), tg.type_of))
        cursor = 0
        for (X, c) in counts
            get(model.strata, X, nothing) === a || continue
            cursor + c <= length(nodes) || throw(ArgumentError(
                "$(nameof(typeof(seed))) places $(cursor + c) nodes in compartments of stratum :$a, but the " *
                "TypedGraph has only $(length(nodes)) nodes of type :$a (seed fractions are fractions of all N = " *
                "$(n) nodes, design §J.6)"))
            idx = model.index_of[X]
            for q in (cursor + 1):(cursor + c)
                state[nodes[q]] = idx
            end
            cursor += c
        end
        free[i] = nodes[(cursor + 1):end]
    end
    # Compartments without a stratum take nodes of any type, uniformly among the unseeded ones.
    loose = [(X, c) for (X, c) in counts if !haskey(model.strata, X)]
    if !isempty(loose)
        pool = _gen_shuffle!(rng, reduce(vcat, free; init = Int[]))
        need = sum(last, loose)
        need <= length(pool) || throw(ArgumentError(
            "$(nameof(typeof(seed))) seeds more nodes than the TypedGraph has ($(n))"))
        cursor = 0
        for (X, c) in loose
            idx = model.index_of[X]
            for q in (cursor + 1):(cursor + c)
                state[pool[q]] = idx
            end
            cursor += c
        end
    end
    for v in 1:n
        state[v] == 0 && (state[v] = bgs[tg.type_of[v]])
    end
    return state
end

function _gen_typed_seed!(state::Vector{Int}, model::OutbreakModel, seed::SeedNodes, tg::TypedGraph, rng::AbstractRNG)
    n = length(state)
    seed_counts(seed, n)                                     # validates the node ranges
    bgs = _gen_typed_backgrounds(model, seed, tg)
    for v in 1:n
        state[v] = bgs[tg.type_of[v]]
    end
    for (X, nodes) in seed.assignments
        a = get(model.strata, X, nothing)
        idx = model.index_of[X]
        for v in nodes
            a === nothing || tg.types[tg.type_of[v]] === a || throw(ArgumentError(
                "SeedNodes assigns node $v (type :$(tg.types[tg.type_of[v]])) to $X, a compartment of stratum :$a"))
            state[v] = idx
        end
    end
    return state
end
