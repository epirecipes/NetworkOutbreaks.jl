# Owner: WP36a (DESIGN_NetworkEpiCore.md §K WP36a, §C.3).
#
# sample_graph for DegreeCorrelatedNetwork: the joint-degree-matrix ("2K") configuration model. The degree classes get
# N·pₖ nodes (largest remainders), the edge-end matrix e is rounded to an integer joint degree matrix with exactly the
# class stub totals as row sums, and the stubs of each class are split uniformly at random over its partner classes and
# matched block by block, so every node keeps its class degree and the number of edges between two classes is
# prescribed (Newman 2002; Stanton & Pinar 2012). The nodes carry their degree class as a node type (a TypedGraph with
# the types of `MultitypeNetwork(net)`). Tests in test/suites/joint_degree.jl.
#
# Created empty by WP16 (the owner of src/NetworkOutbreaks.jl and its include list, design §G.1); filled by WP36a.

"""
    sample_graph(net::DegreeCorrelatedNetwork, N::Integer; rng = Random.default_rng()) -> (g::TypedGraph, info::GraphInfo)

A joint-degree-matrix ("2K") configuration-model graph with `N` nodes (design §C.3), `info.method == :joint_degree`:
nodes of degree k attach to nodes of degree l in the proportions Q(l | k) = e_kl/Σ_m e_km of the edge-end matrix
`net.edge_ends`, which fixes Newman's degree assortativity (`degree_assortativity(net)`).

1. The degree classes k (`net.degrees`, the support of the degree law) get N_k nodes, N·pₖ rounded by largest
   remainders (ties to the smaller degree), so ΣN_k = N; nodes `1:N_1` are in the first class, the next N_2 in the
   second, and so on. A class may get no node when N·pₖ < 1 (whether it does depends on the rank of its remainder).
2. The edge ends are fixed as integers: m[k, l] ends at degree-k nodes lead to degree-l nodes (m symmetric, the
   diagonal even, twice the number of edges within a class), with row sums exactly the stub totals S_k = k·N_k and
   entries within O(1) of L·e_kl, L = ΣS_k. m is L·e rounded to the nearest (the diagonal to the nearest even
   number) and then balanced one edge at a time: rows that exceed S_k give up the edges that were rounded up most;
   rows short of S_k take an edge from the block with the largest remaining shortfall L·e_kl − m_kl among the classes
   that are also short (e_kl > 0), else an edge within the class (e_kk > 0), else an edge e_ij that is redirected to
   k from two of its partner classes. Blocks with e_kl = 0 (to within 10⁻¹² of the smaller of the two class edge-end
   fractions, which absorbs rounding in `degree_correlated(d; r)`) are never used. A shortfall that cannot be
   balanced — an
   odd total number of stubs, a class whose partner classes have no nodes at this N, or, for a class-bipartite e
   (e_kk = 0), stub totals that differ — is **trimmed**: that many stubs of uniformly chosen nodes of the class are
   removed (`info.trimmed_stubs`, typically 0 or 1).
3. The stubs of each class are shuffled and cut into consecutive runs of lengths m[k, 1], m[k, 2], …, so each node's
   stubs are split over the partner classes uniformly at random (multivariate hypergeometric, → multinomial with
   Q(· | k) as N → ∞). The runs (k → l) and (l → k) are paired position by position and the run (k → k) with itself,
   so each block is a uniform random matching.
4. Self-loops and repeated edges are erased (`info.erased`, `info.erased_fraction`), as in the configuration model.

The graph is a [`TypedGraph`](@ref) whose node types are the degree classes, named as in `MultitypeNetwork(net)`
(`:k2`, `:k10`, …): an unstratified model is seeded uniformly over all nodes, and a model stratified over the degree
classes (`stratify(model, [:k2, :k10])` on `MultitypeNetwork(net)`) seeds each stratum within its class (design §J.6).
A node keeps its class even when erasure or trimming lowers its realised degree.

`info.details` (also properties of `info`, see [`GraphInfo`](@ref)):

- `types`, `degrees`, `class_sizes` (N_k), `stub_totals` (S_k before trimming);
- `edge_end_counts` (the integer matrix m, after trimming, before erasure) and `target_edge_ends` (L·e);
- `realised_edge_ends`: the edge ends of the final graph, from class k to class l, counted by class (the diagonal
  twice the edges within a class), and `realised_joint_degree`, the same normalised to sum 1 (the realised e);
- `trimmed_stubs`, `trimmed_fraction` (of ΣS_k);
- `assortativity`: Newman's degree assortativity of the realised graph (the Pearson correlation of the realised
  degrees at the two ends of an edge; `NaN` if they do not vary), next to `target_assortativity =
  degree_assortativity(net)`.

Reproducible for a given `rng` state (e.g. `rng = NetworkOutbreaks.stable_rng(s)`): the only random draws are the
Fisher–Yates shuffles of the stub lists and the choice of trimmed stubs.

```julia
net = degree_correlated(EmpiricalDegree(2 => 5/6, 10 => 1/6); r = 0.5)
g, info = sample_graph(net, 10_000; rng = NetworkOutbreaks.stable_rng(1))
info.assortativity, info.target_assortativity      # ≈ (0.50, 0.5)
info.realised_joint_degree                          # ≈ net.edge_ends = [0.375 0.125; 0.125 0.375]
```
"""
function sample_graph(net::DegreeCorrelatedNetwork, N::Integer; rng::AbstractRNG = Random.default_rng())
    N = _gen_check_N(N)
    ks = net.degrees
    K = length(ks)
    types = [Symbol("k", k) for k in ks]
    sizes = _jd_class_sizes(net.probabilities, N)
    ranges = Vector{UnitRange{Int}}(undef, K)
    type_of = Vector{Int}(undef, N)
    start = 1
    for i in 1:K
        ranges[i] = start:(start + sizes[i] - 1)
        type_of[ranges[i]] .= i
        start += sizes[i]
    end

    # Stubs per node, and the integer joint degree matrix with these row sums (step 2).
    stubs_of = Vector{Int}(undef, N)
    for i in 1:K
        stubs_of[ranges[i]] .= ks[i]
    end
    S = sizes .* ks
    target = sum(S) .* net.edge_ends
    m, short = _jd_edge_end_counts(target, net.edge_ends, S)
    trimmed = 0
    for i in 1:K
        short[i] > 0 || continue
        _jd_trim_stubs!(stubs_of, ranges[i], short[i], rng)
        trimmed += short[i]
    end

    # Split each class's stubs over its partner classes (step 3) and match the blocks.
    runs = Matrix{Vector{Int}}(undef, K, K)
    for i in 1:K
        stubs = _gen_shuffle!(rng, _gen_append_stubs!(sizehint!(Int[], S[i]), ranges[i], @view stubs_of[ranges[i]]))
        length(stubs) == sum(@view m[i, :]) || error("sample_graph: internal error, class $(ks[i]) has " *
                                                     "$(length(stubs)) stubs for $(sum(@view m[i, :])) edge ends")
        pos = 0
        for j in 1:K
            runs[i, j] = stubs[(pos + 1):(pos + m[i, j])]
            pos += m[i, j]
        end
    end
    g = SimpleGraph(N)
    candidates = 0
    erased = 0
    for i in 1:K, j in i:K
        pairs, e = i == j ? _gen_pair_within!(g, runs[i, i]) : _gen_pair_across!(g, runs[i, j], runs[j, i])
        candidates += pairs
        erased += e
    end

    realised = zeros(Int, K, K)
    for ed in edges(g)
        a, b = type_of[src(ed)], type_of[dst(ed)]
        realised[a, b] += 1
        realised[b, a] += 1
    end
    total_ends = sum(realised)
    details = (types = types, degrees = copy(ks), class_sizes = sizes, stub_totals = S, edge_end_counts = m,
               target_edge_ends = target, realised_edge_ends = realised,
               realised_joint_degree = total_ends == 0 ? zeros(K, K) : realised ./ total_ends,
               trimmed_stubs = trimmed, trimmed_fraction = sum(S) == 0 ? 0.0 : trimmed / sum(S),
               assortativity = _jd_assortativity(g), target_assortativity = degree_assortativity(net))
    return TypedGraph(g, types, type_of), _gen_info(net, :joint_degree, g, candidates, erased; details)
end

# Nodes per degree class: N·p rounded by largest remainders (ties to the earlier class), summing to N.
function _jd_class_sizes(p::Vector{Float64}, N::Int)
    x = p .* N
    n = floor.(Int, x)
    order = sortperm(x .- n; rev = true, alg = Base.Sort.DEFAULT_STABLE)
    for q in 1:(N - sum(n))
        n[order[q]] += 1
    end
    return n
end

# The integer joint degree matrix of step 2: symmetric, even diagonal, zero where e is zero, row sums at most S and equal
# to S except for the returned shortfalls (the stubs to trim).
function _jd_edge_end_counts(target::Matrix{Float64}, e::Matrix{Float64}, S::Vector{Int})
    K = length(S)
    # A block is used only if e_ij > 0 beyond rounding: e = r q_k δ + (1 − r) q_k q_l can cancel on the diagonal to
    # O(eps·q_k), which must not open a block that the model leaves empty.
    q = vec(sum(e; dims = 2))
    allowed = [e[i, j] > 1e-12 * min(q[i], q[j]) for i in 1:K, j in 1:K]
    m = zeros(Int, K, K)
    for i in 1:K, j in i:K
        allowed[i, j] || continue
        if i == j
            m[i, i] = 2 * round(Int, target[i, i] / 2)
        else
            m[i, j] = m[j, i] = round(Int, target[i, j])
        end
    end
    d = S .- vec(sum(m; dims = 2))                     # d_i > 0: short of S_i; d_i < 0: over
    res = target .- m                                  # how far each block is below its target
    function give!(i, j, n)                            # add n ends to block (i, j) (and (j, i))
        m[i, j] += n
        res[i, j] -= n
        d[i] -= n
        if i != j
            m[j, i] += n
            res[j, i] -= n
            d[j] -= n
        end
    end

    # Rows over their stub totals give up the blocks that were rounded up the most.
    while (i = argmin(d); d[i] < 0)
        j = _jd_best(j -> j != i && d[j] < 0 && m[i, j] > 0, j -> -res[i, j], K)
        if j != 0
            give!(i, j, -1)
        elseif m[i, i] >= 2
            give!(i, i, -2)
        else                                           # the partner goes short and is refilled below
            j = _jd_best(j -> j != i && m[i, j] > 0, j -> -res[i, j], K)
            give!(i, j, -1)
        end
    end

    # Rows short of their stub totals take an edge from a class that is also short, one within the class, or one
    # redirected from two of their partner classes; what cannot be placed is trimmed.
    short = zeros(Int, K)
    while (i = argmax(d); d[i] > 0)
        j = _jd_best(j -> j != i && d[j] > 0 && allowed[i, j], j -> res[i, j], K)
        if j != 0
            give!(i, j, 1)
            continue
        end
        if d[i] >= 2 && allowed[i, i]
            give!(i, i, 2)
            continue
        end
        if d[i] >= 2 && _jd_redirect!(m, res, i, allowed, K)
            d[i] -= 2
            continue
        end
        short[i] = d[i]                                # nothing else can absorb these stubs
        d[i] = 0
    end
    return m, short
end

# The index j in 1:K with `ok(j)` that maximises `score(j)` (ties to the smaller j), or 0.
function _jd_best(ok, score, K::Int)
    best, bs = 0, -Inf
    for j in 1:K
        ok(j) || continue
        s = score(j)
        if best == 0 || s > bs
            best, bs = j, s
        end
    end
    return best
end

# Redirect one edge j–l (j, l ≠ i, both allowed to connect to i; j = l takes an edge within class j) into the two
# edges i–j and i–l: row i gains two ends and every other row keeps its sum. Picks the edge whose block is furthest above
# its target. Returns whether an edge was found.
function _jd_redirect!(m, res, i::Int, allowed, K::Int)
    best, bj, bl = -Inf, 0, 0
    for j in 1:K, l in j:K
        (j == i || l == i || !allowed[i, j] || !allowed[i, l]) && continue
        m[j, l] >= (j == l ? 2 : 1) || continue
        s = -res[j, l]
        if bj == 0 || s > best
            best, bj, bl = s, j, l
        end
    end
    bj == 0 && return false
    if bj == bl
        m[bj, bj] -= 2
        res[bj, bj] += 2
        m[i, bj] += 2
        m[bj, i] += 2
        res[i, bj] -= 2
        res[bj, i] -= 2
    else
        for (a, b) in ((bj, bl), (bl, bj))
            m[a, b] -= 1
            res[a, b] += 1
        end
        for c in (bj, bl), (a, b) in ((i, c), (c, i))
            m[a, b] += 1
            res[a, b] -= 1
        end
    end
    return true
end

# Remove `n` stubs of the nodes in `range`, each chosen uniformly among the stubs that are left.
function _jd_trim_stubs!(stubs_of::Vector{Int}, range::UnitRange{Int}, n::Int, rng::AbstractRNG)
    left = sum(@view stubs_of[range])
    n <= left || error("sample_graph: internal error, cannot trim $n of $left stubs")
    for _ in 1:n
        u = rand(rng, 1:left)                      # the u-th remaining stub of the class
        for v in range
            u <= stubs_of[v] && (stubs_of[v] -= 1; break)
            u -= stubs_of[v]
        end
        left -= 1
    end
    return stubs_of
end

# Newman's degree assortativity of a simple graph: the Pearson correlation of the degrees at the two ends of an edge,
# over both orientations of every edge. NaN when the end degrees do not vary (a regular graph) or there is no edge.
function _jd_assortativity(g::AbstractGraph)
    ds = degree(g)
    n = 0
    s1 = 0.0
    s2 = 0.0
    sxy = 0.0
    for e in edges(g)
        a, b = Float64(ds[src(e)]), Float64(ds[dst(e)])
        n += 2
        s1 += a + b
        s2 += a^2 + b^2
        sxy += 2a * b
    end
    n == 0 && return NaN
    μ = s1 / n
    σ2 = s2 / n - μ^2
    σ2 <= 1e-14 * max(1.0, μ^2) && return NaN
    return (sxy / n - μ^2) / σ2
end
