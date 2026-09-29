# Owner: WP25 (DESIGN_NetworkEpiCore.md §G.2 WP25, §C.3).
#
# sample_graph for ClusteredNetwork: the Newman–Miller construction (single stubs and triangle corners), realised
# transitivity reported.

"""
    sample_graph(net::ClusteredNetwork, N::Integer; rng = Random.default_rng()) -> (g::SimpleGraph, info::GraphInfo)

A clustered configuration-model graph with `N` nodes (Newman 2009; Miller 2009), `info.method == :newman_miller`:

1. every node draws (s, t) from `net.joint`: s single stubs and t triangle corners, so its degree is s + 2t;
2. the single stubs are paired uniformly at random (one uniformly chosen stub is trimmed if their number is odd);
3. the triangle corners are shuffled and grouped in threes, each group becoming a triangle (the 1 or 2 corners left
   over when their number is not a multiple of 3 are trimmed);
4. self-loops and repeated edges are erased, including the edges of a *degenerate* group that repeats a node.

`info` records `triangles_placed`, `degenerate_triangles`, the trimmed stubs and corners, the realised number of
`triangles` and of `connected_triples`, the realised `transitivity` = 3·triangles/connected triples and the target
`target_transitivity = clustering_coefficient(net)` = 2E[t]/E[k(k − 1)] (Volz et al. 2011), e.g. 2/15 for
`ClusteredNetwork(RegularDegree(2), RegularDegree(2))`. The realised transitivity exceeds the target by O(1/N):
single and triangle edges close O(1) accidental triangles, as in any finite configuration model.

```julia
g, info = sample_graph(ClusteredNetwork(RegularDegree(2), RegularDegree(2)), 10_000;
                       rng = NetworkOutbreaks.stable_rng(1))
info.transitivity, info.target_transitivity      # ≈ (0.1337, 0.1333)
```
"""
function sample_graph(net::ClusteredNetwork, N::Integer; rng::AbstractRNG = Random.default_rng())
    N = _gen_check_N(N)
    s, t = _gen_draw_clustered(rng, net.joint, N)
    g = SimpleGraph(N)

    singles = _gen_append_stubs!(sizehint!(Int[], sum(s)), 1:N, s)
    _gen_shuffle!(rng, singles)
    pairs, erased = _gen_pair_within!(g, singles)
    trimmed_stubs = length(singles) - 2pairs

    corners = _gen_append_stubs!(sizehint!(Int[], sum(t)), 1:N, t)
    _gen_shuffle!(rng, corners)
    groups = length(corners) ÷ 3
    trimmed_corners = length(corners) - 3groups
    degenerate = 0
    @inbounds for q in 1:groups
        a, b, c = corners[3q - 2], corners[3q - 1], corners[3q]
        (a == b || a == c || b == c) && (degenerate += 1)
        _gen_add_edge!(g, a, b) || (erased += 1)
        _gen_add_edge!(g, a, c) || (erased += 1)
        _gen_add_edge!(g, b, c) || (erased += 1)
    end

    triangles, triples = _gen_triangles_and_triples(g)
    details = (triangles_placed = groups, degenerate_triangles = degenerate, trimmed_stubs = trimmed_stubs,
               trimmed_corners = trimmed_corners, triangles = triangles, connected_triples = triples,
               transitivity = triples == 0 ? 0.0 : 3triangles / triples,
               target_transitivity = Float64(clustering_coefficient(net)))
    return g, _gen_info(net, :newman_miller, g, pairs + 3groups, erased; details)
end

# The (s, t) of N nodes.
function _gen_draw_clustered(rng::AbstractRNG, cd::ClusteredDegree{<:Tuple}, N::Int)
    ds, dt = cd.joint
    ss, st = _gen_sampler(ds), _gen_sampler(dt)
    s = Vector{Int}(undef, N)
    t = Vector{Int}(undef, N)
    for v in 1:N
        s[v] = _gen_draw(rng, ss)
        t[v] = _gen_draw(rng, st)
    end
    return s, t
end

function _gen_draw_clustered(rng::AbstractRNG, cd::ClusteredDegree{Matrix{Float64}}, N::Int)
    P = cd.joint
    table = _gen_table(vec(P))
    ci = CartesianIndices(P)
    s = Vector{Int}(undef, N)
    t = Vector{Int}(undef, N)
    for v in 1:N
        c = ci[_gen_draw(rng, table) + 1]
        s[v], t[v] = c[1] - 1, c[2] - 1
    end
    return s, t
end
