# Owner: WP25 (DESIGN_NetworkEpiCore.md §G.2 WP25, §C.3).
#
# sample_graph for ConfigurationNetwork: random_regular_graph, erdos_renyi for Poisson, erased stub matching with
# GraphInfo (replaces the fallback of src/convenience.jl by dispatch), and for ExplicitGraph.

"""
    sample_graph(net::ConfigurationNetwork, N::Integer; rng = Random.default_rng()) -> (g::SimpleGraph, info::GraphInfo)

A configuration-model graph with `N` nodes and the degree law `net.degrees` (design §C.3):

- `RegularDegree(k)`: a uniform random k-regular simple graph (`Graphs.random_regular_graph`; `N·k` must be even
  and k < N), `info.method == :random_regular`;
- `PoissonDegree(μ)`: the Erdős–Rényi graph G(N, μ/(N − 1)) (`Graphs.erdos_renyi`), whose degrees are
  Binomial(N − 1, μ/(N − 1)), the binomial approximation of Poisson(μ) (the total-variation distance is at most
  μ/(N − 1)); `info.method == :erdos_renyi`;
- any other law (binomial, negative binomial, power law, empirical, mixture): the **erased configuration model**,
  `info.method == :erased_configuration`. The N degrees are drawn iid from the law; while their sum is odd, the degree
  of one uniformly chosen node is redrawn (an `ArgumentError` after 10 000 redraws, e.g. for a law on odd degrees
  with N odd); the stubs are paired uniformly at random, and self-loops and repeated edges are erased
  (`info.erased`, `info.erased_fraction`). If the largest possible degree (the largest drawn degree for an
  unbounded law) exceeds the structural cutoff √(N⟨k⟩), multi-edges are no longer rare and the erased graph is
  disassortative: `info.cutoff_exceeded` is set and a warning is logged (once per session).

The realised mean and excess degrees are in `info` (see [`GraphInfo`](@ref)). Reproducible for a given `rng` state,
e.g. `rng = NetworkOutbreaks.stable_rng(s)`.

```julia
g, info = sample_graph(ConfigurationNetwork(EmpiricalDegree(Dict(2 => 5/6, 10 => 1/6))), 10_000;
                       rng = NetworkOutbreaks.stable_rng(1))
info.mean_degree, info.excess_degree, info.erased_fraction      # ≈ (10/3, 5, 5e-4)
```
"""
function sample_graph(net::ConfigurationNetwork, N::Integer; rng::AbstractRNG = Random.default_rng())
    N = _gen_check_N(N)
    d = net.degrees
    d isa RegularDegree && return _gen_regular_graph(net, d.k, N, rng)
    d isa PoissonDegree && return _gen_poisson_graph(net, d, N, rng)
    return _gen_erased_configuration(net, d, N, rng)
end

function _gen_regular_graph(net::ConfigurationNetwork, k::Int, N::Int, rng::AbstractRNG)
    k <= N - 1 || throw(ArgumentError("sample_graph: no $(k)-regular simple graph on N = $(N) nodes (k must be < N)"))
    iseven(N * k) || throw(ArgumentError("sample_graph: no $(k)-regular graph on N = $(N) nodes (N·k is odd)"))
    g = k == 0 ? SimpleGraph(N) : random_regular_graph(N, k; rng)
    return g, _gen_info(net, :random_regular, g, 0, 0)
end

function _gen_poisson_graph(net::ConfigurationNetwork, d::PoissonDegree, N::Int, rng::AbstractRNG)
    μ = Float64(mean_degree(d))
    if N == 1
        g = SimpleGraph(1)
        return g, _gen_info(net, :erdos_renyi, g, 0, 0; details = (edge_probability = 0.0,))
    end
    pe = μ / (N - 1)
    pe <= 1 || throw(ArgumentError(
        "sample_graph: PoissonDegree($(μ)) needs N ≥ $(ceil(Int, μ) + 1) nodes; got N = $(N)"))
    g = erdos_renyi(N, pe; rng)
    return g, _gen_info(net, :erdos_renyi, g, 0, 0; details = (edge_probability = pe,))
end

const _GEN_MAX_PARITY_REDRAWS = 10_000

function _gen_erased_configuration(net::ConfigurationNetwork, d::DegreeDistribution, N::Int, rng::AbstractRNG)
    s = _gen_sampler(d)
    k = _gen_draw_degrees(rng, s, N)
    redraws = 0
    while isodd(sum(k))
        redraws < _GEN_MAX_PARITY_REDRAWS || throw(ArgumentError(
            "sample_graph: could not make the degree sum even on N = $(N) nodes after $(_GEN_MAX_PARITY_REDRAWS) " *
            "redraws; does $(d) put all its mass on odd degrees with N odd?"))
        redraws += 1
        k[rand(rng, 1:N)] = _gen_draw(rng, s)
    end
    stubs = _gen_append_stubs!(sizehint!(Int[], sum(k)), 1:N, k)
    _gen_shuffle!(rng, stubs)
    g = SimpleGraph(N)
    pairs, erased = _gen_pair_within!(g, stubs)
    kmax = maximum(k; init = 0)
    cutoff = sqrt(N * Float64(mean_degree(d)))
    bound = something(_gen_support_max(d), kmax)
    exceeded = bound > cutoff
    exceeded && @warn("sample_graph: the largest degree $(bound) of $(d) exceeds the structural cutoff " *
                      "√(N⟨k⟩) = $(round(cutoff; sigdigits = 4)) at N = $(N): multi-edges are no longer rare, so " *
                      "the erased graph loses edges among hubs and is disassortative (see info.erased_fraction)",
                      maxlog = 1)
    details = (parity_redraws = redraws, max_degree = kmax, structural_cutoff = cutoff, cutoff_exceeded = exceeded)
    return g, _gen_info(net, :erased_configuration, g, pairs, erased; details)
end

"""
    sample_graph(net::ExplicitGraph, N::Integer; rng = Random.default_rng()) -> (net.graph, info::GraphInfo)

The explicit graph itself (the same object; `rng` is not used), with `info.method == :explicit`. `N` must equal its
number of nodes.
"""
function sample_graph(net::ExplicitGraph, N::Integer; rng::AbstractRNG = Random.default_rng())
    g = net.graph
    g isa AbstractGraph || throw(ArgumentError("sample_graph: the ExplicitGraph does not hold a Graphs.AbstractGraph"))
    nv(g) == N || throw(ArgumentError("sample_graph: the ExplicitGraph has $(nv(g)) nodes, not N = $(N)"))
    return g, _gen_info(net, :explicit, g, 0, 0)
end
