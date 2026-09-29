# Owner: WP25 (DESIGN_NetworkEpiCore.md §G.2 WP25, §C.3).
#
# sample_graph for WellMixed: complete_graph(N) with rates κτ/(N − 1), for the lumping test only.

const _GEN_MAX_COMPLETE_N = 5000

"""
    sample_graph(net::WellMixed, N::Integer; rng = Random.default_rng()) -> (MultiplexGraph, info::GraphInfo)

The complete graph K_N as a one-layer [`MultiplexGraph`](@ref) whose layer rate is κ/(N − 1): a node with I
infectious neighbours then has the infection hazard τκ·I/(N − 1), i.e. mass action with β = κτ (design §C.2, §C.3),
and the node-level process on K_N lumps exactly to the count process of `MassActionSSA`. `info.method == :complete`,
and `info.rate_scale == κ/(N − 1)`. `rng` is not used (K_N is not random).

This graph exists for the lumping test: every event touches O(N) nodes and the adjacency lists take O(N²) memory, so
N is limited to $(_GEN_MAX_COMPLETE_N). Simulate well-mixed populations with the count-based sampler instead (the
well-mixed `simulate` method with `MassActionSSA`).
"""
function sample_graph(net::WellMixed, N::Integer; rng::AbstractRNG = Random.default_rng())
    N = _gen_check_N(N)
    N >= 2 || throw(ArgumentError("sample_graph: a well-mixed population needs N ≥ 2 nodes; got N = $(N)"))
    N <= _GEN_MAX_COMPLETE_N || throw(ArgumentError(
        "sample_graph: the complete graph for WellMixed is meant for the lumping test and is limited to N ≤ " *
        "$(_GEN_MAX_COMPLETE_N) (its adjacency lists take O(N²) memory); got N = $(N). Simulate well-mixed " *
        "populations with the count-based MassActionSSA"))
    net.κ isa Real || throw(ArgumentError("sample_graph: WellMixed needs a numeric κ; got $(net.κ)"))
    κ = Float64(net.κ)
    scale = κ / (N - 1)
    g = complete_graph(N)
    info = _gen_info(net, :complete, g, 0, 0; details = (κ = κ, rate_scale = scale))
    return MultiplexGraph([g], [scale]), info
end
