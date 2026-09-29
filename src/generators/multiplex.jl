# Owner: WP25 (DESIGN_NetworkEpiCore.md §G.2 WP25, §C.3).
#
# sample_graph for the NetworkEpiCore MultiplexNetwork descriptor: independent layer samples on one node set, as a
# MultiplexGraph.

"""
    sample_graph(net::MultiplexNetwork, N::Integer; rng = Random.default_rng())
        -> (MultiplexGraph, info::GraphInfo)

Independent samples of the layers of a multiplex descriptor on the same `N` nodes (Miller & Volz 2013), in layer
order and from the same `rng`, each with `sample_graph(layer, N; rng)`, returned as a [`MultiplexGraph`](@ref)
with every layer rate 1 (a contact acting on all layers has the same per-contact τ on each, design §C.2) and the
layers named after the descriptor's layers (so a contact on a named layer acts on its layer).
`info.method == :multiplex`; `info.layers` holds `layer name => GraphInfo` for each layer, and the top-level degree
statistics are those of a node's total degree over the layers (an edge present in two layers counts twice), with
`erased` and `candidate_edges` summed over the layers.
"""
function sample_graph(net::MultiplexNetwork, N::Integer; rng::AbstractRNG = Random.default_rng())
    N = _gen_check_N(N)
    names = layer_names(net)
    graphs = SimpleGraph{Int}[]
    infos = Pair{Symbol, GraphInfo}[]
    for (name, layer) in net.layers
        g, info = sample_graph(layer, N; rng)
        push!(graphs, g)
        push!(infos, name => info)
    end
    total = zeros(Int, N)
    for g in graphs
        total .+= degree(g)
    end
    info = _gen_info(net, :multiplex, N, sum(ne, graphs), total, sum(i -> last(i).candidate_edges, infos),
                      sum(i -> last(i).erased, infos); details = (layers = infos,))
    return MultiplexGraph(graphs, ones(length(names)); names), info
end
