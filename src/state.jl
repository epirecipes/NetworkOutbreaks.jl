#=
state.jl (owner: WP16)

Mutable simulation state: per-node compartment indices, the number of nodes per compartment, and per-node
infection counts (for `final_size` and reinfection counting).
=#

"""
    OutbreakState(model, node_state)

The mutable state of a run: `node_state` (the compartment index of every node), `counts` (nodes per compartment)
and `infection_counts` (per node, the number of entries into an infected compartment). The constructor validates
the indices and counts every node that starts in one of the model's infected compartments once (so latent seeds count
too, see `final_size`).
"""
mutable struct OutbreakState
    model::OutbreakModel
    node_state::Vector{Int}            # length nv(graph): compartment index per node
    counts::Vector{Int}                # length C: number of nodes in each compartment
    infection_counts::Vector{Int}      # length nv(graph): per-node times-infected
end

function OutbreakState(model::OutbreakModel, node_state::Vector{Int})
    n = length(node_state)
    C = ncompartments(model)
    counts = zeros(Int, C)
    for v in 1:n
        idx = node_state[v]
        1 <= idx <= C ||
            throw(ArgumentError("node $v has invalid state index $idx"))
        counts[idx] += 1
    end
    infection_counts = zeros(Int, n)
    for v in 1:n
        if model.infected[node_state[v]]
            infection_counts[v] = 1
        end
    end
    return OutbreakState(model, node_state, counts, infection_counts)
end
