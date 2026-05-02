#=
state.jl

Mutable simulation state. Holds per-node compartment indices and counters
of nodes per compartment. Per-node infection counters are tracked
optionally for reinfection-counting analyses.
=#

mutable struct OutbreakState
    model::OutbreakModel
    node_state::Vector{Int}            # length nv(graph): compartment index per node
    counts::Vector{Int}                # length C: number of nodes in each compartment
    infection_counts::Vector{Int}      # length nv(graph): per-node times-infected
end

function OutbreakState(model::OutbreakModel, node_state::Vector{Int})
    n = length(node_state)
    counts = zeros(Int, ncompartments(model))
    @inbounds for v in 1:n
        idx = node_state[v]
        1 <= idx <= ncompartments(model) ||
            throw(ArgumentError("node $v has invalid state index $idx"))
        counts[idx] += 1
    end
    infection_counts = zeros(Int, n)
    # If a node starts in an infectious compartment, count that as one
    # infection (the seed event).
    @inbounds for v in 1:n
        if model.infectious[node_state[v]]
            infection_counts[v] = 1
        end
    end
    return OutbreakState(model, node_state, counts, infection_counts)
end
