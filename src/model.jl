#=
model.jl

Internal `OutbreakModel` type. Adapters in extensions convert from
EdgeBasedModels' `DiseaseProgression` and NodeBasedModels'
`CompartmentalModel` into this normalized form.

A model is a finite set of compartments plus a list of transitions.
Each transition is either:

- `:infection`    — requires an infectious neighbour. Fires across an edge
                    (s, n) where node `s` is in `from` and node `n` is in
                    any infectious compartment. Per-edge rate = `rate`.
- `:spontaneous`  — fires per-node at rate `rate` for every node in
                    compartment `from`.

`rate` is a `Float64` (numeric, parameters already substituted).
=#

struct OutbreakTransition
    from::Symbol
    to::Symbol
    rate::Float64
    type::Symbol  # :infection | :spontaneous
    # For :infection: optional restriction on which infectious compartments
    # act as catalysts. Empty means "any infectious compartment".
    via::Vector{Symbol}
end

OutbreakTransition(from::Symbol, to::Symbol, rate::Real, type::Symbol;
                   via = Symbol[]) =
    OutbreakTransition(from, to, Float64(rate), type, collect(via))

struct OutbreakModel
    compartments::Vector{Symbol}
    infectious::Vector{Bool}                  # parallel to compartments
    transitions::Vector{OutbreakTransition}
    name::Symbol
    # Indexing helpers
    index_of::Dict{Symbol, Int}
end

function OutbreakModel(compartments::Vector{Symbol},
                       infectious::Vector{Bool},
                       transitions::Vector{OutbreakTransition};
                       name::Symbol = :outbreak)
    length(compartments) == length(infectious) ||
        throw(ArgumentError("compartments and infectious must have the same length"))
    allunique(compartments) ||
        throw(ArgumentError("compartments must be unique"))
    index_of = Dict(c => i for (i, c) in enumerate(compartments))
    for tr in transitions
        haskey(index_of, tr.from) ||
            throw(ArgumentError("unknown source compartment $(tr.from)"))
        haskey(index_of, tr.to)   ||
            throw(ArgumentError("unknown target compartment $(tr.to)"))
        tr.type in (:infection, :spontaneous) ||
            throw(ArgumentError("unknown transition type $(tr.type)"))
        tr.rate >= 0 ||
            throw(ArgumentError("transition rate must be non-negative; got $(tr.rate)"))
        for v in tr.via
            haskey(index_of, v) ||
                throw(ArgumentError("unknown via compartment $(v) in transition $(tr.from)→$(tr.to)"))
        end
    end
    return OutbreakModel(compartments, infectious, transitions, name, index_of)
end

ncompartments(m::OutbreakModel) = length(m.compartments)
infectious_indices(m::OutbreakModel) = findall(m.infectious)

"""
    OutbreakModel(compartments, infectious_set, transitions; name)

Convenience constructor where `infectious_set` is an iterable of
compartment names that should be flagged as infectious.
"""
function OutbreakModel(compartments::AbstractVector{Symbol},
                       infectious_set,
                       transitions::AbstractVector{OutbreakTransition};
                       name::Symbol = :outbreak)
    inf_flags = [c in infectious_set for c in compartments]
    return OutbreakModel(collect(compartments), inf_flags,
                         collect(transitions); name = name)
end
