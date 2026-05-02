#=
spec.jl

`OutbreakSpec` bundles the model, the contact network, the initial
condition, and the time horizon. It is the main user-facing input to
`simulate`.

Seeding helpers:
- `SeedFraction(:I => p, :S => 1-p)` — sample uniformly with given
  fractions; rounds to integer counts.
- `SeedNodes(:I => [1, 17, 42])`     — explicit node assignments;
  unspecified nodes default to the first susceptible-like compartment.
=#

abstract type SeedSpec end

struct SeedFraction <: SeedSpec
    fractions::Vector{Pair{Symbol, Float64}}
end
SeedFraction(pairs::Pair{Symbol, <:Real}...) =
    SeedFraction([k => Float64(v) for (k, v) in pairs])

struct SeedNodes <: SeedSpec
    assignments::Vector{Pair{Symbol, Vector{Int}}}
    default::Union{Symbol, Nothing}
end
SeedNodes(pairs::Pair{Symbol, <:AbstractVector{<:Integer}}...; default = nothing) =
    SeedNodes([k => Int.(collect(v)) for (k, v) in pairs], default)

struct OutbreakSpec{N}
    model::OutbreakModel
    network::N
    initial::SeedSpec
    tspan::Tuple{Float64, Float64}
end

# Auto-wrap a plain AbstractGraph in StaticNetwork; accept AbstractContactNetwork as-is.
function OutbreakSpec(; model::OutbreakModel,
                      network,
                      initial::SeedSpec,
                      tspan::Tuple{<:Real, <:Real})
    net = network isa AbstractGraph ? StaticNetwork(network) : network
    return OutbreakSpec(model, net, initial, (Float64(tspan[1]), Float64(tspan[2])))
end

"""
    initial_state(spec, rng) -> Vector{Int}

Materialize a per-node compartment-index vector from the spec's seed
specification.
"""
function initial_state(spec::OutbreakSpec, rng::AbstractRNG)
    n = nv(spec.network)
    state = Vector{Int}(undef, n)
    return _apply_seed!(state, spec.model, spec.initial, n, rng)
end

function _apply_seed!(state, model::OutbreakModel, seed::SeedFraction,
                      n::Integer, rng::AbstractRNG)
    counts = Dict{Symbol, Int}()
    total_assigned = 0
    # Allocate integer counts proportionally
    sorted = sort(seed.fractions; by = x -> -x[2])
    for (sym, frac) in sorted
        haskey(model.index_of, sym) ||
            throw(ArgumentError("unknown compartment $(sym) in SeedFraction"))
        counts[sym] = round(Int, frac * n)
        total_assigned += counts[sym]
    end
    # Default fill to first non-infectious compartment, or first compartment.
    default = _default_compartment(model, keys(counts))
    counts[default] = get(counts, default, 0) + (n - total_assigned)
    counts[default] >= 0 ||
        throw(ArgumentError("seed fractions exceed 1.0"))

    perm = randperm(rng, n)
    cursor = 1
    for (sym, c) in counts
        idx = model.index_of[sym]
        @inbounds for j in 1:c
            state[perm[cursor]] = idx
            cursor += 1
        end
    end
    return state
end

function _apply_seed!(state, model::OutbreakModel, seed::SeedNodes,
                      n::Integer, rng::AbstractRNG)
    assigned = falses(n)
    for (sym, nodes) in seed.assignments
        haskey(model.index_of, sym) ||
            throw(ArgumentError("unknown compartment $(sym) in SeedNodes"))
        idx = model.index_of[sym]
        @inbounds for v in nodes
            1 <= v <= n || throw(BoundsError("node $v out of range 1:$n"))
            state[v] = idx
            assigned[v] = true
        end
    end
    default = something(seed.default,
                        _default_compartment(model, (k for (k, _) in seed.assignments)))
    default_idx = model.index_of[default]
    @inbounds for v in 1:n
        if !assigned[v]
            state[v] = default_idx
        end
    end
    return state
end

# Pick a "background" compartment: prefer the first non-infectious one not
# already enumerated in the seed; otherwise fall back to the first
# compartment.
function _default_compartment(model::OutbreakModel, named)
    named_set = Set{Symbol}(named)
    for (i, c) in enumerate(model.compartments)
        if !model.infectious[i] && !(c in named_set)
            return c
        end
    end
    return model.compartments[1]
end
