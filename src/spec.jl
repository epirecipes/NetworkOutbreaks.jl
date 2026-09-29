#=
spec.jl (owner: WP16)

`OutbreakSpec` bundles the model, the contact network, the seeding and the time span; it is the input of
`simulate(spec)`. The seeding types `SeedSpec`, `SeedFraction`, `SeedCount` and `SeedNodes` are NetworkEpiCore's
(re-exported); this file places them on nodes:

- the integer counts are NetworkEpiCore's `seed_counts` (n_X = ρ_X N rounded to the nearest integer, ties away from
  zero; an error, never a silent zero seed, if a positive ρ_X rounds to 0 or the counts exceed N; verified issue N01);
- the nodes of each compartment are chosen uniformly at random, without replacement and disjointly;
- the unseeded nodes go to the background compartment chosen by `_seed_background`;
- `_seed_nodes!` is the hook for networks that record node types (design §J.6): on the untyped containers of
  network.jl it refuses a background that would lump the unseeded nodes of several strata into one stratum.
=#

"""
    OutbreakSpec(; model, network, initial, tspan)
    OutbreakSpec(model, network, initial, tspan)

A simulation input: an [`OutbreakModel`](@ref), a contact `network` (a `Graphs.AbstractGraph`, wrapped as a
[`StaticNetwork`](@ref), or an [`AbstractContactNetwork`](@ref)), the seeding `initial` (a NetworkEpiCore
`SeedSpec`) and the time span `tspan = (t0, t1)` with t0 ≤ t1 (t1 may be `Inf`).

The seeding is applied at the start of every run (see [`initial_state`](@ref)):
- `SeedFraction(:I => ρ, …)`: `seed_counts` nodes, n_X = ρ_X N rounded to the nearest integer with ties away from
  zero; a positive ρ_X that rounds to 0 nodes, or counts that exceed N, are an `ArgumentError`;
- `SeedCount(:I => n, …)`: exactly n_X nodes;
- `SeedNodes(:I => [1, 17], …)`: the listed nodes.

Seeded nodes are chosen uniformly at random without replacement, disjointly across compartments. Every other node
starts in the **background** compartment:
1. the specification's `default` when it is set;
2. when the specification covers the whole population (fractions summing to 1, counts summing to N, or every node
   listed), its first named susceptible compartment, which absorbs the rounding (the model's first susceptible
   compartment if it names none);
3. otherwise the model's first susceptible compartment that the specification does not name;
4. if it names every susceptible compartment, the remaining nodes go to the first compartment that is neither
   susceptible, infected nor named, in compartment order (R in SIR and SEIR, but the traced class Q in a tracing
   model whose Q precedes R), and it is an `ArgumentError` if there is none.
So `SeedFraction(:S => 0.9, :I => 0.1)` puts the rounding into S (also in a model with a second susceptible class,
such as aware susceptibles Sa, which then start empty), while `SeedFraction(:S => 0.5, :I => 0.1)` puts the remaining
40% into R. Pass `default` to choose the background explicitly.

**Stratified models.** On a network whose nodes carry no types (a `Graphs.AbstractGraph`, `StaticNetwork`,
`MultiplexGraph` or `TimeVaryingNetwork`), the seeding itself assigns the strata. The background rule would then put
every unseeded node into one stratum (for `stratify(sir_model(), [:a, :b])` and `SeedFraction(:I_a => 0.05)`, all of
them into S_a and none into S_b), so it is an `ArgumentError` when the compartments it chooses among (the unnamed
susceptible classes, or the compartments that would take a remainder) belong to more than one stratum of the model's
`strata`. Name the susceptible class of every stratum (`SeedFraction(:S_a => 0.5, :S_b => 0.45, :I_a => 0.05)`),
place the nodes with `SeedNodes`, or pass `default`. Seeding `:I_a => ρ` on the nodes of type a (design §J.6) needs a
network that records node types.

A `TimeVaryingNetwork` is validated here against `tspan[1]` (strict add/remove semantics, see
[`TimeVaryingNetwork`](@ref)).
"""
struct OutbreakSpec{N}
    model::OutbreakModel
    network::N
    initial::SeedSpec
    tspan::Tuple{Float64, Float64}
    function OutbreakSpec{N}(model::OutbreakModel, network::N, initial::SeedSpec,
                             tspan::Tuple{Float64, Float64}) where {N}
        (isnan(tspan[1]) || isnan(tspan[2]) || !isfinite(tspan[1]) || tspan[2] < tspan[1]) &&
            throw(ArgumentError("tspan must be (t0, t1) with a finite t0 and t1 ≥ t0; got $(tspan)"))
        network isa TimeVaryingNetwork && _check_tvn_updates(network, tspan[1])
        return new{N}(model, network, initial, tspan)
    end
end

_as_contact_network(network::AbstractGraph) = StaticNetwork(network)
_as_contact_network(network) = network

function OutbreakSpec(model::OutbreakModel, network, initial::SeedSpec, tspan::Tuple{<:Real, <:Real})
    net = _as_contact_network(network)
    return OutbreakSpec{typeof(net)}(model, net, initial, (Float64(tspan[1]), Float64(tspan[2])))
end

OutbreakSpec(; model::OutbreakModel, network, initial::SeedSpec, tspan::Tuple{<:Real, <:Real}) =
    OutbreakSpec(model, network, initial, tspan)

"""
    NetworkOutbreaks.initial_state(spec, rng) -> Vector{Int}

The per-node compartment indices at the start of a run: the seeding of `spec` applied with the random-number
generator `rng` (see [`OutbreakSpec`](@ref) for the rules).
"""
function initial_state(spec::OutbreakSpec, rng::AbstractRNG)
    state = Vector{Int}(undef, nv(spec.network))
    return _seed_nodes!(state, spec.model, spec.initial, spec.network, rng)
end

# The seeding of the nodes of `network` into `state`. The nodes of the network containers of network.jl carry no types,
# so the seeding decides the strata, and a background shared by several strata is refused (`_check_untyped_strata`).
# A container that records node types (design §J.6; the typed generators of src/generators/multitype.jl) adds a
# method for its own type that places the compartments of stratum a on the nodes of type a.
function _seed_nodes!(state::Vector{Int}, model::OutbreakModel, seed::SeedSpec, network, rng::AbstractRNG)
    _check_seed_compartments(model, seed)
    _check_untyped_strata(model, seed, length(state))
    return _apply_seed!(state, model, seed, length(state), rng)
end

_seed_names(seed::SeedFraction) = Symbol[first(f) for f in seed.fractions]
_seed_names(seed::SeedCount) = Symbol[first(c) for c in seed.counts]
_seed_names(seed::SeedNodes) = Symbol[first(a) for a in seed.assignments]

# Does the specification itself account for every one of the n nodes?
_seed_covers_all(seed::SeedFraction, n) = abs(sum(last, seed.fractions; init = 0.0) - 1) <= sqrt(eps(Float64))
_seed_covers_all(seed::SeedCount, n) = sum(last, seed.counts; init = 0) == n
_seed_covers_all(seed::SeedNodes, n) = sum(a -> length(last(a)), seed.assignments; init = 0) == n

function _check_seed_compartments(model::OutbreakModel, seed::SeedSpec)
    what = nameof(typeof(seed))
    for X in _seed_names(seed)
        haskey(model.index_of, X) ||
            throw(ArgumentError("unknown compartment $(X) in $(what); the model has $(model.compartments)"))
    end
    seed.default === nothing || haskey(model.index_of, seed.default) ||
        throw(ArgumentError("unknown default compartment $(seed.default) in $(what); the model has " *
                            "$(model.compartments)"))
    return nothing
end

# The background compartment of a seeding (see the OutbreakSpec docstring).
_seed_background(model::OutbreakModel, seed::SeedSpec, n::Integer) = first(_background_rule(model, seed, n))

# The background compartment and the compartments the rule chose it from (its first element): the unnamed susceptible
# compartments, or the compartments that could hold a remainder. The second is empty when the specification fixes
# the background: its `default`, or, when it covers the population, its first named susceptible compartment (which
# only absorbs the rounding).
function _background_rule(model::OutbreakModel, seed::SeedSpec, n::Integer)
    seed.default === nothing || return seed.default, Symbol[]
    named = Set(_seed_names(seed))
    sus = isempty(model.susceptible) ? _legacy_background(model) : model.susceptible
    if _seed_covers_all(seed, n)
        i = findfirst(in(named), sus)
        return (i === nothing ? first(sus) : sus[i]), Symbol[]
    end
    open = Symbol[s for s in sus if !(s in named)]
    isempty(open) || return first(open), open
    infected = _infected_mask(model)
    rest = Symbol[c for (i, c) in enumerate(model.compartments) if !(c in named || c in sus || infected[i])]
    isempty(rest) && throw(ArgumentError(
        "$(nameof(typeof(seed))) names every susceptible compartment ($(join(sus, ", "))) but does not account for " *
        "all $(n) nodes, and the model has no other compartment that is neither infected nor named to hold the rest; " *
        "make the seeding cover the population or pass `default`"))
    return first(rest), rest
end

# On a network without node types, the background must not be chosen among compartments of several strata: every
# unseeded node would start in the first of them (S_a) and none in the others (S_b).
function _check_untyped_strata(model::OutbreakModel, seed::SeedSpec, n::Integer)
    isempty(model.strata) && return nothing
    bg, pool = _background_rule(model, seed, n)
    strata = unique(Symbol[model.strata[c] for c in pool if haskey(model.strata, c)])
    length(strata) >= 2 || return nothing
    others = [c for c in pool if c !== bg && haskey(model.strata, c) && model.strata[c] !== get(model.strata, bg, nothing)]
    throw(ArgumentError(
        "$(nameof(typeof(seed))) on the stratified model :$(model.name) (strata $(join(strata, ", "))): the network " *
        "records no node types, so every unseeded node would start in $(bg) and none in $(join(others, ", ")). " *
        "Name the susceptible compartment of every stratum (e.g. SeedFraction(:$(bg) => …, :$(first(others)) => …, " *
        "…)), place the nodes with SeedNodes, or pass `default`; seeding each stratum on its own node type " *
        "(design §J.6) needs a network that records node types"))
end

# A model without infections has no susceptible compartment: its first non-infectious compartment is the background
# (the NetworkOutbreaks 0.1 rule), else its first compartment.
function _legacy_background(model::OutbreakModel)
    i = findfirst(!, model.infectious)
    return [model.compartments[something(i, 1)]]
end

function _apply_seed!(state, model::OutbreakModel, seed::Union{SeedFraction, SeedCount},
                      n::Integer, rng::AbstractRNG)
    _check_seed_compartments(model, seed)
    counts = seed_counts(seed, n; background = _seed_background(model, seed, n))
    sum(last, counts; init = 0) == n ||
        throw(ArgumentError("$(nameof(typeof(seed))) does not account for all $(n) nodes; pass `default`"))
    perm = randperm(rng, n)
    cursor = 0
    for (X, c) in counts
        idx = model.index_of[X]
        for j in 1:c
            state[perm[cursor + j]] = idx
        end
        cursor += c
    end
    return state
end

function _apply_seed!(state, model::OutbreakModel, seed::SeedNodes, n::Integer, rng::AbstractRNG)
    _check_seed_compartments(model, seed)
    seed_counts(seed, n)                     # validates the node ranges
    bg = _seed_background(model, seed, n)
    fill!(state, model.index_of[bg])
    for (X, nodes) in seed.assignments
        idx = model.index_of[X]
        for v in nodes
            state[v] = idx
        end
    end
    return state
end
