#=
model.jl (owner: WP16)

The simulation model `OutbreakModel`: a finite set of compartments, the infectious flags, the infected compartments
and a list of transitions with numeric rates. Build it directly, or from any model that NetworkEpiCore's
`contact_model` accepts (`OutbreakModel(cm::ContactModel, p)`; design §A.5), which replaces the 0.1 package
extensions.

Each transition is one of
- `:infection`: edge-mediated. It fires across an edge (v, u) where v is in `from` and u in one of the `via`
  compartments (the catalysts), at rate `rate` per such edge. An empty `via` means the infectious compartments;
  `via` compartments need not be infectious.
- `:contact_trace`: edge-mediated with exactly the same hazard, but it never counts as an infection. It is the
  status-preserving contact: tracing or quarantine of latent or infectious nodes, awareness, and every contact whose
  recipient or product is susceptible. `OutbreakModel(cm)` emits it for the contacts that the NetworkEpiCore typing
  says are not node contacts (s ∈ Σ, J ∉ Σ, X ∉ Σ).
- `:spontaneous`: node-local, at rate `rate` for every node in `from`.

Which compartments are *infected* (for `final_size`, `reinfection_histogram` and the infection counts) is stored in
the model (`infected`, design §J.8):
- for `OutbreakModel(cm)` it is NetworkEpiCore's `infected_species(cm)`, the single implementation of §J.8 shared
  with the edge-based and pairwise back ends (design §L.6);
- for a hand-built model it is the structural rule `_structural_infected` below, unless the `infected` keyword
  names the compartments explicitly (for the structurally ambiguous cases in its docstring).
=#

"""
    OutbreakTransition(from, to, rate, type; via = Symbol[], layer = :all)

A transition of an [`OutbreakModel`](@ref): a node in compartment `from` moves to `to` at the numeric rate
`rate ≥ 0`.

- `type = :infection`: an edge-mediated contact. The hazard of a node v in `from` is
  `rate × #{neighbours u of v : state(u) ∈ via}` (layer-weighted on a [`MultiplexGraph`](@ref)). An empty `via`
  means the infectious compartments of the model; `via` compartments need not be infectious (tracing from diagnosed
  nodes, peer vaccination, awareness). Whether it is an *infection* is decided by the model's infected compartments
  (see [`OutbreakModel`](@ref)): a node that moves from a non-infected into an infected compartment is infected.
- `type = :contact_trace`: an edge-mediated contact with the same hazard that never counts as an infection in the
  structural rule of hand-built models (tracing or quarantine of latent or infectious nodes, awareness).
  `OutbreakModel(cm::ContactModel, p)` uses it for every contact that is not a node contact of the NetworkEpiCore
  typing.
- `type = :spontaneous`: node-local, at rate `rate` per node in `from` (`via` and `layer` are ignored).

`layer` restricts an edge-mediated transition to one layer of a [`MultiplexGraph`](@ref), named as in its `names`
(`sample_graph(::MultiplexNetwork, N)` names the layers after the descriptor's layers): the hazard then counts only
the catalyst neighbours linked in that layer, weighted by its layer rate. The default `:all` acts on every layer
(and is the only value allowed on a network that is not a `MultiplexGraph`).
"""
struct OutbreakTransition
    from::Symbol
    to::Symbol
    rate::Float64
    type::Symbol  # :infection | :contact_trace | :spontaneous
    via::Vector{Symbol}
    layer::Symbol # :all, or the name of a MultiplexGraph layer
end

OutbreakTransition(from::Symbol, to::Symbol, rate::Real, type::Symbol; via = Symbol[], layer::Symbol = :all) =
    OutbreakTransition(from, to, Float64(rate), type, collect(Symbol, via), layer)

"""
    OutbreakModel(compartments, infectious, transitions; name = :outbreak, susceptible = nothing, strata = nothing,
                  infected = nothing)
    OutbreakModel(compartments, infectious_set, transitions; name, susceptible, strata, infected)
    OutbreakModel(model, p = Dict{Symbol,Float64}(); network = nothing, name, infected = nothing)

The simulation model: `compartments` (unique names), their `infectious` flags (a `Vector{Bool}` in the same order,
or any collection `infectious_set` of the infectious names) and the [`OutbreakTransition`](@ref)s with numeric rates.

`susceptible` lists the susceptible compartments Σ (the classes a node is in before it is infected). It is inferred
when not given: the non-infectious compartments that are the source of an `:infection` transition, in compartment
order. It decides which compartment receives the unseeded nodes (see [`OutbreakSpec`](@ref)).

`strata` maps compartments to the stratum they belong to (a `Dict{Symbol,Symbol}` or pairs `X => a`); it is empty
unless given. It does not change the dynamics: it lets the seeding refuse to put the unseeded nodes of several strata
into one of them on a network without node types (see [`OutbreakSpec`](@ref)).

`infected` lists the compartments counted as *infected* (design §J.8): a node's infection count goes up each time it
moves from a non-infected into an infected compartment (by a contact, a node-local transition or an intervention),
a node that starts in an infected compartment counts once, and `final_size` is the fraction of nodes with a positive
count. When it is not given it is decided structurally from the transitions (`_structural_infected`: latent and
infectious compartments are infected; recovered, vaccinated, traced and aware ones are not). Give it for the cases
the structural rule cannot tell apart, e.g. peer vaccination `S → V` via `V`, where `V` is a catalyst but not
infected: `infected = [:I]`. A susceptible compartment is never infected.

# From a model description (design §A.5)

`OutbreakModel(model, p)` converts anything that `contact_model` accepts (a `ContactModel` such as
`sir_model()`, a Catalyst `ReactionSystem`, the legacy EdgeBasedModels and NodeBasedModels types) with the
parameter values `p` (a `Dict{Symbol}`; the model's defaults fill the rest):

- the compartments are the model's species in model order (susceptible species first), plus an absorbing sink
  `:removed` when the model has removals `X → ∅` (design §J.2); `susceptible` is the model's Σ, and `strata` the
  strata of the species of a stratified model (`stratify`; the species labels whose stratum is not `:all`);
- each contact `s + J → X + J` becomes `OutbreakTransition(s, X, τ, type; via = [J])` with the per-contact rate τ
  of `per_contact_rates` (the model's rate convention applied with the nominal `mean_degree(network)`, so τ is the
  same as in the edge-based and pairwise back ends); `type` is `:infection` for a contact the typing classifies as
  `:contact` (s ∈ Σ, J ∉ Σ, X ∉ Σ) and `:contact_trace` otherwise (a contact whose recipient or product is
  susceptible, such as awareness, or whose recipient is already infected, such as quarantine of exposed contacts
  `E + I → Eq + I` or superinfection);
- each node transition `X → Y` becomes `:spontaneous`, and `X → ∅` becomes `X → :removed`;
- the infectious compartments are the infectors of the contacts that are not susceptible (design §A.5). This is
  structural (§J.8): a contact whose rate is 0 still marks its infector infectious, so the seeds count in
  `final_size` and the seeding background is the same for every value of the rates, while a susceptible catalyst
  (awareness, S + Sa → 2Sa) is never infectious;
- the infected compartments are NetworkEpiCore's `infected_species(model)` (design §J.8, §L.6: the one rule behind
  the final size and the `:cumulative` observable of every back end), unless `infected` is given. So E of SEIR and E,
  Eq of quarantine of exposed contacts are infected, and R, a vaccinated V, a traced Q (tracing `S + D → Q + D`, an
  `:infection` contact whose product never becomes infectious) and an aware Sa are not. A non-susceptible catalyst is
  an infector, so peer vaccination `S + V → 2V` counts V as infected; pass `infected` (or declare V susceptible) to
  count it otherwise;
- the transitions are in the model's reaction order (contacts, then node transitions), so the
  `transition_index` of an `OutbreakEvent` is the index of the NetworkEpiCore reaction.

`network` (a `NetworkDescriptor` or a `Graphs.AbstractGraph`) is needed only for a frequency- or density-dependent
rate convention (a contact on a named layer takes that layer's mean degree, as in every back end). The model must be
admissible for the `:stochastic` back end (constant rates, known layers). A contact on a named layer keeps its layer
(`OutbreakTransition(…; layer)`), so the model runs on a [`MultiplexGraph`](@ref) whose layers carry those names,
such as `sample_graph(net::MultiplexNetwork, N)`; when `network` is given, the layers are checked against it. An
`OutbreakModel` passes through unchanged.

```julia
OutbreakModel(seir_model(), Dict(:τ => 1/6, :σ => 1/5, :γ => 1/4))   # compartments [:S, :E, :I, :R]
```
"""
struct OutbreakModel
    compartments::Vector{Symbol}
    infectious::Vector{Bool}                  # parallel to compartments
    transitions::Vector{OutbreakTransition}
    name::Symbol
    index_of::Dict{Symbol, Int}               # compartment name => index
    susceptible::Vector{Symbol}               # the susceptible compartments Σ, in compartment order
    strata::Dict{Symbol, Symbol}              # compartment name => stratum (only for stratified models)
    infected::Vector{Bool}                    # parallel to compartments: counted as infected (design §J.8)
end

const _TRANSITION_TYPES = (:infection, :contact_trace, :spontaneous)

function OutbreakModel(compartments::Vector{Symbol},
                       infectious::Vector{Bool},
                       transitions::Vector{OutbreakTransition};
                       name::Symbol = :outbreak,
                       susceptible = nothing,
                       strata = nothing,
                       infected = nothing)
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
        tr.type in _TRANSITION_TYPES ||
            throw(ArgumentError("unknown transition type $(tr.type); expected one of $(_TRANSITION_TYPES)"))
        (isfinite(tr.rate) && tr.rate >= 0) ||
            throw(ArgumentError("transition rate must be finite and non-negative; got $(tr.rate)"))
        tr.type === :spontaneous && tr.layer !== :all && throw(ArgumentError(
            "the node-local transition $(tr.from)→$(tr.to) has layer :$(tr.layer); only edge-mediated transitions " *
            "act on a layer"))
        for v in tr.via
            haskey(index_of, v) ||
                throw(ArgumentError("unknown via compartment $(v) in transition $(tr.from)→$(tr.to)"))
        end
    end
    sus = susceptible === nothing ? _infer_susceptible(compartments, infectious, transitions) :
                                    _check_susceptible(collect(Symbol, susceptible), index_of)
    inf = infected === nothing ? _structural_infected(compartments, infectious, transitions, index_of) :
                                 _check_infected(infected, compartments, index_of, sus)
    return OutbreakModel(compartments, infectious, transitions, name, index_of, sus,
                         _check_strata(strata, index_of), inf)
end

function OutbreakModel(compartments::AbstractVector{Symbol},
                       infectious_set,
                       transitions::AbstractVector{OutbreakTransition};
                       name::Symbol = :outbreak,
                       susceptible = nothing,
                       strata = nothing,
                       infected = nothing)
    inf_flags = Bool[c in infectious_set for c in compartments]
    return OutbreakModel(collect(Symbol, compartments), inf_flags,
                         collect(OutbreakTransition, transitions); name, susceptible, strata, infected)
end

# The non-infectious sources of `:infection` transitions, in compartment order.
function _infer_susceptible(compartments, infectious, transitions)
    srcs = Set{Symbol}(tr.from for tr in transitions if tr.type === :infection)
    return Symbol[c for (c, inf) in zip(compartments, infectious) if !inf && c in srcs]
end

function _check_susceptible(sus::Vector{Symbol}, index_of)
    for s in sus
        haskey(index_of, s) || throw(ArgumentError("unknown susceptible compartment $(s)"))
    end
    allunique(sus) || throw(ArgumentError("susceptible compartments must be unique"))
    return sort!(sus; by = s -> index_of[s])
end

function _check_infected(infected, compartments, index_of, sus)
    names = Symbol[]
    for X in infected
        X isa Symbol || throw(ArgumentError("`infected` must list compartment names (Symbols); got $(repr(X))"))
        haskey(index_of, X) || throw(ArgumentError("unknown infected compartment $(X)"))
        X in names && throw(ArgumentError("infected compartment $(X) is listed twice"))
        X in sus && throw(ArgumentError(
            "the susceptible compartment $(X) cannot be infected (a node in Σ is by definition not yet infected)"))
        push!(names, X)
    end
    return Bool[c in names for c in compartments]
end

_check_strata(::Nothing, index_of) = Dict{Symbol, Symbol}()
function _check_strata(strata, index_of)
    d = Dict{Symbol, Symbol}()
    for (X, a) in strata
        X isa Symbol && a isa Symbol ||
            throw(ArgumentError("strata must map compartment names to stratum names (Symbols); got $(X) => $(a)"))
        haskey(index_of, X) || throw(ArgumentError("unknown compartment $(X) in strata"))
        haskey(d, X) && throw(ArgumentError("compartment $(X) is given two strata"))
        d[X] = a
    end
    return d
end

ncompartments(m::OutbreakModel) = length(m.compartments)
infectious_indices(m::OutbreakModel) = findall(m.infectious)

"""
    _structural_infected(compartments, infectious, transitions, index_of) -> Vector{Bool}

The infected compartments of a hand-built [`OutbreakModel`](@ref) without an explicit `infected` list, decided
structurally from its transitions (models converted from a `ContactModel` use NetworkEpiCore's `infected_species`
instead, which applies the same idea to the typing):

1. An *infection* is an `:infection` transition that has an infectious catalyst (in its effective `via`: an empty
   `via` means the infectious compartments) and whose product *leads to infectiousness*: it is infectious or reaches
   an infectious compartment through transitions other than infectious-catalysed `:infection` contacts (spontaneous
   ones, tracing, awareness). `:contact_trace` transitions and contacts catalysed only by non-infectious compartments
   (awareness, peer vaccination) are never infections.
2. The *susceptible* (barrier) compartments are the non-infectious recipients of infections.
3. All other transitions (spontaneous ones, tracing, awareness, ...) preserve infection status. The forward closure
   starts at the products of infections, the backward closure at every infectious compartment; both follow the
   status-preserving transitions and never add a susceptible compartment.
4. A compartment is infected if it is infectious or lies in both closures, i.e. on a status-preserving path from an
   infection to infectiousness.

So E in SEIR is infected (E seeds count from `t = 0`), also when E is itself a contact recipient (tracing `E → Q`,
quarantine `E → Eq → Iq`, where `Eq` is infected too) or when an infectious compartment is (`I → Q` tracing,
superinfection `I1 → I2`), while R, a vaccinated V (`S → V`), a traced Q (`S → Q` by `:contact_trace`) and an aware
susceptible (`S → Sa` via `Sa`, even with importation `Sa → E`) are not.

Limit: the rule is structural. A compartment entered from an infectious one and left only by status-preserving
transitions towards infectiousness is indistinguishable from a latent phase (so recovery `I → Sa` into a class whose
only exit is importation `Sa → E` counts `Sa` as infected), and a latent compartment that is itself the recipient of
an infection (re-exposure `E → E2` via `I`) is indistinguishable from a susceptible one. Pass `infected` to the
constructor in such cases, or build the model from a `ContactModel`, whose declared susceptible classes resolve them.
"""
function _structural_infected(compartments::Vector{Symbol}, infectious_flags::Vector{Bool},
                              transitions::Vector{OutbreakTransition}, index_of::Dict{Symbol, Int})
    C = length(compartments)
    T = length(transitions)
    infectious = BitVector(infectious_flags)
    src = Int[index_of[tr.from] for tr in transitions]
    dst = Int[index_of[tr.to] for tr in transitions]
    # 1. infections: an infectious catalyst and a product that leads to infectiousness
    catalysed = BitVector([tr.type === :infection &&
                           (isempty(tr.via) ? any(infectious) : any(c -> infectious[index_of[c]], tr.via))
                           for tr in transitions])
    leads = _backward_closure!(copy(infectious), src, dst, .!catalysed, falses(C))
    infecting = BitVector([catalysed[j] && leads[dst[j]] for j in 1:T])
    # 2. susceptible compartments, and the start of the forward closure
    barrier = falses(C)
    fwd = falses(C)
    for j in 1:T
        infecting[j] || continue
        infectious[src[j]] || (barrier[src[j]] = true)
        fwd[dst[j]] = true
    end
    fwd .&= .!barrier
    # 3. closures along the status-preserving transitions
    preserving = .!infecting
    changed = true
    while changed
        changed = false
        for j in 1:T
            preserving[j] || continue
            if fwd[src[j]] && !fwd[dst[j]] && !barrier[dst[j]]
                fwd[dst[j]] = true
                changed = true
            end
        end
    end
    bwd = _backward_closure!(copy(infectious), src, dst, preserving, barrier)
    # 4. infectious, or on a status-preserving path from an infection to infectiousness
    return Vector{Bool}(infectious .| (fwd .& bwd))
end

# Grow `set` backwards along the transitions `j` with `use[j]` (from `src[j]` to `dst[j]`), never adding a
# compartment in `barrier`.
function _backward_closure!(set::BitVector, src::Vector{Int}, dst::Vector{Int}, use::BitVector, barrier::BitVector)
    changed = true
    while changed
        changed = false
        for j in eachindex(src)
            use[j] || continue
            if set[dst[j]] && !set[src[j]] && !barrier[src[j]]
                set[src[j]] = true
                changed = true
            end
        end
    end
    return set
end

# ---------------------------------------------------------------------------------------------------------------
# From a ContactModel (design §A.5, §J.2, §J.8)
# ---------------------------------------------------------------------------------------------------------------

"""
    NetworkOutbreaks.REMOVED_SINK

The name (`:removed`) of the absorbing compartment that `OutbreakModel(cm, p)` adds for the removals `X → ∅` of a
`ContactModel` (design §J.2).
"""
const REMOVED_SINK = :removed

OutbreakModel(m::OutbreakModel) = m
function OutbreakModel(m::OutbreakModel, p::AbstractDict; kw...)
    isempty(p) || throw(ArgumentError("OutbreakModel :$(m.name) already has numeric rates; parameter values " *
                                      "($(join(keys(p), ", "))) apply only to a model description such as a " *
                                      "ContactModel"))
    return m
end

function OutbreakModel(cm::ContactModel, p::AbstractDict = Dict{Symbol,Float64}();
                       network = nothing, name::Symbol = nameof(cm), infected = nothing)
    # a MultiplexGraph has no nominal mean degree, which only the frequency- and density-dependent conventions need
    net = network isa MultiplexGraph && rate_convention(cm) isa PerContact ? nothing : _rate_network(network)
    require_admissible(cm, :stochastic; network = net)
    for c in contacts(cm)
        _check_contact_layer(cm, c, network)
    end
    inst = instantiate(cm, p)
    τ = Float64.(per_contact_rates(inst, net))
    kinds = Dict(typing(inst).reaction_types)
    Σ = susceptible_species(inst)
    species = species_names(inst)
    cs = contacts(inst)
    ts = node_transitions(inst)
    if any(t -> t.to === nothing, ts)
        REMOVED_SINK in species && throw(ArgumentError(
            "OutbreakModel(:$(nameof(cm))): the model has removals X → ∅ and a species named :$(REMOVED_SINK), " *
            "the name of the absorbing sink that removals are lowered to; rename that species"))
        push!(species, REMOVED_SINK)
    end
    # Infection status is structural (§A.5, §J.8): every non-susceptible infector is infectious, whatever the value of
    # its rate, and the infected compartments are NetworkEpiCore's `infected_species` (§L.6: one implementation for
    # every back end), so a rate of 0 changes the dynamics but never which compartments count as infected.
    infectors = Set{Symbol}(c.infector for c in cs if !(c.infector in Σ))
    infected === nothing && (infected = infected_species(inst))
    trs = OutbreakTransition[]
    for (c, r) in zip(cs, τ)
        type = kinds[c.name] === :contact ? :infection : :contact_trace
        push!(trs, OutbreakTransition(c.recipient, c.product, r, type; via = [c.infector], layer = c.layer))
    end
    for t in ts
        push!(trs, OutbreakTransition(t.from, something(t.to, REMOVED_SINK), Float64(t.rate), :spontaneous))
    end
    strata = Dict{Symbol, Symbol}(X => l.stratum for (X, l) in species_labels(inst) if l.stratum !== :all)
    return OutbreakModel(species, Bool[X in infectors for X in species], trs; name, susceptible = Σ, strata,
                         infected)
end

# A contact on a named layer needs a multiplex network with that layer; without a network the check is left to the
# sampler (`_bind_layers!`, src/algorithms/common.jl).
function _check_contact_layer(cm, c, network)
    (c.layer === :all || network === nothing) && return nothing
    names = _layer_names_of(network)
    names === nothing && throw(ArgumentError(
        "OutbreakModel(:$(nameof(cm))): the contact `$(c.name)` is on the layer :$(c.layer), which needs a " *
        "multiplex network (a MultiplexNetwork descriptor or a MultiplexGraph); got $(nameof(typeof(network)))"))
    c.layer in names || throw(ArgumentError(
        "OutbreakModel(:$(nameof(cm))): the contact `$(c.name)` is on the layer :$(c.layer), which is not a layer " *
        "of the network (layers: $(join(names, ", ")))"))
    return nothing
end

# The layer names of a multiplex network, or `nothing` for a network without layers (methods for the contact-network
# containers are in network.jl).
_layer_names_of(net::MultiplexNetwork) = layer_names(net)
_layer_names_of(net) = nothing

# Any other model description goes through `contact_model` (this replaces the 0.1 package extensions).
function OutbreakModel(x, p::AbstractDict = Dict{Symbol,Float64}(); kw...)
    applicable(contact_model, x) || throw(ArgumentError(
        "OutbreakModel: cannot convert a $(typeof(x)); pass a ContactModel (e.g. sir_model()), a model that " *
        "`contact_model` accepts (load Catalyst for a ReactionSystem), or build an OutbreakModel from " *
        "compartments, infectious flags and OutbreakTransitions"))
    return OutbreakModel(contact_model(x), p; kw...)
end

# The descriptor whose nominal mean degree the rate convention uses (methods for the contact-network types are in
# network.jl).
_rate_network(::Nothing) = nothing
_rate_network(net::NetworkDescriptor) = net
_rate_network(g::AbstractGraph) = ExplicitGraph(g)
_rate_network(x) = throw(ArgumentError("`network` must be a NetworkDescriptor or a Graphs.AbstractGraph; got $(typeof(x))"))

function Base.show(io::IO, ::MIME"text/plain", m::OutbreakModel)
    println(io, "OutbreakModel :", m.name, " with ", ncompartments(m), " compartments and ",
            length(m.transitions), " transitions")
    println(io, "  compartments: ", join((inf ? "$(c)*" : string(c) for (c, inf) in zip(m.compartments, m.infectious)),
                                        ", "), "   (* infectious)")
    println(io, "  susceptible:  ", isempty(m.susceptible) ? "none" : join(m.susceptible, ", "))
    inf = [c for (c, f) in zip(m.compartments, m.infected) if f]
    print(io, "  infected:     ", isempty(inf) ? "none" : join(inf, ", "))
    if !isempty(m.strata)
        byst = Dict{Symbol, Vector{Symbol}}()
        for c in m.compartments
            haskey(m.strata, c) && push!(get!(byst, m.strata[c], Symbol[]), c)
        end
        print(io, "\n  strata:       ", join(("$(a) ($(join(cs, ", ")))" for (a, cs) in sort!(collect(byst); by = first)), "; "))
    end
    for tr in m.transitions
        print(io, "\n  ", tr.from, " → ", tr.to, "  ", tr.type, " at ", tr.rate)
        tr.type === :spontaneous || isempty(tr.via) || print(io, " via ", join(tr.via, ", "))
        tr.type === :spontaneous || tr.layer === :all || print(io, " on layer ", tr.layer)
    end
    return nothing
end
