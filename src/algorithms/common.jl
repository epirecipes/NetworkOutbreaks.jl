#=
algorithms/common.jl

Algorithm dispatch hierarchy and the machinery shared by every sampler.

- `ALGORITHM_REVISION`: bumped whenever a change could alter random streams or distributions.
- RNG resolution: `seed::Integer` builds a `StableRNG` from a splitmix64-mixed seed; `rng` is used as given.
- `_RunModel`: the per-run compiled model. It holds the run's own copy of the transition rates, so an intervention
  never writes into `spec.model` (M1). Transitions are referred to by their index in `model.transitions`.
- One definition of the contact hazard, used by DirectSSA, NextReaction, CompositionRejection and HAS alike (M2):
  an edge-mediated transition `j` (`:infection` or `:contact_trace`, M3) out of the compartment of node `v` fires at
  rate `rates[j] · Σ_{u ∈ N(v)} w(u, v) · [state(u) ∈ via_j]`, where `w` is the layer multiplier of a multiplex
  network (1 otherwise) and an empty `via` means "the infectious compartments". The `via` compartments need not be
  infectious (tracing from diagnosed nodes, peer vaccination, awareness). The choice among several transitions at a
  fired node uses exactly the same weights.
- Infected compartments and infection counts (m5, see `_infected_mask`).
- Network validation: directed graphs and self-loops are rejected (m3).
- The scheduled-event queue: time-varying-network updates and scheduled interventions are merged into one
  time-ordered stream (M5), and a run only stops at zero total rate when nothing is scheduled within the horizon (M4).

Every algorithm implements

    _simulate_impl(alg, spec, rng::AbstractRNG, seed::Union{Nothing, UInt64}, keep::Symbol, plan::InterventionPlan)

and returns an `OutbreakTrajectory` whose `model` is the unmodified `spec.model`.
=#

using StableRNGs: StableRNG

export ALGORITHM_REVISION

"""
    ALGORITHM_REVISION :: String

Revision of the stochastic samplers in NetworkOutbreaks. It is bumped whenever a change could alter the random
streams or the distribution of any simulated quantity, and it is part of the cache key of stored ensemble
summaries (together with the scenario hash), so a summary produced by an older revision is never silently reused.

Revisions:
- `"1"`: NetworkOutbreaks 0.1 (Xoshiro streams; DirectSSA counted only infectious neighbours; `:contact_trace` only in
  DirectSSA; rate changes mutated the shared model; runs stopped at zero total rate even with scheduled work pending).
- `"2"`: `StableRNG` streams from splitmix64-mixed seeds; one contact-hazard definition in all algorithms (`via`
  catalysts need not be infectious); `:contact_trace` everywhere; per-run rates; one merged, time-ordered queue of
  network updates and interventions; interventions in DirectSSA, NextReaction and HAS; `infection_counts` count entries
  into infected compartments (so latent seeds count); directed graphs rejected.
"""
const ALGORITHM_REVISION = "2"

"""
    OutbreakAlgorithm

Supertype of the exact stochastic samplers. The graph samplers `DirectSSA`, `NextReaction`, `CompositionRejection`
and `HAS` sample the same continuous-time Markov chain on a contact graph (see `simulate` for the shared semantics);
they differ in cost and in which network types and interventions they support. `MassActionSSA` samples a well-mixed
population (`WellMixed(κ)`) and `FleetingContactSSA` mean-field social heterogeneity (`MFSHNetwork(d)`) at the level
of compartment counts.
"""
abstract type OutbreakAlgorithm end

# Transition types that are edge-mediated ("contacts"). `:contact_trace` is a contact whose product is not
# infectious; it has exactly the same hazard semantics as `:infection`.
const _CONTACT_TYPES = (:infection, :contact_trace)

# The catalyst compartments of a transition as the samplers use them: an empty `via` on a contact transition means
# the infectious compartments (see `_RunModel`).
function _effective_via(tr::OutbreakTransition, model::OutbreakModel)
    (isempty(tr.via) && tr.type in _CONTACT_TYPES) || return tr.via
    return Symbol[c for (c, inf) in zip(model.compartments, model.infectious) if inf]
end

# Algorithms that implement scheduled and threshold interventions.
_supports_interventions(::OutbreakAlgorithm) = false

# ---------------------------------------------------------------------------------------------------------------
# Random numbers
# ---------------------------------------------------------------------------------------------------------------

const _GOLDEN64 = 0x9e3779b97f4a7c15

@inline function _splitmix64_mix(z::UInt64)
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    return z ⊻ (z >> 31)
end

function _seed_u64(seed::Integer)
    0 <= seed <= typemax(UInt64) ||
        throw(ArgumentError("seed must be an integer in 0:typemax(UInt64); got $(seed)"))
    return UInt64(seed)
end

"""
    NetworkOutbreaks.stable_rng(seed::Integer) -> StableRNG

The random-number generator that `simulate(spec; seed)` uses: a `StableRNG` (stable across Julia versions) seeded
with the splitmix64 mix of `seed`. The mixing matters: `StableRNG` is a Lehmer generator, and the streams of nearby
raw seeds are strongly dependent: `StableRNG(s)` starts from the state 2s + 1, so after n draws the states of
`StableRNG(s + 1)` and `StableRNG(s)` differ by the constant 2a^n mod 2^128 (a the Lehmer multiplier). Hence the
difference u_n(s + 1) − u_n(s) (mod 1) of their n-th uniforms is a constant c_n, independent of s up to one unit in
the last place, and the first draws of adjacent seeds have correlation ≈ −0.43. So nearby raw seeds do not give
independent runs; `stable_rng(s)` and `stable_rng(s + 1)` do.

`simulate(spec; seed = s)` and `simulate(spec; rng = NetworkOutbreaks.stable_rng(s))` give identical trajectories.
"""
stable_rng(seed::Integer) = StableRNG(_splitmix64_mix(_seed_u64(seed) + _GOLDEN64))

# The seed recorded in a trajectory: the seed of a seeded run, `nothing` for a run driven by an explicit `rng`.
const _RecordedSeed = Union{Nothing, UInt64}

# Resolve the (`seed`, `rng`) keyword pair of `simulate`. Returns the RNG to use and the seed to record in the
# trajectory (`nothing` when an explicit `rng` was supplied).
function _resolve_rng(seed, rng)
    if rng !== nothing
        seed === nothing ||
            throw(ArgumentError("pass either `seed` or `rng` to simulate, not both"))
        return rng, nothing
    end
    s = seed === nothing ? rand(UInt64) : _seed_u64(seed)
    return stable_rng(s), s
end

# ---------------------------------------------------------------------------------------------------------------
# The per-run compiled model
# ---------------------------------------------------------------------------------------------------------------

"""
    _infected_mask(model) -> BitVector

The compartments counted as *infected* (`model.infected`, design §J.8): a node's `infection_counts` entry is
incremented each time it enters an infected compartment from a non-infected one (by any event: contact, node-local
transition or intervention), and nodes that start in an infected compartment count once. `final_size` is the
fraction of nodes with a positive count.

For a model converted from a `ContactModel` the infected compartments are NetworkEpiCore's `infected_species` (the
single implementation of §J.8, design §L.6); for a hand-built `OutbreakModel` they are the `infected` keyword or the
structural rule `_structural_infected` (src/model.jl).
"""
_infected_mask(model::OutbreakModel) = BitVector(model.infected)

# Per-run compiled model. Everything here is freshly allocated for each run, so interventions may write to
# `rates` and `spont_total` without affecting `model` or any other run (M1).
struct _RunModel
    model::OutbreakModel
    C::Int
    rates::Vector{Float64}              # per-run copy of the transition rates
    to_idx::Vector{Int}                 # target compartment of each transition
    from_idx::Vector{Int}               # source compartment of each transition
    is_contact::BitVector               # edge-mediated transition?
    spont_by_src::Vector{Vector{Int}}   # spontaneous transition indices by source compartment
    contact_by_src::Vector{Vector{Int}} # contact transition indices by source compartment
    via_mask::Vector{BitVector}         # catalyst compartments of each contact transition
    catalyst::BitVector                 # union of the via masks
    spont_total::Vector{Float64}        # Σ spontaneous rates out of each compartment
    infected::BitVector                 # see `_infected_mask`
    layer_class::Vector{Int}            # layer class k of each transition (1 for every :all transition)
    class_on::Vector{BitVector}         # per layer class: the network layers it acts on
    all_layers::Bool                    # a single class acting on every layer (the unlayered fast path)
end

# The run model of `model` on `network` (the spec's network; `nothing` for a network without layers). The contact
# transitions are grouped into layer classes by their `layer`: class k acts on the network layers `class_on[k]`, and
# the catalyst tally of a node has one block of C entries per class (see `_tally_catalysts!`). Without named layers
# there is one class acting on every layer, and the tally is the plain layer-weighted count of 0.1.
function _RunModel(model::OutbreakModel, network = nothing)
    C = ncompartments(model)
    T = length(model.transitions)
    rates = Float64[tr.rate for tr in model.transitions]
    to_idx = Int[model.index_of[tr.to] for tr in model.transitions]
    from_idx = Int[model.index_of[tr.from] for tr in model.transitions]
    is_contact = BitVector([tr.type in _CONTACT_TYPES for tr in model.transitions])
    spont_by_src = [Int[] for _ in 1:C]
    contact_by_src = [Int[] for _ in 1:C]
    via_mask = [falses(C) for _ in 1:T]
    catalyst = falses(C)
    for (j, tr) in pairs(model.transitions)
        if is_contact[j]
            push!(contact_by_src[from_idx[j]], j)
            if isempty(tr.via)
                via_mask[j] .= model.infectious
            else
                for sym in tr.via
                    via_mask[j][model.index_of[sym]] = true
                end
            end
            catalyst .|= via_mask[j]
        else
            push!(spont_by_src[from_idx[j]], j)
        end
    end
    layer_class, class_on = _bind_layers(model, is_contact, network)
    all_layers = length(class_on) == 1 && all(class_on[1])
    rm = _RunModel(model, C, rates, to_idx, from_idx, is_contact, spont_by_src, contact_by_src,
                   via_mask, catalyst, zeros(Float64, C), _infected_mask(model), layer_class, class_on, all_layers)
    _refresh_spont_totals!(rm)
    return rm
end

# The layer names of the network a sampler iterates over, or `nothing` for a single-graph network.
_network_layer_names(net::MultiplexGraph) = net.names
_network_layer_names(net) = nothing

# Resolve the `layer` of every contact transition against the network: returns the class of each transition and, per
# class, the network layers it acts on (class 1 is `:all` when some transition acts on every layer).
function _bind_layers(model::OutbreakModel, is_contact::BitVector, network)
    names = _network_layer_names(network)
    L = names === nothing ? 1 : length(names)
    labels = Symbol[:all]
    layer_class = ones(Int, length(model.transitions))
    for (j, tr) in pairs(model.transitions)
        (is_contact[j] && tr.layer !== :all) || continue
        names === nothing && throw(ArgumentError(
            "the transition $(tr.from)→$(tr.to) of model :$(model.name) acts on the layer :$(tr.layer), which needs a " *
            "MultiplexGraph with a layer of that name (e.g. sample_graph(net::MultiplexNetwork, N)); the network is a " *
            "$(nameof(typeof(network)))"))
        tr.layer in names || throw(ArgumentError(
            "the transition $(tr.from)→$(tr.to) of model :$(model.name) acts on the layer :$(tr.layer), which is not " *
            "a layer of the MultiplexGraph (layers: $(join(names, ", ")))"))
        k = findfirst(==(tr.layer), labels)
        k === nothing && (push!(labels, tr.layer); k = length(labels))
        layer_class[j] = k
    end
    class_on = BitVector[lab === :all ? trues(L) : BitVector(names .=== lab) for lab in labels]
    return layer_class, class_on
end

# A catalyst tally buffer for `rm`: C entries per layer class.
_tally_buffer(rm::_RunModel) = zeros(Float64, rm.C * length(rm.class_on))

function _refresh_spont_totals!(rm::_RunModel)
    for c in 1:rm.C
        rm.spont_total[c] = sum((rm.rates[j] for j in rm.spont_by_src[c]); init = 0.0)
    end
    return rm
end

_has_contacts(rm::_RunModel) = any(!isempty, rm.contact_by_src)

# Initial state; the constructor validates the indices, fills the counts and counts the infected seeds once.
_initial_outbreak_state(rm::_RunModel, node_state::Vector{Int}) = OutbreakState(rm.model, node_state)

# Move node `v` to compartment `new`, updating counts and infection counts. Returns the old compartment.
@inline function _move_node!(state::OutbreakState, rm::_RunModel, v::Integer, new::Int)
    @inbounds begin
        old = state.node_state[v]
        state.node_state[v] = new
        state.counts[old] -= 1
        state.counts[new] += 1
        if rm.infected[new] && !rm.infected[old]
            state.infection_counts[v] += 1
        end
    end
    return old
end

@inline _fire!(state::OutbreakState, rm::_RunModel, v::Integer, j::Int) =
    _move_node!(state, rm, v, rm.to_idx[j])

# True when no node is in a catalyst compartment: then no contact hazard can become positive, whatever the network
# does, so with zero total rate and no pending intervention the state is absorbing.
_no_catalysts(rm::_RunModel, counts::Vector{Int}) =
    !any(c -> rm.catalyst[c] && counts[c] > 0, 1:rm.C)

# ---------------------------------------------------------------------------------------------------------------
# Contact hazards (one definition for all algorithms)
# ---------------------------------------------------------------------------------------------------------------

# Tally the (layer-weighted) catalyst neighbours of `v` by compartment into `tally` (`_tally_buffer(rm)`: C entries
# per layer class k, at offset (k − 1)·C, each counting only the layers of that class). Only compartments in
# `rm.catalyst` are counted. Returns true if any catalyst neighbour was found.
@inline function _tally_catalysts!(tally::Vector{Float64}, v::Integer, layers, weights,
                                   node_state::Vector{Int}, rm::_RunModel)
    fill!(tally, 0.0)
    found = false
    if rm.all_layers
        @inbounds for l in eachindex(layers)
            w = weights[l]
            w > 0 || continue
            for u in neighbors(layers[l], v)
                c = node_state[u]
                if rm.catalyst[c]
                    tally[c] += w
                    found = true
                end
            end
        end
    else
        C = rm.C
        @inbounds for l in eachindex(layers)
            w = weights[l]
            w > 0 || continue
            for u in neighbors(layers[l], v)
                c = node_state[u]
                rm.catalyst[c] || continue
                for k in eachindex(rm.class_on)
                    if rm.class_on[k][l]
                        tally[(k - 1) * C + c] += w
                        found = true
                    end
                end
            end
        end
    end
    return found
end

# Weight of contact transition j at a node whose catalyst tally is `tally`.
@inline function _contact_weight(rm::_RunModel, j::Int, tally::Vector{Float64})
    mask = rm.via_mask[j]
    off = (rm.layer_class[j] - 1) * rm.C
    w = 0.0
    @inbounds for c in 1:rm.C
        mask[c] && (w += tally[off + c])
    end
    return rm.rates[j] * w
end

# Total contact hazard at node v (uses `tally` as scratch space).
function _contact_hazard!(tally::Vector{Float64}, v::Integer, layers, weights,
                          node_state::Vector{Int}, rm::_RunModel)
    trs = rm.contact_by_src[node_state[v]]
    isempty(trs) && return 0.0
    _tally_catalysts!(tally, v, layers, weights, node_state, rm) || return 0.0
    h = 0.0
    for j in trs
        h += _contact_weight(rm, j, tally)
    end
    return h
end

# Choose which contact transition fires at node v, with probability proportional to its weight.
function _sample_contact_transition!(tally::Vector{Float64}, v::Integer, layers, weights,
                                     node_state::Vector{Int}, rm::_RunModel, rng::AbstractRNG)
    trs = rm.contact_by_src[node_state[v]]
    length(trs) == 1 && return trs[1]
    _tally_catalysts!(tally, v, layers, weights, node_state, rm)
    total = 0.0
    for j in trs
        total += _contact_weight(rm, j, tally)
    end
    target = rand(rng) * total
    cum = 0.0
    last_positive = 0
    for j in trs
        w = _contact_weight(rm, j, tally)
        w > 0 || continue
        last_positive = j
        cum += w
        target < cum && return j
    end
    return last_positive   # floating-point guard: the last transition with positive weight
end

# Choose which spontaneous transition fires out of compartment c.
function _sample_spontaneous_transition(rm::_RunModel, c::Int, rng::AbstractRNG)
    trs = rm.spont_by_src[c]
    length(trs) == 1 && return trs[1]
    target = rand(rng) * rm.spont_total[c]
    cum = 0.0
    last_positive = 0
    for j in trs
        r = rm.rates[j]
        r > 0 || continue
        last_positive = j
        cum += r
        target < cum && return j
    end
    return last_positive
end

# ---------------------------------------------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------------------------------------------

const _DIRECTED_MSG =
    "directed contact graphs are not supported: NetworkOutbreaks treats every edge as a symmetric contact " *
    "(u infects v and v infects u). Convert with SimpleGraph(g) if the contacts are symmetric."

function _check_contact_graph(g::AbstractGraph)
    is_directed(g) && throw(ArgumentError(_DIRECTED_MSG))
    has_self_loops(g) &&
        throw(ArgumentError("contact graphs must not contain self-loops (a node is not its own contact)"))
    return g
end

# Validate the network of `spec` and return (graph, layers, weights, updates):
# - `graph` is the (per-run copy of the) graph for static and time-varying networks, or the first layer;
# - `layers`/`weights` are what the contact hazard iterates over;
# - `updates` is the TVN update list, or `nothing`.
function _prepare_network(network)
    if network isa MultiplexGraph
        foreach(_check_contact_graph, network.layers)
        return network.layers[1], network.layers, network.layer_rates, nothing
    elseif network isa TimeVaryingNetwork
        _check_contact_graph(network.graph)
        g = deepcopy(network.graph)        # the spec's graph is never mutated
        n = nv(g)
        for u in network.updates
            (1 <= u.src <= n && 1 <= u.dst <= n) ||
                throw(ArgumentError("TimeVaryingNetwork update $(u) refers to a node outside 1:$(n)"))
            u.src == u.dst &&
                throw(ArgumentError("TimeVaryingNetwork update $(u) would add a self-loop"))
            u.action in (:add, :remove) ||
                throw(ArgumentError("TimeVaryingNetwork update action must be :add or :remove; got $(u.action)"))
        end
        return g, (g,), (1.0,), network.updates
    else
        g = _check_contact_graph(_outbreak_graph(network))
        return g, (g,), (1.0,), nothing
    end
end

function _apply_network_update!(g::AbstractGraph, upd)
    upd.action === :add ? add_edge!(g, upd.src, upd.dst) : rem_edge!(g, upd.src, upd.dst)
    return nothing
end

# ---------------------------------------------------------------------------------------------------------------
# Scheduled events: TVN updates and interventions in one time-ordered queue
# ---------------------------------------------------------------------------------------------------------------

mutable struct _Schedule
    updates::Union{Nothing, Vector{_TVN_Update}}
    next_update::Int
    scheduled::Vector{AbstractIntervention}
    next_iv::Int
    thresholds::Vector{ThresholdIntervention}
    fired::BitVector
end

function _Schedule(updates, plan::InterventionPlan, t0::Float64)
    # Network updates strictly before the start time describe the network's past (the base graph is the state at
    # tspan[1]); they are skipped. Updates at t0 are applied before the first event.
    next_update = 1
    if updates !== nothing
        while next_update <= length(updates) && updates[next_update].t < t0
            next_update += 1
        end
    end
    return _Schedule(updates, next_update, plan.scheduled, 1, plan.thresholds,
                     falses(length(plan.thresholds)))
end

_next_update_time(s::_Schedule) =
    (s.updates !== nothing && s.next_update <= length(s.updates)) ? s.updates[s.next_update].t : Inf
_next_intervention_time(s::_Schedule) =
    s.next_iv <= length(s.scheduled) ? _intervention_time(s.scheduled[s.next_iv]) : Inf
_next_scheduled_time(s::_Schedule) = min(_next_update_time(s), _next_intervention_time(s))
_interventions_pending(s::_Schedule) = s.next_iv <= length(s.scheduled)

# Apply every scheduled item due at time `t` (network updates first, then interventions, each in list order).
# `on_update(upd)` is called after each network update, `on_change(touched)` after each intervention, where
# `touched` is the vector of moved nodes or `nothing` when rates changed (all hazards may have changed).
# Returns true if an intervention was applied (the caller then records a snapshot).
function _apply_due!(s::_Schedule, t::Float64, g, state, rm, n, rng, on_update, on_change)
    while s.updates !== nothing && s.next_update <= length(s.updates) && s.updates[s.next_update].t <= t
        upd = s.updates[s.next_update]
        _apply_network_update!(g, upd)
        on_update(upd)
        s.next_update += 1
    end
    applied = false
    while s.next_iv <= length(s.scheduled) && _intervention_time(s.scheduled[s.next_iv]) <= t
        touched = _apply_intervention!(s.scheduled[s.next_iv], rm, state, n, rng)
        on_change(touched)
        s.next_iv += 1
        applied = true
    end
    return applied
end

# Fire every threshold intervention whose condition holds. Each fires at most once; an action may trigger further
# thresholds, so the check repeats until nothing new fires. Returns true if any fired.
function _check_thresholds!(s::_Schedule, state, rm, n, rng, on_change)
    any_fired = false
    progress = true
    while progress
        progress = false
        for (i, tiv) in pairs(s.thresholds)
            s.fired[i] && continue
            cnt = state.counts[rm.model.index_of[tiv.compartment]]
            hit = tiv.direction === :above ? cnt >= tiv.threshold : cnt <= tiv.threshold
            if hit
                s.fired[i] = true
                touched = _apply_intervention!(tiv.action, rm, state, n, rng)
                on_change(touched)
                any_fired = progress = true
            end
        end
    end
    return any_fired
end

# ---------------------------------------------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------------------------------------------

struct _Recorder
    times::Vector{Float64}
    counts::Vector{Int}                 # flattened C × length(times)
    events::Vector{OutbreakEvent}
    keep_events::Bool
end

_Recorder(keep::Symbol) = _Recorder(Float64[], Int[], OutbreakEvent[], keep === :events)

function _record!(rec::_Recorder, t::Float64, counts::Vector{Int})
    push!(rec.times, t)
    append!(rec.counts, counts)
    return rec
end

@inline _log_event!(rec::_Recorder, t::Float64, j::Int, v::Integer) =
    rec.keep_events && push!(rec.events, OutbreakEvent(t, j, Int(v)))

function _trajectory(rec::_Recorder, rm::_RunModel, state::OutbreakState, t_end::Float64,
                     seed::_RecordedSeed, name::Symbol)
    if isfinite(t_end) && rec.times[end] < t_end
        _record!(rec, t_end, state.counts)
    end
    counts = reshape(rec.counts, rm.C, length(rec.times))
    return OutbreakTrajectory(rm.model, rec.times, counts, copy(state.infection_counts),
                              rec.events, seed, name)
end

# ---------------------------------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------------------------------

"""
    simulate(spec::OutbreakSpec; algorithm = DirectSSA(), seed = nothing, rng = nothing,
             keep = :counts, interventions = InterventionPlan()) -> OutbreakTrajectory

Run one exact stochastic simulation of `spec`.

- **Random numbers.** Pass `seed::Integer` to use `NetworkOutbreaks.stable_rng(seed)` (a `StableRNG`, stable across
  Julia versions, seeded through splitmix64 so that nearby seeds give independent runs), or `rng::AbstractRNG` to use
  that generator directly (it is advanced). With neither, a seed is drawn from the global RNG. The seed is stored in
  `traj.seed` (`nothing` when `rng` was given), and `simulate(spec; seed = traj.seed)` reproduces a seeded run
  exactly.
- **`keep`**: `:counts` stores event times and compartment counts; `:events` also stores the event log
  (`OutbreakEvent(time, transition_index, node)`).
- **`interventions`**: an `InterventionPlan`; supported by `DirectSSA`, `NextReaction`, `HAS`, `MassActionSSA` and
  `FleetingContactSSA` (not by `CompositionRejection`).

Semantics shared by all algorithms:
- an edge-mediated transition (`:infection` or `:contact_trace`) at node `v` has hazard
  `rate × #{neighbours u of v : state(u) ∈ via}` (layer-weighted on a `MultiplexGraph`); an empty `via` means the
  infectious compartments, and `via` compartments need not be infectious;
- network updates and scheduled interventions are processed in time order, and the run continues through periods of
  zero total rate while any of them is pending within `tspan`;
- contact graphs must be undirected and free of self-loops.
"""
function simulate(spec::OutbreakSpec;
                  algorithm::OutbreakAlgorithm = DirectSSA(),
                  seed::Union{Nothing, Integer} = nothing,
                  rng::Union{Nothing, AbstractRNG} = nothing,
                  keep::Symbol = :counts,
                  interventions::InterventionPlan = InterventionPlan())
    keep in (:counts, :events) ||
        throw(ArgumentError("keep must be :counts or :events"))
    if !isempty(interventions)
        _supports_interventions(algorithm) ||
            throw(ArgumentError("interventions are not supported by $(nameof(typeof(algorithm))); use DirectSSA, " *
                                "NextReaction or HAS (MassActionSSA and FleetingContactSSA for their populations)"))
        _validate_interventions(interventions, spec.model, spec.tspan)
    end
    run_rng, recorded = _resolve_rng(seed, rng)
    return _simulate_impl(algorithm, spec, run_rng, recorded, keep, interventions)
end
