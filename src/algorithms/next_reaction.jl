#=
algorithms/next_reaction.jl (owner: WP26, after WP3)

Gibson–Bruck next-reaction method with node-level channels.

Each node owns two reaction channels: a spontaneous channel whose hazard is the sum of the spontaneous rates out of
the node's compartment, and a contact channel whose hazard is the node's total contact hazard (the shared definition
in `algorithms/common.jl`, which covers `:infection` and `:contact_trace` with catalysts named by `via`). The
channels' absolute firing times live in a mutable binary min-heap. When a channel fires, the concrete transition is
chosen in proportion to the contributing hazards.

After an event the fired node's channels get fresh exponential clocks (its compartment, and so its channel identity,
changed), and the contact channels of its neighbours are rescaled with the Gibson–Bruck identity
t_new = t + (a_old / a_new)(t_old − t). Network updates rescale the two endpoints; a rate change rescales every
channel; a state change gives fresh clocks to the moved nodes and rescales their neighbours. All of these are exact.

Multiplex networks (`MultiplexGraph`, design §C.3, WP26): the contact hazard is the shared layer-weighted one, so
contact transition j acts through layer ℓ at the (layer × contact) rate
`rates[j] · [layer_j ∈ (:all, name_ℓ)] · layer_rates[ℓ]` (a transition on a named layer acts on that layer only),
exactly as in DirectSSA. After an event the neighbours of the fired node in every layer of positive weight are rescaled. A
neighbour linked in several layers is visited once per layer; the rescaling is idempotent (an unchanged hazard keeps
its clock and draws no random number), so this is exact.

Graph processes (`DynamicGraph`, src/processes/, WP26): one more channel, whose hazard is the process rate. When it
fires, the process changes the run's own copy of the graph and returns the nodes whose neighbourhoods changed (at
most four for neighbour exchange); their contact channels are rescaled and the process channel gets a fresh clock.
The process rate is re-read after every event (Gibson–Bruck again, so a rate that depends on the state stays exact).
Once no catalyst is left and no spontaneous transition or intervention can fire, nothing the graph does can change
the epidemic, and the run ends there (the final snapshot is recorded at the end of the time span as usual).
=#

"""
    NextReaction <: OutbreakAlgorithm

Gibson–Bruck next-reaction method with two channels per node (spontaneous and contact) in a binary heap: O(k̄ log N)
per event. Supports static, time-varying (`TimeVaryingNetwork`) and multiplex (`MultiplexGraph`) networks, graph
processes (`DynamicGraph`, e.g. neighbour exchange, with one more channel for the process events) and
interventions.
"""
struct NextReaction <: OutbreakAlgorithm end

_supports_interventions(::NextReaction) = true

const _SPONTANEOUS_CHANNEL = 1
const _INFECTION_CHANNEL = 2

# Channel storage for one run: 2n node channels and one graph-process channel (event id 2n + 1).
struct _NRChannels
    heap::MutableBinaryMinHeap{Tuple{Float64, Int}}
    handles::Vector{Int}
    hazards::Vector{Float64}
    times::Vector{Float64}
    tally::Vector{Float64}
    n::Int
end

_NRChannels(n::Int, rm::_RunModel) =
    _NRChannels(MutableBinaryMinHeap{Tuple{Float64, Int}}(), zeros(Int, 2n + 1), zeros(Float64, 2n + 1),
                fill(Inf, 2n + 1), _tally_buffer(rm), n)

function _simulate_impl(::NextReaction, spec::OutbreakSpec, rng::AbstractRNG, seed::_RecordedSeed, keep::Symbol,
                        plan::InterventionPlan = InterventionPlan())
    g, layers, weights, updates = _prepare_network(spec.network)
    return _nr_run(spec, rng, seed, keep, plan, g, layers, weights, updates, nothing)
end

function _simulate_impl(::NextReaction, spec::OutbreakSpec{<:DynamicGraph}, rng::AbstractRNG, seed::_RecordedSeed,
                        keep::Symbol, plan::InterventionPlan = InterventionPlan())
    g, proc = _prepare_dynamic(spec.network, rng)
    return _nr_run(spec, rng, seed, keep, plan, g, (g,), (1.0,), nothing, proc)
end

# Function barrier: the loop is compiled for the concrete graph, layer and process types (`proc === nothing` when
# the network has no graph process).
function _nr_run(spec::OutbreakSpec, rng::AbstractRNG, seed::_RecordedSeed, keep::Symbol, plan::InterventionPlan,
                 g, layers, weights, updates, proc)
    n = nv(g)

    rm = _RunModel(spec.model, spec.network)
    node_state = initial_state(spec, rng)
    state = _initial_outbreak_state(rm, node_state)
    ch = _NRChannels(n, rm)

    t_now = spec.tspan[1]
    t_end = spec.tspan[2]
    rec = _Recorder(keep)
    _record!(rec, t_now, state.counts)
    sched = _Schedule(updates, plan, t_now)

    for v in 1:n
        _nr_refresh_spontaneous!(ch, v, rm, node_state, t_now, rng; fresh = true)
        _nr_refresh_contact!(ch, v, layers, weights, rm, node_state, t_now, rng; fresh = true)
    end
    _nr_refresh_process!(ch, proc, t_now, rng; fresh = true)
    let t = t_now
        on_change = touched -> _nr_after_intervention!(ch, touched, layers, weights, rm, node_state, proc, t, rng)
        _check_thresholds!(sched, state, rm, n, rng, on_change) && _record!(rec, t_now, state.counts)
    end

    while true
        t_next, event_id = isempty(ch.heap) ? (Inf, 0) : first(ch.heap)

        # --- scheduled network updates and interventions come first when due (M4, M5) ---
        t_sched = _next_scheduled_time(sched)
        if t_sched <= t_next
            (isfinite(t_sched) && t_sched <= t_end) || break
            if !isfinite(t_next) && !_interventions_pending(sched) && _no_catalysts(rm, state.counts)
                break     # absorbing: no hazard can ever become positive
            end
            t_now = t_sched
            let t = t_now
                on_update = upd -> for v in (upd.src, upd.dst)
                    _nr_refresh_contact!(ch, v, layers, weights, rm, node_state, t, rng; fresh = false)
                end
                on_change = touched ->
                    _nr_after_intervention!(ch, touched, layers, weights, rm, node_state, proc, t, rng)
                if _apply_due!(sched, t, g, state, rm, n, rng, on_update, on_change)
                    _record!(rec, t, state.counts)
                    _check_thresholds!(sched, state, rm, n, rng, on_change) && _record!(rec, t, state.counts)
                end
            end
            continue
        end
        (isfinite(t_next) && t_next <= t_end) || break
        t_now = t_next

        # --- a graph-process event: rewire, rescale the touched nodes, fresh clock for the process ---
        if event_id == _process_event_id(n)
            _epidemic_absorbing(rm, state.counts, sched) && break
            for v in _process_fire!(proc, g, node_state, rng)
                _nr_refresh_contact!(ch, v, layers, weights, rm, node_state, t_now, rng; fresh = false)
            end
            _nr_refresh_process!(ch, proc, t_now, rng; fresh = true)
            continue
        end

        # --- fire ---
        fired_node = _channel_node(event_id, n)
        c = node_state[fired_node]
        fired_j = if _channel_kind(event_id, n) == _SPONTANEOUS_CHANNEL
            _sample_spontaneous_transition(rm, c, rng)
        else
            _sample_contact_transition!(ch.tally, fired_node, layers, weights, node_state, rm, rng)
        end
        _fire!(state, rm, fired_node, fired_j)
        _record!(rec, t_now, state.counts)
        _log_event!(rec, t_now, fired_j, fired_node)

        _nr_refresh_node!(ch, fired_node, layers, weights, rm, node_state, t_now, rng)
        _nr_refresh_process!(ch, proc, t_now, rng; fresh = false)
        if !isempty(sched.thresholds)
            let t = t_now
                on_change = touched ->
                    _nr_after_intervention!(ch, touched, layers, weights, rm, node_state, proc, t, rng)
                _check_thresholds!(sched, state, rm, n, rng, on_change) && _record!(rec, t, state.counts)
            end
        end
    end

    return _trajectory(rec, rm, state, t_end, seed, :NextReaction)
end

# --- channel bookkeeping ---

_spontaneous_event_id(v::Integer) = Int(v)
_infection_event_id(v::Integer, n::Integer) = Int(n + v)
_process_event_id(n::Integer) = Int(2n + 1)
_channel_kind(event_id::Integer, n::Integer) = event_id <= n ? _SPONTANEOUS_CHANNEL : _INFECTION_CHANNEL
_channel_node(event_id::Integer, n::Integer) = event_id <= n ? Int(event_id) : Int(event_id - n)

function _nr_refresh_spontaneous!(ch::_NRChannels, v::Integer, rm::_RunModel, node_state::Vector{Int},
                                  t_now::Float64, rng::AbstractRNG; fresh::Bool)
    _reschedule_channel!(ch.heap, ch.handles, ch.hazards, ch.times, _spontaneous_event_id(v),
                         rm.spont_total[node_state[v]], t_now, rng; fresh = fresh)
end

function _nr_refresh_contact!(ch::_NRChannels, v::Integer, layers, weights, rm::_RunModel,
                              node_state::Vector{Int}, t_now::Float64, rng::AbstractRNG; fresh::Bool)
    h = _contact_hazard!(ch.tally, v, layers, weights, node_state, rm)
    _reschedule_channel!(ch.heap, ch.handles, ch.hazards, ch.times, _infection_event_id(v, ch.n),
                         h, t_now, rng; fresh = fresh)
end

# The graph-process channel (nothing to do without a process).
_nr_refresh_process!(::_NRChannels, ::Nothing, ::Float64, ::AbstractRNG; fresh::Bool) = nothing
function _nr_refresh_process!(ch::_NRChannels, proc, t_now::Float64, rng::AbstractRNG; fresh::Bool)
    _reschedule_channel!(ch.heap, ch.handles, ch.hazards, ch.times, _process_event_id(ch.n),
                         _process_rate(proc), t_now, rng; fresh = fresh)
end

# Node v changed compartment: fresh clocks for its own channels, Gibson–Bruck rescaling for its neighbours in every
# layer that carries weight (one layer for a single graph, in the graph's neighbour order).
function _nr_refresh_node!(ch::_NRChannels, v::Integer, layers, weights, rm::_RunModel,
                           node_state::Vector{Int}, t_now::Float64, rng::AbstractRNG)
    _nr_refresh_spontaneous!(ch, v, rm, node_state, t_now, rng; fresh = true)
    _nr_refresh_contact!(ch, v, layers, weights, rm, node_state, t_now, rng; fresh = true)
    for l in eachindex(layers)
        weights[l] > 0 || continue
        for u in neighbors(layers[l], v)
            _nr_refresh_contact!(ch, u, layers, weights, rm, node_state, t_now, rng; fresh = false)
        end
    end
    return nothing
end

function _nr_after_intervention!(ch::_NRChannels, touched, layers, weights, rm::_RunModel,
                                 node_state::Vector{Int}, proc, t_now::Float64, rng::AbstractRNG)
    if touched === nothing           # rates changed: every hazard may differ; rescale all clocks
        for v in 1:ch.n
            _nr_refresh_spontaneous!(ch, v, rm, node_state, t_now, rng; fresh = false)
            _nr_refresh_contact!(ch, v, layers, weights, rm, node_state, t_now, rng; fresh = false)
        end
    else
        for v in touched
            _nr_refresh_node!(ch, v, layers, weights, rm, node_state, t_now, rng)
        end
    end
    _nr_refresh_process!(ch, proc, t_now, rng; fresh = false)
    return nothing
end

function _reschedule_channel!(heap::MutableBinaryMinHeap{Tuple{Float64, Int}},
                              handles::Vector{Int}, hazards::Vector{Float64},
                              scheduled_times::Vector{Float64}, event_id::Int,
                              new_hazard::Float64, t_now::Float64,
                              rng::AbstractRNG; fresh::Bool)
    old_hazard = hazards[event_id]
    old_time = scheduled_times[event_id]
    # Nothing to do (and no random draw) when a silent channel stays silent, or a live one keeps its hazard.
    new_hazard <= 0.0 && old_hazard <= 0.0 && return nothing
    !fresh && new_hazard == old_hazard && old_time > t_now && return nothing
    new_time = if new_hazard <= 0.0
        Inf
    elseif fresh || old_hazard <= 0.0 || !isfinite(old_time) || old_time <= t_now
        t_now + randexp(rng) / new_hazard
    else
        t_now + (old_hazard / new_hazard) * (old_time - t_now)
    end

    hazards[event_id] = new_hazard
    scheduled_times[event_id] = new_time

    handle = handles[event_id]
    if handle == 0
        if isfinite(new_time)
            handles[event_id] = push!(heap, (new_time, event_id))
        end
    else
        update!(heap, handle, (new_time, event_id))
    end
    return nothing
end
