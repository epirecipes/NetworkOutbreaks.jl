#=
Gibson--Bruck next-reaction method with node-level channels.

Each node owns up to two reaction channels: a spontaneous channel whose
hazard is the sum of all spontaneous transitions out of the node's current
compartment, and an infection channel whose hazard sums infection pressure
from infectious neighbours. Infection channels are node-level aggregates; on
firing, the concrete transition is sampled from the contributing hazards.
=#

struct NextReaction <: OutbreakAlgorithm end

const _SPONTANEOUS_CHANNEL = 1
const _INFECTION_CHANNEL = 2

function _simulate_impl(::NextReaction, spec::OutbreakSpec, seed::UInt64, keep::Symbol, interventions::InterventionPlan = InterventionPlan())
    rng = Xoshiro(seed)
    model = spec.model
    network = spec.network

    network isa MultiplexNetwork &&
        throw(ArgumentError("NextReaction does not yet support MultiplexNetwork; use DirectSSA"))

    # Time-varying network support
    is_tvn = network isa TimeVaryingNetwork
    g = is_tvn ? deepcopy(network.graph) : _outbreak_graph(network)
    updates = is_tvn ? network.updates : nothing
    next_update_idx = 1

    n = nv(g)
    C = ncompartments(model)

    node_state = initial_state(spec, rng)
    state = OutbreakState(model, node_state)

    spont_by_src = [OutbreakTransition[] for _ in 1:C]
    infection_by_src = [OutbreakTransition[] for _ in 1:C]
    for tr in model.transitions
        src_idx = model.index_of[tr.from]
        if tr.type == :spontaneous
            push!(spont_by_src[src_idx], tr)
        else
            push!(infection_by_src[src_idx], tr)
        end
    end

    spont_total_rate = zeros(Float64, C)
    for i in 1:C
        spont_total_rate[i] = sum((tr.rate for tr in spont_by_src[i]); init = 0.0)
    end

    via_mask = Dict{OutbreakTransition, BitVector}()
    for tr in model.transitions
        tr.type == :infection || continue
        mask = falses(C)
        if isempty(tr.via)
            for i in 1:C
                model.infectious[i] && (mask[i] = true)
            end
        else
            for sym in tr.via
                mask[model.index_of[sym]] = true
            end
        end
        via_mask[tr] = mask
    end

    n_channels = 2n
    heap = MutableBinaryMinHeap{Tuple{Float64, Int}}()
    handles = zeros(Int, n_channels)
    hazards = zeros(Float64, n_channels)
    scheduled_times = fill(Inf, n_channels)

    t_now = spec.tspan[1]
    t_end = spec.tspan[2]

    for v in 1:n
        _refresh_spontaneous_channel!(heap, handles, hazards, scheduled_times,
                                      v, node_state, spont_total_rate, t_now, rng;
                                      fresh = true)
        _refresh_infection_channel!(heap, handles, hazards, scheduled_times,
                                    v, g, model, node_state, infection_by_src,
                                    via_mask, t_now, rng; fresh = true)
    end

    times_buf  = Float64[t_now]
    counts_buf = Vector{Vector{Int}}()
    push!(counts_buf, copy(state.counts))
    events_buf = OutbreakEvent[]

    while t_now < t_end
        isempty(heap) && break
        t_next, event_id = first(heap)
        isfinite(t_next) || break

        # --- time-varying network: apply any graph updates before the next reaction ---
        if is_tvn
            t_update = (next_update_idx <= length(updates)) ?
                       updates[next_update_idx].t : Inf
            if t_update <= t_next
                t_now = min(t_update, t_end)
                t_now >= t_end && break
                while next_update_idx <= length(updates) &&
                        updates[next_update_idx].t <= t_now
                    upd = updates[next_update_idx]
                    if upd.action == :add
                        add_edge!(g, upd.src, upd.dst)
                    else
                        rem_edge!(g, upd.src, upd.dst)
                    end
                    # Reschedule infection channels for both endpoints using the
                    # Gibson–Bruck identity (fresh = false preserves old random clock).
                    for v in (upd.src, upd.dst)
                        _refresh_infection_channel!(heap, handles, hazards, scheduled_times,
                                                    v, g, model, node_state,
                                                    infection_by_src, via_mask, t_now, rng;
                                                    fresh = false)
                    end
                    next_update_idx += 1
                end
                continue  # re-peek the heap with rescheduled channels
            end
        end

        if t_next > t_end
            break
        end
        t_now = t_next

        fired_node = _channel_node(event_id, n)
        fired_kind = _channel_kind(event_id, n)
        src_idx = node_state[fired_node]
        fired_tr = if fired_kind == _SPONTANEOUS_CHANNEL
            _sample_spontaneous_transition(spont_by_src[src_idx], rng)
        else
            _sample_infection_transition(fired_node, g, model, node_state,
                                         infection_by_src[src_idx], via_mask, rng)
        end

        old_idx = src_idx
        new_idx = model.index_of[fired_tr.to]
        node_state[fired_node] = new_idx
        state.counts[old_idx] -= 1
        state.counts[new_idx] += 1
        if model.infectious[new_idx] && !model.infectious[old_idx]
            state.infection_counts[fired_node] += 1
        end

        push!(times_buf, t_now)
        push!(counts_buf, copy(state.counts))
        if keep == :events
            push!(events_buf,
                  OutbreakEvent(t_now,
                                _transition_index(model, fired_tr),
                                fired_node))
        end

        # The fired node changed compartment, so its node-level channel identities
        # are rebuilt with fresh exponential clocks. Neighbour infection hazards
        # changed only because one adjacent catalyst state changed, so the
        # Gibson--Bruck rescheduling identity preserves their old random clocks.
        _refresh_spontaneous_channel!(heap, handles, hazards, scheduled_times,
                                      fired_node, node_state, spont_total_rate,
                                      t_now, rng; fresh = true)
        _refresh_infection_channel!(heap, handles, hazards, scheduled_times,
                                    fired_node, g, model, node_state,
                                    infection_by_src, via_mask, t_now, rng;
                                    fresh = true)
        for u in neighbors(g, fired_node)
            _refresh_infection_channel!(heap, handles, hazards, scheduled_times,
                                        u, g, model, node_state,
                                        infection_by_src, via_mask, t_now, rng;
                                        fresh = false)
        end
    end

    if times_buf[end] < t_end
        push!(times_buf, t_end)
        push!(counts_buf, copy(state.counts))
    end

    counts_mat = Matrix{Int}(undef, C, length(times_buf))
    @inbounds for k in 1:length(times_buf)
        counts_mat[:, k] = counts_buf[k]
    end

    return OutbreakTrajectory(model, times_buf, counts_mat,
                              copy(state.infection_counts),
                              events_buf, seed, :NextReaction)
end

# --- channel bookkeeping ---

_spontaneous_event_id(v::Integer) = Int(v)
_infection_event_id(v::Integer, n::Integer) = Int(n + v)
_channel_kind(event_id::Integer, n::Integer) = event_id <= n ? _SPONTANEOUS_CHANNEL : _INFECTION_CHANNEL
_channel_node(event_id::Integer, n::Integer) = event_id <= n ? Int(event_id) : Int(event_id - n)

function _refresh_spontaneous_channel!(heap, handles, hazards, scheduled_times,
                                       v::Integer, node_state::Vector{Int},
                                       spont_total_rate::Vector{Float64},
                                       t_now::Float64, rng::AbstractRNG;
                                       fresh::Bool)
    event_id = _spontaneous_event_id(v)
    new_hazard = spont_total_rate[node_state[v]]
    _reschedule_channel!(heap, handles, hazards, scheduled_times,
                         event_id, new_hazard, t_now, rng; fresh = fresh)
end

function _refresh_infection_channel!(heap, handles, hazards, scheduled_times,
                                     v::Integer, g::AbstractGraph,
                                     model::OutbreakModel,
                                     node_state::Vector{Int},
                                     infection_by_src,
                                     via_mask::Dict{OutbreakTransition, BitVector},
                                     t_now::Float64, rng::AbstractRNG;
                                     fresh::Bool)
    event_id = _infection_event_id(v, nv(g))
    new_hazard = _infection_hazard(v, g, model, node_state,
                                   infection_by_src[node_state[v]], via_mask)
    _reschedule_channel!(heap, handles, hazards, scheduled_times,
                         event_id, new_hazard, t_now, rng; fresh = fresh)
end

function _reschedule_channel!(heap::MutableBinaryMinHeap{Tuple{Float64, Int}},
                              handles::Vector{Int}, hazards::Vector{Float64},
                              scheduled_times::Vector{Float64}, event_id::Int,
                              new_hazard::Float64, t_now::Float64,
                              rng::AbstractRNG; fresh::Bool)
    old_hazard = hazards[event_id]
    old_time = scheduled_times[event_id]
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

# --- hazards and event sampling ---

function _infection_hazard(v::Integer, g::AbstractGraph, model::OutbreakModel,
                           node_state::Vector{Int}, trs::Vector{OutbreakTransition},
                           via_mask::Dict{OutbreakTransition, BitVector})
    isempty(trs) && return 0.0
    hazard = 0.0
    for tr in trs
        mask = via_mask[tr]
        n_via = 0
        for u in neighbors(g, v)
            mask[node_state[u]] && (n_via += 1)
        end
        hazard += tr.rate * n_via
    end
    return hazard
end

function _sample_spontaneous_transition(trs::Vector{OutbreakTransition}, rng::AbstractRNG)
    length(trs) == 1 && return trs[1]
    total = sum((tr.rate for tr in trs); init = 0.0)
    target = rand(rng) * total
    cum = 0.0
    for tr in trs
        cum += tr.rate
        target <= cum && return tr
    end
    return trs[end]
end

function _sample_infection_transition(v::Integer, g::AbstractGraph, model::OutbreakModel,
                                      node_state::Vector{Int},
                                      trs::Vector{OutbreakTransition},
                                      via_mask::Dict{OutbreakTransition, BitVector},
                                      rng::AbstractRNG)
    length(trs) == 1 && return trs[1]
    weights = zeros(Float64, length(trs))
    total = 0.0
    for (j, tr) in pairs(trs)
        mask = via_mask[tr]
        n_via = 0
        for u in neighbors(g, v)
            mask[node_state[u]] && (n_via += 1)
        end
        w = tr.rate * n_via
        weights[j] = w
        total += w
    end
    target = rand(rng) * total
    cum = 0.0
    for (j, tr) in pairs(trs)
        cum += weights[j]
        target <= cum && return tr
    end
    return trs[end]
end
