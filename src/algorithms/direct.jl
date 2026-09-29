#=
algorithms/direct.jl

Gillespie direct method (SSA) for compartmental epidemics on a contact network.

Before every event the total rate is recomputed from scratch: the spontaneous part from the compartment counts, and
the contact part by a sweep over the nodes that have a contact transition, tallying their catalyst neighbours (the
compartments named in `via`, weighted by layer on a multiplex network; see `algorithms/common.jl`). The cost is
O(N·k̄) per event, which is fine for N ≲ 10⁴ and makes DirectSSA the reference implementation for the faster
samplers. It supports static, time-varying and multiplex networks, and interventions.
=#

"""
    DirectSSA <: OutbreakAlgorithm

Gillespie's direct method with a full propensity sweep per event (O(N·k̄) per event). Supports static,
time-varying (`TimeVaryingNetwork`) and multiplex (`MultiplexGraph`) networks, and interventions. It is the
simplest and most general sampler and serves as the reference for the others.
"""
struct DirectSSA <: OutbreakAlgorithm end

_supports_interventions(::DirectSSA) = true

function _simulate_impl(::DirectSSA, spec::OutbreakSpec, rng::AbstractRNG, seed::_RecordedSeed, keep::Symbol,
                        plan::InterventionPlan = InterventionPlan())
    g, layers, weights, updates = _prepare_network(spec.network)
    n = spec.network isa MultiplexGraph ? nv(spec.network) : nv(g)
    return _direct_run(spec, rng, seed, keep, plan, g, layers, weights, updates, n)
end

# Function barrier: the loop is compiled for the concrete graph and layer types.
function _direct_run(spec::OutbreakSpec, rng::AbstractRNG, seed::_RecordedSeed, keep::Symbol, plan::InterventionPlan,
                     g, layers, weights, updates, n::Int)

    rm = _RunModel(spec.model, spec.network)
    C = rm.C
    node_state = initial_state(spec, rng)
    state = _initial_outbreak_state(rm, node_state)

    t_now = spec.tspan[1]
    t_end = spec.tspan[2]
    rec = _Recorder(keep)
    _record!(rec, t_now, state.counts)

    sched = _Schedule(updates, plan, t_now)
    noop(_) = nothing
    _check_thresholds!(sched, state, rm, n, rng, noop) && _record!(rec, t_now, state.counts)

    tally = _tally_buffer(rm)
    node_hazard = zeros(Float64, n)
    has_contacts = _has_contacts(rm)

    while true
        # --- propensities ---
        spontaneous_rate = 0.0
        @inbounds for c in 1:C
            spontaneous_rate += state.counts[c] * rm.spont_total[c]
        end
        contact_rate = 0.0
        if has_contacts
            fill!(node_hazard, 0.0)
            @inbounds for v in 1:n
                h = _contact_hazard!(tally, v, layers, weights, node_state, rm)
                node_hazard[v] = h
                contact_rate += h
            end
        end
        total_rate = spontaneous_rate + contact_rate

        # --- next reaction time, merged with the scheduled queue (M4, M5) ---
        t_next = total_rate > 0 ? t_now + randexp(rng) / total_rate : Inf
        t_sched = _next_scheduled_time(sched)
        if t_sched <= t_next
            (isfinite(t_sched) && t_sched <= t_end) || break
            # Nothing can ever happen again: zero rate, no catalyst left and no intervention pending.
            if total_rate == 0 && !_interventions_pending(sched) && _no_catalysts(rm, state.counts)
                break
            end
            t_now = t_sched
            if _apply_due!(sched, t_now, g, state, rm, n, rng, noop, noop)
                _record!(rec, t_now, state.counts)
                _check_thresholds!(sched, state, rm, n, rng, noop) && _record!(rec, t_now, state.counts)
            end
            continue    # rates are constant between scheduled times, so redrawing from t_now is exact
        end
        (isfinite(t_next) && t_next <= t_end) || break
        t_now = t_next

        # --- choose the event ---
        u = rand(rng) * total_rate
        local fired_node::Int
        local fired_j::Int
        if u < spontaneous_rate
            # Compartment ∝ counts[c]·spont_total[c], then a uniform node in it, then a transition ∝ its rate.
            picked_c = 0
            cum = 0.0
            last_c = 0
            @inbounds for c in 1:C
                w = state.counts[c] * rm.spont_total[c]
                w > 0 || continue
                last_c = c
                cum += w
                if u < cum
                    picked_c = c
                    break
                end
            end
            picked_c == 0 && (picked_c = last_c)
            fired_node = _sample_node_in_compartment(node_state, picked_c, state.counts[picked_c], rng)
            fired_j = _sample_spontaneous_transition(rm, picked_c, rng)
        else
            target = u - spontaneous_rate
            cum = 0.0
            picked_v = 0
            @inbounds for v in 1:n
                h = node_hazard[v]
                h > 0 || continue
                cum += h
                if target < cum
                    picked_v = v
                    break
                end
            end
            picked_v == 0 && (picked_v = _last_nonzero(node_hazard))
            fired_node = picked_v
            # Same catalyst weights as the hazard sweep (M2).
            fired_j = _sample_contact_transition!(tally, picked_v, layers, weights, node_state, rm, rng)
        end

        # --- apply and record ---
        _fire!(state, rm, fired_node, fired_j)
        _record!(rec, t_now, state.counts)
        _log_event!(rec, t_now, fired_j, fired_node)
        _check_thresholds!(sched, state, rm, n, rng, noop) && _record!(rec, t_now, state.counts)
    end

    return _trajectory(rec, rm, state, t_end, seed, :DirectSSA)
end

# --- helpers ---

function _sample_node_in_compartment(node_state::Vector{Int}, c::Int, count::Int, rng::AbstractRNG)
    target = rand(rng, 1:count)
    seen = 0
    @inbounds for v in eachindex(node_state)
        if node_state[v] == c
            seen += 1
            seen == target && return v
        end
    end
    throw(ErrorException("internal error: compartment $(c) holds fewer than $(count) nodes"))
end

function _last_nonzero(v::AbstractVector{<:Real})
    @inbounds for i in length(v):-1:1
        v[i] > 0 && return i
    end
    return 0
end
