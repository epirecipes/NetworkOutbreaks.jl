#=
algorithms/direct.jl

Gillespie Direct method (SSA) for compartmental epidemics on a contact
network. Recomputes the total rate by summing per-node (spontaneous) and
per-(node × infectious-neighbour) (infection) contributions on each
event. This is `O(events × N)` worst-case; sufficient for `N ≲ 10⁴` and
the small-graph validation use-case targeted in Phase 1.

Future algorithms (Next-Reaction, Composition–Rejection, HAS) will share
the model/spec/event interface but maintain caches that update only on
local neighborhood changes.
=#

struct DirectSSA <: OutbreakAlgorithm end

function simulate(spec::OutbreakSpec;
                  algorithm::OutbreakAlgorithm = DirectSSA(),
                  seed::Integer = rand(UInt64),
                  keep::Symbol = :counts,
                  interventions::InterventionPlan = InterventionPlan())
    keep in (:counts, :events) ||
        throw(ArgumentError("keep must be :counts or :events"))
    if !isempty(interventions) && !(algorithm isa DirectSSA)
        throw(ArgumentError("interventions are currently only supported with DirectSSA"))
    end
    return _simulate_impl(algorithm, spec, UInt64(seed), keep, interventions)
end

function _simulate_impl(::DirectSSA, spec::OutbreakSpec, seed::UInt64, keep::Symbol,
                        interventions::InterventionPlan = InterventionPlan())
    rng = Xoshiro(seed)
    model = spec.model
    network = spec.network

    # Time-varying network support
    is_tvn = network isa TimeVaryingNetwork
    is_mpx = network isa MultiplexNetwork
    is_tvn && is_mpx &&
        throw(ArgumentError("MultiplexNetwork combined with TimeVaryingNetwork is not supported"))
    g = is_tvn ? deepcopy(network.graph) : _outbreak_graph(network)
    updates = is_tvn ? network.updates : nothing
    next_update_idx = 1

    # Layer view: a vector of (graph, weight) pairs the SSA loops over when
    # computing per-node infection hazards. For static / time-varying networks
    # this is a 1-element view over `g` with weight 1.0; for multiplex networks
    # each layer contributes its own neighbour set scaled by `layer_rates[ℓ]`.
    if is_mpx
        mpx_layers = network.layers
        mpx_weights = network.layer_rates
    else
        mpx_layers = (g,)
        mpx_weights = (1.0,)
    end

    n = nv(is_mpx ? network : g)
    C = ncompartments(model)

    node_state = initial_state(spec, rng)
    state = OutbreakState(model, node_state)

    # Pre-bucket transitions by source compartment, by type.
    spont_by_src = [OutbreakTransition[] for _ in 1:C]
    infection_by_src = [OutbreakTransition[] for _ in 1:C]
    for tr in model.transitions
        src_idx = model.index_of[tr.from]
        if tr.type == :spontaneous
            push!(spont_by_src[src_idx], tr)
        elseif tr.type in (:infection, :contact_trace)
            push!(infection_by_src[src_idx], tr)
        end
    end

    # Pre-compute per-source-compartment total spontaneous rate (per node)
    spont_total_rate = zeros(Float64, C)
    for i in 1:C
        spont_total_rate[i] = sum((tr.rate for tr in spont_by_src[i]); init = 0.0)
    end

    # Pre-compute per-transition via-mask: BitVector of length C marking
    # which compartments count as catalysts. Empty `via` ⇒ all infectious.
    via_mask = Dict{OutbreakTransition, BitVector}()
    for tr in model.transitions
        tr.type in (:infection, :contact_trace) || continue
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

    # Record initial state.
    times_buf  = Float64[spec.tspan[1]]
    counts_buf = Vector{Vector{Int}}()
    push!(counts_buf, copy(state.counts))
    events_buf = OutbreakEvent[]

    t_now = spec.tspan[1]
    t_end = spec.tspan[2]

    # Intervention tracking
    next_sched_idx = 1
    threshold_fired = falses(length(interventions.thresholds))

    # Reusable buffer for per-node infection-source weights.
    # For each node v, accumulate the layer-weighted count of infectious
    # neighbours partitioned by their compartment, so per-edge rates can
    # multiply by the per-source-comp transition rate.
    infectious_neighbour_weight = zeros(Float64, C)

    while t_now < t_end
        # --- propensity sweep ---
        # Compute the total propensity over all (node, transition)
        # combinations, plus a flat list for sampling.
        total_rate = 0.0

        # Spontaneous part: rate per node = spont_total_rate[state[v]].
        # Total = sum over compartments c of counts[c] * spont_total_rate[c].
        spontaneous_rate = 0.0
        for c in 1:C
            spontaneous_rate += state.counts[c] * spont_total_rate[c]
        end

        # Infection part: for each susceptible-with-an-infection-rule node v
        # and each of its infectious neighbours, add the transition's rate
        # weighted by the per-layer multiplier (1.0 for non-multiplex).
        infection_node_hazard = zeros(Float64, n)  # only nonzero where applicable
        infection_rate_total  = 0.0
        if any(!isempty, infection_by_src)
            @inbounds for v in 1:n
                src_idx = node_state[v]
                trs = infection_by_src[src_idx]
                isempty(trs) && continue
                # Tally infectious neighbours by compartment, summed across
                # layers and weighted by each layer's rate multiplier.
                fill!(infectious_neighbour_weight, 0.0)
                any_infectious = false
                for (l_idx, g_l) in pairs(mpx_layers)
                    w_l = mpx_weights[l_idx]
                    w_l > 0 || continue
                    for u in neighbors(g_l, v)
                        nc = node_state[u]
                        if model.infectious[nc]
                            infectious_neighbour_weight[nc] += w_l
                            any_infectious = true
                        end
                    end
                end
                any_infectious || continue
                # Sum hazard contributions: for each infection transition
                # `from → to` available at this source, add rate × Σ-via
                # weighted infectious neighbours.
                hazard_v = 0.0
                for tr in trs
                    mask = via_mask[tr]
                    w_via = 0.0
                    for i in 1:C
                        mask[i] && (w_via += infectious_neighbour_weight[i])
                    end
                    hazard_v += tr.rate * w_via
                end
                infection_node_hazard[v] = hazard_v
                infection_rate_total += hazard_v
            end
        end

        total_rate = spontaneous_rate + infection_rate_total
        total_rate > 0 || break  # absorbing state

        # --- time advance ---
        Δt = randexp(rng) / total_rate
        t_next = t_now + Δt

        # --- time-varying network: apply any updates that fire before t_next ---
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
                    next_update_idx += 1
                end
                continue  # re-enter loop: recompute total_rate with updated graph
            end
        end

        # --- scheduled interventions: apply any that fire before t_next ---
        if next_sched_idx <= length(interventions.scheduled)
            iv = interventions.scheduled[next_sched_idx]
            t_iv = _intervention_time(iv)
            if t_iv <= t_next
                t_now = min(t_iv, t_end)
                t_now >= t_end && break
                while next_sched_idx <= length(interventions.scheduled) &&
                        _intervention_time(interventions.scheduled[next_sched_idx]) <= t_now
                    _apply_intervention!(interventions.scheduled[next_sched_idx],
                                         model, node_state, state, n,
                                         spont_by_src, infection_by_src,
                                         spont_total_rate, via_mask, rng)
                    next_sched_idx += 1
                end
                push!(times_buf, t_now)
                push!(counts_buf, copy(state.counts))
                continue  # recompute propensities
            end
        end

        if t_next > t_end
            break
        end
        t_now = t_next

        # --- choose which event ---
        u = rand(rng) * total_rate
        local fired_node::Int
        local fired_tr::OutbreakTransition
        if u < spontaneous_rate
            # Sample a (compartment, transition) pair, then a node.
            # Two-stage: pick compartment proportional to counts[c]*spont_total_rate[c]
            cum = 0.0
            picked_c = 0
            picked_tr_idx = 0
            for c in 1:C
                w = state.counts[c] * spont_total_rate[c]
                if w > 0 && u < cum + w
                    picked_c = c
                    # Pick transition within this compartment.
                    local_u = (u - cum) / state.counts[c]
                    cum2 = 0.0
                    for (j, tr) in pairs(spont_by_src[c])
                        if local_u < cum2 + tr.rate
                            picked_tr_idx = j
                            break
                        end
                        cum2 += tr.rate
                    end
                    if picked_tr_idx == 0
                        picked_tr_idx = length(spont_by_src[c])
                    end
                    break
                end
                cum += w
            end
            # Pick a node uniformly from compartment picked_c.
            fired_node = _sample_node_in_compartment(node_state, picked_c, state.counts[picked_c], rng)
            fired_tr = spont_by_src[picked_c][picked_tr_idx]
        else
            # Infection: sample a node weighted by infection_node_hazard.
            target = u - spontaneous_rate
            cum = 0.0
            picked_v = 0
            @inbounds for v in 1:n
                h = infection_node_hazard[v]
                h > 0 || continue
                if target < cum + h
                    picked_v = v
                    break
                end
                cum += h
            end
            picked_v == 0 && (picked_v = _last_nonzero(infection_node_hazard))
            fired_node = picked_v
            # Pick which infection transition (if multiple share the source).
            src_idx = node_state[picked_v]
            trs = infection_by_src[src_idx]
            fired_tr = if length(trs) == 1
                trs[1]
            else
                fill!(infectious_neighbour_weight, 0.0)
                for (l_idx, g_l) in pairs(mpx_layers)
                    w_l = mpx_weights[l_idx]
                    w_l > 0 || continue
                    for u2 in neighbors(g_l, picked_v)
                        infectious_neighbour_weight[node_state[u2]] += w_l
                    end
                end
                weights = Float64[]
                for tr in trs
                    mask = via_mask[tr]
                    w_via = 0.0
                    for i in 1:C
                        mask[i] && (w_via += infectious_neighbour_weight[i])
                    end
                    push!(weights, tr.rate * w_via)
                end
                trs[sample(rng, 1:length(trs), Weights(weights))]
            end
        end

        # --- apply event ---
        old_idx = node_state[fired_node]
        new_idx = model.index_of[fired_tr.to]
        node_state[fired_node] = new_idx
        state.counts[old_idx] -= 1
        state.counts[new_idx] += 1
        if model.infectious[new_idx] && !model.infectious[old_idx]
            state.infection_counts[fired_node] += 1
        end

        # Record snapshot.
        push!(times_buf, t_now)
        push!(counts_buf, copy(state.counts))
        if keep == :events
            push!(events_buf,
                  OutbreakEvent(t_now,
                                _transition_index(model, fired_tr),
                                fired_node))
        end

        # --- threshold interventions: check after each event ---
        for (ti, tiv) in enumerate(interventions.thresholds)
            threshold_fired[ti] && continue
            c_idx = model.index_of[tiv.compartment]
            cnt = state.counts[c_idx]
            triggered = (tiv.direction == :above && cnt >= tiv.threshold) ||
                        (tiv.direction == :below && cnt <= tiv.threshold)
            if triggered
                threshold_fired[ti] = true
                _apply_intervention!(tiv.action, model, node_state, state, n,
                                     spont_by_src, infection_by_src,
                                     spont_total_rate, via_mask, rng)
                push!(times_buf, t_now)
                push!(counts_buf, copy(state.counts))
            end
        end
    end

    # Final snapshot at t_end if we exited via time horizon.
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
                              events_buf, seed, :DirectSSA)
end

# --- helpers ---

function _sample_node_in_compartment(node_state::Vector{Int}, c::Int,
                                     count::Int, rng::AbstractRNG)
    target = rand(rng, 1:count)
    seen = 0
    @inbounds for v in eachindex(node_state)
        if node_state[v] == c
            seen += 1
            seen == target && return v
        end
    end
    return last(eachindex(node_state))  # defensive fallback
end

function _transition_index(model::OutbreakModel, tr::OutbreakTransition)
    @inbounds for (k, t) in pairs(model.transitions)
        if t.from == tr.from && t.to == tr.to && t.rate == tr.rate &&
                t.type == tr.type && t.via == tr.via
            return k
        end
    end
    return 0
end

function _last_nonzero(v::AbstractVector{<:Real})
    @inbounds for i in length(v):-1:1
        v[i] > 0 && return i
    end
    return 0
end

# --- intervention application ---

function _apply_intervention!(iv::ScheduledRateChange, model, node_state, state, n,
                               spont_by_src, infection_by_src, spont_total_rate,
                               via_mask, rng)
    from_idx = model.index_of[iv.from]
    for (j, tr) in pairs(model.transitions)
        if tr.from == iv.from && tr.to == iv.to && tr.type == iv.type
            new_tr = OutbreakTransition(tr.from, tr.to, iv.new_rate, tr.type; via = tr.via)
            model.transitions[j] = new_tr
            # Update caches
            if tr.type == :spontaneous
                for (k, st) in pairs(spont_by_src[from_idx])
                    if st === tr
                        spont_by_src[from_idx][k] = new_tr
                        break
                    end
                end
                spont_total_rate[from_idx] = sum(t.rate for t in spont_by_src[from_idx]; init=0.0)
            else
                for (k, it) in pairs(infection_by_src[from_idx])
                    if it === tr
                        infection_by_src[from_idx][k] = new_tr
                        # Update via_mask for the new transition
                        delete!(via_mask, tr)
                        mask = falses(length(model.compartments))
                        if isempty(new_tr.via)
                            for i in eachindex(model.infectious)
                                model.infectious[i] && (mask[i] = true)
                            end
                        else
                            for sym in new_tr.via
                                mask[model.index_of[sym]] = true
                            end
                        end
                        via_mask[new_tr] = mask
                        break
                    end
                end
            end
            return
        end
    end
end

function _apply_intervention!(iv::ScheduledStateChange, model, node_state, state, n,
                               spont_by_src, infection_by_src, spont_total_rate,
                               via_mask, rng)
    target_idx = model.index_of[iv.compartment]
    n_to_move = round(Int, iv.fraction * n)
    n_to_move <= 0 && return
    # Collect eligible nodes (not already in target compartment)
    eligible = Int[]
    @inbounds for v in 1:n
        node_state[v] != target_idx && push!(eligible, v)
    end
    n_to_move = min(n_to_move, length(eligible))
    n_to_move <= 0 && return
    # Shuffle and pick
    selected = sample(rng, eligible, n_to_move; replace = false)
    @inbounds for v in selected
        old_idx = node_state[v]
        node_state[v] = target_idx
        state.counts[old_idx] -= 1
        state.counts[target_idx] += 1
    end
end
