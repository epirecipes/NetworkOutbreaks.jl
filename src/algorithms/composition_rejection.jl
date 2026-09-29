#=
algorithms/composition_rejection.jl

Composition–Rejection SSA (Slepoy, Thompson & Plimpton 2008,
"A constant-time kinetic Monte Carlo algorithm for simulation of large
biochemical reaction networks", J. Chem. Phys. 128, 205101).

Each node carries a single aggregate hazard a_v = spontaneous rate + contact hazard (the shared definition in
`algorithms/common.jl`). Nodes are binned into logarithmic buckets:
    bucket b  covers  [log_base · 2^(b-1),  log_base · 2^b)

Total rate  Λ = Σ_b  S_b  where  S_b  is the bucket sum.

Event selection:
  1. Composition: pick bucket b ∝ S_b (linear scan over ≤ 64 buckets).
  2. Rejection:   pick a node uniformly from bucket b; accept with probability a_v / (log_base · 2^b).
                  Expected acceptance ≥ 0.5. After 256 rejections the node is chosen exactly (∝ a_v) instead.
                  Hazards outside buckets 1…64 are clamped into them; bucket 1 stays exact (its ceiling bounds any
                  smaller hazard) and bucket 64, whose members may exceed its ceiling, is always sampled exactly.
  3. Time:        Δt = Exp(1) / Λ.

Bucket membership is maintained with O(1) swap-and-pop removal, so the per-event cost is O(Δ · log(a_max/a_min))
where Δ is the maximum node degree.

Time-varying networks, multiplex networks and interventions are not supported (use DirectSSA, NextReaction or HAS).
=#

"""
    CompositionRejection <: OutbreakAlgorithm

Composition–rejection SSA (Slepoy, Thompson & Plimpton 2008) with per-node hazards in logarithmic buckets: O(k̄)
expected cost per event. Static networks only; no interventions.
"""
struct CompositionRejection <: OutbreakAlgorithm end

const _CR_MAX_BUCKETS = 64

# ---------------------------------------------------------------------------
# Bucket helpers
# ---------------------------------------------------------------------------

@inline function _cr_bucket_index(a::Float64, log_base::Float64)::Int
    return clamp(1 + floor(Int, log2(a / log_base)), 1, _CR_MAX_BUCKETS)
end

# Total per-node hazard = spontaneous + contact.
@inline function _cr_total_hazard(tally::Vector{Float64}, v::Integer, layers, weights,
                                  node_state::Vector{Int}, rm::_RunModel)
    return rm.spont_total[node_state[v]] + _contact_hazard!(tally, v, layers, weights, node_state, rm)
end

# Remove node v from its bucket (O(1) swap-and-pop). Returns the old hazard. A bucket that becomes empty has its
# sum reset to exactly 0, so floating-point residue can never make an empty bucket selectable.
function _cr_remove!(v::Int, node_hazard::Vector{Float64}, node_bucket::Vector{Int},
                     node_pos::Vector{Int}, bucket_members::Vector{Vector{Int}},
                     bucket_sum::Vector{Float64})
    b = node_bucket[v]
    b == 0 && return 0.0
    old_h = node_hazard[v]
    members = bucket_members[b]
    pos = node_pos[v]
    n_mem = length(members)
    if pos < n_mem
        last_v = members[n_mem]
        members[pos] = last_v
        node_pos[last_v] = pos
    end
    pop!(members)
    bucket_sum[b] = isempty(members) ? 0.0 : bucket_sum[b] - old_h
    node_bucket[v] = 0
    node_pos[v]   = 0
    return old_h
end

# Recompute node v's hazard, remove it from its old bucket and insert it into the new one.
function _cr_update_node!(v::Int, tally::Vector{Float64}, layers, weights, node_state::Vector{Int},
                          rm::_RunModel, node_hazard::Vector{Float64}, node_bucket::Vector{Int},
                          node_pos::Vector{Int}, bucket_members::Vector{Vector{Int}},
                          bucket_sum::Vector{Float64}, log_base::Float64)
    _cr_remove!(v, node_hazard, node_bucket, node_pos, bucket_members, bucket_sum)
    new_h = _cr_total_hazard(tally, v, layers, weights, node_state, rm)
    node_hazard[v] = new_h
    if new_h > 0.0
        b = _cr_bucket_index(new_h, log_base)
        push!(bucket_members[b], v)
        node_pos[v]    = length(bucket_members[b])
        node_bucket[v] = b
        bucket_sum[b] += new_h
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

function _simulate_impl(::CompositionRejection, spec::OutbreakSpec, rng::AbstractRNG, seed::_RecordedSeed,
                        keep::Symbol, plan::InterventionPlan = InterventionPlan())
    spec.network isa TimeVaryingNetwork &&
        throw(ArgumentError(
            "CompositionRejection does not yet support TimeVaryingNetwork. " *
            "Use DirectSSA, NextReaction or HAS for time-varying topologies."))
    spec.network isa MultiplexGraph &&
        throw(ArgumentError(
            "CompositionRejection does not yet support MultiplexGraph; use DirectSSA, NextReaction or HAS"))
    isempty(plan) ||
        throw(ArgumentError("CompositionRejection does not support interventions; use DirectSSA, NextReaction or HAS"))

    g, layers, weights, _ = _prepare_network(spec.network)
    return _cr_run(spec, rng, seed, keep, g, layers, weights)
end

# Function barrier: the loop is compiled for the concrete graph and layer types.
function _cr_run(spec::OutbreakSpec, rng::AbstractRNG, seed::_RecordedSeed, keep::Symbol, g, layers, weights)
    n = nv(g)
    rm = _RunModel(spec.model, spec.network)
    node_state = initial_state(spec, rng)
    state = _initial_outbreak_state(rm, node_state)
    tally = _tally_buffer(rm)

    # --- initial per-node hazards and bucket structure ---
    node_hazard = zeros(Float64, n)
    for v in 1:n
        node_hazard[v] = _cr_total_hazard(tally, v, layers, weights, node_state, rm)
    end
    active = filter(>(0.0), node_hazard)
    # log_base: lower edge of bucket 1; the minimum active hazard falls in bucket 1.
    log_base = isempty(active) ? 1.0 : minimum(active)

    bucket_members = [Int[] for _ in 1:_CR_MAX_BUCKETS]
    bucket_sum     = zeros(Float64, _CR_MAX_BUCKETS)
    node_bucket    = zeros(Int, n)
    node_pos       = zeros(Int, n)
    for v in 1:n
        h = node_hazard[v]
        h > 0.0 || continue
        b = _cr_bucket_index(h, log_base)
        push!(bucket_members[b], v)
        node_pos[v]    = length(bucket_members[b])
        node_bucket[v] = b
        bucket_sum[b] += h
    end

    t_now = spec.tspan[1]
    t_end = spec.tspan[2]
    rec = _Recorder(keep)
    _record!(rec, t_now, state.counts)

    while true
        total_rate = sum(bucket_sum)     # exact resync every event (no drift)
        total_rate > 0.0 || break        # nothing can be scheduled here, so zero rate is absorbing

        t_next = t_now + randexp(rng) / total_rate
        t_next <= t_end || break
        t_now = t_next

        # --- composition: bucket ∝ bucket_sum ---
        target = rand(rng) * total_rate
        b = 0
        cum = 0.0
        @inbounds for i in 1:_CR_MAX_BUCKETS
            cum += bucket_sum[i]
            if target < cum
                b = i
                break
            end
        end
        if b == 0  # floating-point overshoot: the last non-empty bucket
            for i in _CR_MAX_BUCKETS:-1:1
                if !isempty(bucket_members[i])
                    b = i
                    break
                end
            end
        end

        # --- rejection: uniform node in bucket b, accepted ∝ a_v ---
        # Exact when every member satisfies a_v ≤ log_base · 2^b. The index is clamped to the top bucket, whose
        # members may exceed its ceiling (hazards spanning more than 2^63), so that bucket is always sampled exactly.
        members = bucket_members[b]
        a_max_b = log_base * exp2(b)   # bucket ceiling log_base · 2^b
        fired_node = 0
        if b < _CR_MAX_BUCKETS
            @inbounds for _ in 1:256
                v = members[rand(rng, 1:length(members))]
                if rand(rng) * a_max_b <= node_hazard[v]
                    fired_node = v
                    break
                end
            end
        end
        if fired_node == 0          # exact sampling within the bucket (after 256 rejections, or the top bucket)
            target_in_bucket = rand(rng) * sum(node_hazard[v] for v in members)
            cum_in_bucket = 0.0
            for v in members
                cum_in_bucket += node_hazard[v]
                if target_in_bucket < cum_in_bucket
                    fired_node = v
                    break
                end
            end
            fired_node == 0 && (fired_node = members[end])
        end

        # --- which transition fires ---
        c = node_state[fired_node]
        fired_j = if rand(rng) * node_hazard[fired_node] < rm.spont_total[c]
            _sample_spontaneous_transition(rm, c, rng)
        else
            _sample_contact_transition!(tally, fired_node, layers, weights, node_state, rm, rng)
        end

        _fire!(state, rm, fired_node, fired_j)
        _record!(rec, t_now, state.counts)
        _log_event!(rec, t_now, fired_j, fired_node)

        # --- update the fired node and its neighbours ---
        _cr_update_node!(fired_node, tally, layers, weights, node_state, rm, node_hazard, node_bucket,
                         node_pos, bucket_members, bucket_sum, log_base)
        @inbounds for u in neighbors(g, fired_node)
            _cr_update_node!(u, tally, layers, weights, node_state, rm, node_hazard, node_bucket,
                             node_pos, bucket_members, bucket_sum, log_base)
        end
    end

    return _trajectory(rec, rm, state, t_end, seed, :CompositionRejection)
end
