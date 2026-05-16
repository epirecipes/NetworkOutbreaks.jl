#=
algorithms/composition_rejection.jl

Composition–Rejection SSA (Slepoy, Thompson & Plimpton 2008,
"A constant-time kinetic Monte Carlo algorithm for simulation of large
biochemical reaction networks", J. Chem. Phys. 128, 205101).

Each node carries a single aggregate hazard  a_v = spont_rate + infect_rate.
Nodes are binned into logarithmic buckets:
    bucket b  covers  [log_base · 2^(b-1),  log_base · 2^b)

Total rate  Λ = Σ_b  S_b  where  S_b  is the bucket sum.

Event selection:
  1. Composition: pick bucket b  ∝ S_b  (linear scan over ≤ 64 buckets).
  2. Rejection:   pick node uniformly from bucket b; accept with probability
                  a_v / (log_base · 2^b).  Expected acceptance ≥ 0.5.
  3. Time:        Δt = Exp(1) / Λ.

Bucket membership is maintained with O(1) swap-and-pop removal so the
per-event cost is O(Δ · log(a_max/a_min)) where Δ is the maximum node degree.
For fixed-degree graphs this is effectively constant.

NOTE: TimeVaryingNetwork is not yet supported; combine with DirectSSA or
NextReaction for time-varying topologies.
=#

struct CompositionRejection <: OutbreakAlgorithm end

const _CR_MAX_BUCKETS = 64

# ---------------------------------------------------------------------------
# Bucket helpers
# ---------------------------------------------------------------------------

@inline function _cr_bucket_index(a::Float64, log_base::Float64)::Int
    return clamp(1 + floor(Int, log2(a / log_base)), 1, _CR_MAX_BUCKETS)
end

# Total per-node hazard = spontaneous + infection
@inline function _cr_total_hazard(v::Integer, g::AbstractGraph, model::OutbreakModel,
                                  node_state::Vector{Int},
                                  spont_total_rate::Vector{Float64},
                                  infection_by_src::Vector{Vector{OutbreakTransition}},
                                  via_mask::Dict{OutbreakTransition, BitVector})
    src_idx = node_state[v]
    return spont_total_rate[src_idx] +
           _infection_hazard(v, g, model, node_state, infection_by_src[src_idx], via_mask)
end

# Remove node v from its bucket (O(1) swap-and-pop).  Returns old hazard.
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
    bucket_sum[b] -= old_h
    node_bucket[v] = 0
    node_pos[v]   = 0
    return old_h
end

# Recompute node v's hazard, remove from old bucket, insert into new bucket.
# Returns the change in total_rate (new_h − old_h).
function _cr_update_node!(v::Int, g::AbstractGraph, model::OutbreakModel,
                          node_state::Vector{Int},
                          spont_total_rate::Vector{Float64},
                          infection_by_src::Vector{Vector{OutbreakTransition}},
                          via_mask::Dict{OutbreakTransition, BitVector},
                          node_hazard::Vector{Float64},
                          node_bucket::Vector{Int},
                          node_pos::Vector{Int},
                          bucket_members::Vector{Vector{Int}},
                          bucket_sum::Vector{Float64},
                          log_base::Float64)
    old_h = _cr_remove!(v, node_hazard, node_bucket, node_pos,
                        bucket_members, bucket_sum)
    new_h = _cr_total_hazard(v, g, model, node_state, spont_total_rate,
                             infection_by_src, via_mask)
    node_hazard[v] = new_h
    if new_h > 0.0
        b = _cr_bucket_index(new_h, log_base)
        push!(bucket_members[b], v)
        node_pos[v]    = length(bucket_members[b])
        node_bucket[v] = b
        bucket_sum[b] += new_h
    end
    return new_h - old_h
end

# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

function _simulate_impl(::CompositionRejection, spec::OutbreakSpec,
                        seed::UInt64, keep::Symbol,
                        interventions::InterventionPlan = InterventionPlan())
    spec.network isa TimeVaryingNetwork &&
        throw(ArgumentError(
            "CompositionRejection does not yet support TimeVaryingNetwork. " *
            "Use DirectSSA or NextReaction for time-varying topologies."))
    spec.network isa MultiplexNetwork &&
        throw(ArgumentError(
            "CompositionRejection does not yet support MultiplexNetwork; use DirectSSA"))

    rng   = Xoshiro(seed)
    model = spec.model
    g     = _outbreak_graph(spec.network)
    n     = nv(g)
    C     = ncompartments(model)

    node_state = initial_state(spec, rng)
    state      = OutbreakState(model, node_state)

    # Pre-bucket transitions by source compartment and type.
    spont_by_src     = [OutbreakTransition[] for _ in 1:C]
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

    # --- Compute initial per-node hazards and initialise bucket structure. ---
    node_hazard = zeros(Float64, n)
    for v in 1:n
        node_hazard[v] = _cr_total_hazard(v, g, model, node_state, spont_total_rate,
                                          infection_by_src, via_mask)
    end

    active = filter(>(0.0), node_hazard)
    # log_base: lower edge of bucket 1; the minimum active hazard falls in bucket 1.
    log_base = isempty(active) ? 1.0 : minimum(active)

    bucket_members = [Int[] for _ in 1:_CR_MAX_BUCKETS]
    bucket_sum     = zeros(Float64, _CR_MAX_BUCKETS)
    node_bucket    = zeros(Int, n)
    node_pos       = zeros(Int, n)
    total_rate     = 0.0

    for v in 1:n
        h = node_hazard[v]
        h > 0.0 || continue
        b = _cr_bucket_index(h, log_base)
        push!(bucket_members[b], v)
        node_pos[v]    = length(bucket_members[b])
        node_bucket[v] = b
        bucket_sum[b] += h
        total_rate    += h
    end

    t_now = spec.tspan[1]
    t_end = spec.tspan[2]

    times_buf  = Float64[t_now]
    counts_buf = Vector{Vector{Int}}()
    push!(counts_buf, copy(state.counts))
    events_buf = OutbreakEvent[]

    while t_now < t_end
        total_rate <= 0.0 && break

        # --- time advance ---
        t_now += randexp(rng) / total_rate
        t_now > t_end && break

        # --- Composition: pick bucket proportional to bucket_sum ---
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
        if b == 0  # floating-point overshoot fallback
            for i in _CR_MAX_BUCKETS:-1:1
                if bucket_sum[i] > 0.0
                    b = i; break
                end
            end
        end
        b == 0 && break  # absorbing state

        # --- Rejection: pick node uniformly from bucket b, accept ∝ a_v ---
        a_max_b = log_base * exp2(b)   # = log_base · 2^b  (bucket ceiling)
        fired_node = 0
        @inbounds for _ in 1:256
            members = bucket_members[b]
            isempty(members) && break
            v = members[rand(rng, 1:length(members))]
            rand(rng) * a_max_b <= node_hazard[v] && (fired_node = v; break)
        end
        if fired_node == 0
            members = bucket_members[b]
            if isempty(members) || bucket_sum[b] <= 0.0
                total_rate = sum(bucket_sum)
                continue
            end
            target_in_bucket = rand(rng) * bucket_sum[b]
            cum_in_bucket = 0.0
            for v in members
                cum_in_bucket += node_hazard[v]
                if target_in_bucket <= cum_in_bucket
                    fired_node = v
                    break
                end
            end
            fired_node == 0 && (fired_node = members[end])
        end

        # --- Sample which transition fires ---
        src_idx = node_state[fired_node]
        spont_h = spont_total_rate[src_idx]
        fired_tr = if rand(rng) * node_hazard[fired_node] < spont_h
            _sample_spontaneous_transition(spont_by_src[src_idx], rng)
        else
            _sample_infection_transition(fired_node, g, model, node_state,
                                         infection_by_src[src_idx], via_mask, rng)
        end

        # --- Apply event ---
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
                  OutbreakEvent(t_now, _transition_index(model, fired_tr), fired_node))
        end

        # --- Update buckets: fired node + its neighbours ---
        Δ = _cr_update_node!(fired_node, g, model, node_state, spont_total_rate,
                             infection_by_src, via_mask, node_hazard, node_bucket,
                             node_pos, bucket_members, bucket_sum, log_base)
        total_rate += Δ
        @inbounds for u in neighbors(g, fired_node)
            Δ = _cr_update_node!(u, g, model, node_state, spont_total_rate,
                                 infection_by_src, via_mask, node_hazard, node_bucket,
                                 node_pos, bucket_members, bucket_sum, log_base)
            total_rate += Δ
        end
        # Periodic full resync to prevent floating-point drift
        total_rate = sum(bucket_sum)
    end

    if times_buf[end] < t_end
        push!(times_buf, t_end)
        push!(counts_buf, copy(state.counts))
    end

    counts_mat = Matrix{Int}(undef, C, length(times_buf))
    @inbounds for k in eachindex(times_buf)
        counts_mat[:, k] = counts_buf[k]
    end

    return OutbreakTrajectory(model, times_buf, counts_mat,
                              copy(state.infection_counts),
                              events_buf, seed, :CompositionRejection)
end
