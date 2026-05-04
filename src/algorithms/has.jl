#=
algorithms/has.jl

Hierarchical Adaptive Sampling (HAS) — full implementation.

=== Algorithm summary ===

HAS was introduced by the Kleist Lab (https://github.com/KleistLab/HAS) as a
sub-quadratic exact SSA for large heterogeneous networks.  The key idea is to
replace the flat rate list of the Direct method with a complete binary tree of
partial-rate sums:

  • Leaf i stores the total hazard  a_i = spont_i + infect_i  for node i.
  • Each internal node stores the sum of hazards in its subtree.
  • The root stores  Λ = Σ a_i.

Data layout (1-indexed, length = 2 * N_tree where N_tree = nextpow(2, n)):
  tree[1]           = root = total rate Λ
  tree[k]           = sum of leaf hazards under subtree k
  tree[N_tree + v - 1] = leaf for node v  (inactive leaves padded with 0.0)
  parent(k) = k >> 1;  children: 2k (left), 2k+1 (right)

Event selection (O(log N)):
  Starting from the root, at each level draw a uniform [0,1) and compare it
  to the left-child fraction to decide whether to descend left or right.
  Reaching a leaf identifies the firing node.

Per-event update (O(Δ log N)):
  When node v fires and changes compartment, recompute a_v and propagate the
  difference up the tree (O(log N)).  Also update each of v's Δ neighbours
  (O(Δ log N) total).  For fixed-degree graphs this is O(log N) per event.

Comparison with CompositionRejection:
  Both achieve sub-linear update cost; HAS gives O(log N) deterministically
  regardless of rate heterogeneity, while Composition–Rejection is O(1)
  amortised when rates are well-clustered.  HAS is preferred for large N
  (≥ 10^4) or highly heterogeneous degree distributions (power-law networks).

Integration with TimeVaryingNetwork:
  A graph update (add/remove edge) affects hazards for the two endpoint nodes
  only → 2 × O(log N) tree updates per graph-update event.  HAS does not
  store per-channel waiting times, so edge updates are handled naturally: the
  Δt drawn before the update is discarded, and the next randexp() draw uses
  the freshly updated total rate Λ.

Per-node hazard:
  Reuses _cr_total_hazard from composition_rejection.jl (included before
  has.jl in NetworkOutbreaks.jl) — option (a): direct call, no refactoring.

See  docs/HAS_PLAN.md  for the full implementation roadmap.
See  https://github.com/KleistLab/HAS  for the reference Cython implementation.
=#

"""
    HAS <: OutbreakAlgorithm

Hierarchical Adaptive Sampling algorithm.

Maintains per-node hazards in a complete binary sum tree of length
`2 * N_tree` (where `N_tree = nextpow(2, N)`).  Event selection is
O(log N) via a single tree descent; per-event updates touch O(Δ) nodes
each at O(log N) cost, giving O(Δ log N) total.  For fixed-degree graphs
(regular, Erdős–Rényi) this is O(log N) per event regardless of rate
heterogeneity — a deterministic guarantee that `CompositionRejection`
provides only in the amortised sense.

Supports `TimeVaryingNetwork`.

See `docs/HAS_PLAN.md` for design details.
"""
struct HAS <: OutbreakAlgorithm end

# ---------------------------------------------------------------------------
# Binary-tree helpers
# ---------------------------------------------------------------------------

"""    _has_leaf(v, N_tree) → Int
Index of the leaf cell for node `v` (1-indexed) in the sum tree.
"""
@inline _has_leaf(v::Int, N_tree::Int)::Int = N_tree + v - 1

"""    _has_update!(tree, leaf_idx, new_h)
Set `tree[leaf_idx] = new_h` and propagate updated partial sums up to the
root.  O(log N).
"""
function _has_update!(tree::Vector{Float64}, leaf_idx::Int, new_h::Float64)
    tree[leaf_idx] = new_h
    k = leaf_idx >> 1
    while k >= 1
        @inbounds tree[k] = tree[2k] + tree[2k + 1]
        k >>= 1
    end
    return nothing
end

"""    _has_sample_node(tree, N_tree, n, rng) → Int
Descend the sum tree from the root using uniform draws to select a node
proportional to its hazard.  Returns the 1-indexed node number, or 0 if
the total rate is zero (absorbing state).  O(log N).
"""
function _has_sample_node(tree::Vector{Float64}, N_tree::Int, n::Int,
                          rng::AbstractRNG)::Int
    @inbounds tree[1] <= 0.0 && return 0
    k = 1
    @inbounds while k < N_tree
        left  = 2k
        right = 2k + 1
        # Descend left if uniform draw lands below the left-child fraction.
        if rand(rng) * tree[k] < tree[left]
            k = left
        else
            k = right
        end
    end
    # Convert leaf tree-index back to 1-indexed node number. Padding leaves
    # carry zero hazard and should be unreachable; return 0 defensively rather
    # than biasing the final real node if floating-point drift ever reaches one.
    v = k - N_tree + 1
    return v <= n ? v : 0
end

# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

function _simulate_impl(::HAS, spec::OutbreakSpec, seed::UInt64, keep::Symbol)
    rng     = Xoshiro(seed)
    model   = spec.model
    network = spec.network

    # Time-varying network support (mirrors next_reaction.jl / direct.jl).
    is_tvn          = network isa TimeVaryingNetwork
    g               = is_tvn ? deepcopy(network.graph) : _outbreak_graph(network)
    updates         = is_tvn ? network.updates : nothing
    next_update_idx = 1

    n = nv(g)
    C = ncompartments(model)

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

    # --- Build binary sum tree (O(n log n)) ---
    # N_tree = smallest power of 2 ≥ n.  Inactive leaf slots hold 0.0.
    N_tree      = nextpow(2, max(n, 1))
    tree        = zeros(Float64, 2 * N_tree)
    node_hazard = zeros(Float64, n)

    # Populate leaf level.
    for v in 1:n
        h = _cr_total_hazard(v, g, model, node_state, spont_total_rate,
                             infection_by_src, via_mask)
        node_hazard[v]           = h
        tree[_has_leaf(v, N_tree)] = h
    end
    # Fold partial sums up from leaves to root.
    for k in (N_tree - 1):-1:1
        @inbounds tree[k] = tree[2k] + tree[2k + 1]
    end

    t_now = spec.tspan[1]
    t_end = spec.tspan[2]

    times_buf  = Float64[t_now]
    counts_buf = Vector{Vector{Int}}()
    push!(counts_buf, copy(state.counts))
    events_buf = OutbreakEvent[]

    while t_now < t_end
        Λ = tree[1]
        Λ <= 0.0 && break  # absorbing state

        Δt     = randexp(rng) / Λ
        t_next = t_now + Δt

        # --- Time-varying network: apply updates that fire before t_next ---
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
                    # Only the two endpoints' infection hazards are affected.
                    for v in (upd.src, upd.dst)
                        h = _cr_total_hazard(v, g, model, node_state,
                                             spont_total_rate,
                                             infection_by_src, via_mask)
                        node_hazard[v] = h
                        _has_update!(tree, _has_leaf(v, N_tree), h)
                    end
                    next_update_idx += 1
                end
                # Discard Δt drawn under the old graph; redraw next iteration.
                continue
            end
        end

        t_next > t_end && break
        t_now = t_next

        # --- Sample firing node via tree descent ---
        fired_node = _has_sample_node(tree, N_tree, n, rng)
        fired_node == 0 && break  # defensive: absorbing state

        # --- Sample which transition fires (spontaneous vs infection) ---
        src_idx = node_state[fired_node]
        spont_h = spont_total_rate[src_idx]
        fired_tr = if rand(rng) * node_hazard[fired_node] < spont_h
            _sample_spontaneous_transition(spont_by_src[src_idx], rng)
        else
            _sample_infection_transition(fired_node, g, model, node_state,
                                         infection_by_src[src_idx], via_mask, rng)
        end

        # --- Apply state change ---
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

        # --- Update tree: fired node + its neighbours ---
        h = _cr_total_hazard(fired_node, g, model, node_state, spont_total_rate,
                             infection_by_src, via_mask)
        node_hazard[fired_node] = h
        _has_update!(tree, _has_leaf(fired_node, N_tree), h)

        for u in neighbors(g, fired_node)
            h = _cr_total_hazard(u, g, model, node_state, spont_total_rate,
                                 infection_by_src, via_mask)
            node_hazard[u] = h
            _has_update!(tree, _has_leaf(u, N_tree), h)
        end
    end

    # Final snapshot at t_end (same convention as other algorithms).
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
                              events_buf, seed, :HAS)
end
