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

Scheduled events (TimeVaryingNetwork updates and interventions):
  A graph update affects the hazards of its two endpoints only (2 × O(log N)
  tree updates); a state change affects the moved nodes and their neighbours;
  a rate change recomputes every leaf.  HAS does not store per-channel waiting
  times, so the Δt drawn before a scheduled event is discarded and the next
  randexp() draw uses the updated total rate Λ (exact by memorylessness).
  Network updates and interventions are merged into one time-ordered queue,
  and the run does not stop at Λ = 0 while any of them is pending.

Per-node hazard:
  `_cr_total_hazard` (composition_rejection.jl): the spontaneous rate plus the
  shared contact hazard of algorithms/common.jl.

See  docs/HAS_PLAN.md  for the full implementation roadmap.
See  https://github.com/KleistLab/HAS  for the reference Cython implementation.
Multiplex networks (`MultiplexGraph`, WP26):
  the leaf hazard uses the shared layer-weighted contact hazard (transition j
  acts through layer ℓ at the (layer × contact) rate
  rates[j]·[layer_j ∈ (:all, name_ℓ)]·layer_rates[ℓ], as in DirectSSA), and after an event the neighbours in every layer of
  positive weight are refreshed (a neighbour in several layers is recomputed
  once per layer, which is idempotent).

Graph processes (`DynamicGraph`, src/processes/, WP26):
  the total rate is Λ = tree[1] + λ_p, where λ_p is the process rate (re-read
  at every step). With probability λ_p/Λ the event is a process event: it
  changes the run's own copy of the graph, and the leaves of the (at most four,
  for neighbour exchange) nodes whose neighbourhoods changed are recomputed.
  Once no catalyst is left and no spontaneous transition or intervention can
  fire, the run ends (the graph can no longer change the epidemic). Without a
  process λ_p = 0 and no extra random number is drawn, so static runs are
  unchanged.
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

Supports static, time-varying (`TimeVaryingNetwork`) and multiplex (`MultiplexGraph`) networks, graph processes
(`DynamicGraph`, e.g. neighbour exchange) and interventions.

See `docs/HAS_PLAN.md` for design details.
"""
struct HAS <: OutbreakAlgorithm end

_supports_interventions(::HAS) = true

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
    # carry zero hazard and are unreachable (see the caller); return 0 rather
    # than biasing the final real node, and let the caller raise an error.
    v = k - N_tree + 1
    return v <= n ? v : 0
end

# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

# Recompute node v's hazard and propagate it up the tree.
@inline function _has_refresh!(tree::Vector{Float64}, node_hazard::Vector{Float64}, N_tree::Int,
                               tally::Vector{Float64}, v::Integer, layers, weights,
                               node_state::Vector{Int}, rm::_RunModel)
    h = _cr_total_hazard(tally, v, layers, weights, node_state, rm)
    node_hazard[v] = h
    _has_update!(tree, _has_leaf(Int(v), N_tree), h)
    return nothing
end

# Node v changed compartment: refresh it and its neighbours in every layer that carries weight (one layer for a
# single graph, in the graph's neighbour order).
function _has_refresh_node!(tree, node_hazard, N_tree, tally, v::Integer, layers, weights,
                            node_state::Vector{Int}, rm::_RunModel)
    _has_refresh!(tree, node_hazard, N_tree, tally, v, layers, weights, node_state, rm)
    for l in eachindex(layers)
        weights[l] > 0 || continue
        for u in neighbors(layers[l], v)
            _has_refresh!(tree, node_hazard, N_tree, tally, u, layers, weights, node_state, rm)
        end
    end
    return nothing
end

function _has_after_intervention!(tree, node_hazard, N_tree, tally, touched, layers, weights,
                                  node_state::Vector{Int}, rm::_RunModel, n::Int)
    if touched === nothing           # rates changed: recompute every leaf
        for v in 1:n
            _has_refresh!(tree, node_hazard, N_tree, tally, v, layers, weights, node_state, rm)
        end
    else
        for v in touched
            _has_refresh_node!(tree, node_hazard, N_tree, tally, v, layers, weights, node_state, rm)
        end
    end
    return nothing
end

function _simulate_impl(::HAS, spec::OutbreakSpec, rng::AbstractRNG, seed::_RecordedSeed, keep::Symbol,
                        plan::InterventionPlan = InterventionPlan())
    g, layers, weights, updates = _prepare_network(spec.network)
    return _has_run(spec, rng, seed, keep, plan, g, layers, weights, updates, nothing)
end

function _simulate_impl(::HAS, spec::OutbreakSpec{<:DynamicGraph}, rng::AbstractRNG, seed::_RecordedSeed, keep::Symbol,
                        plan::InterventionPlan = InterventionPlan())
    g, proc = _prepare_dynamic(spec.network, rng)
    return _has_run(spec, rng, seed, keep, plan, g, (g,), (1.0,), nothing, proc)
end

# Function barrier: the loop is compiled for the concrete graph, layer and process types (`proc === nothing` when
# the network has no graph process).
function _has_run(spec::OutbreakSpec, rng::AbstractRNG, seed::_RecordedSeed, keep::Symbol, plan::InterventionPlan,
                  g, layers, weights, updates, proc)
    n = nv(g)

    rm = _RunModel(spec.model, spec.network)
    node_state = initial_state(spec, rng)
    state = _initial_outbreak_state(rm, node_state)
    tally = _tally_buffer(rm)

    # --- binary sum tree: N_tree = smallest power of 2 ≥ n; padding leaves hold 0.0 ---
    N_tree      = nextpow(2, max(n, 1))
    tree        = zeros(Float64, 2 * N_tree)
    node_hazard = zeros(Float64, n)
    for v in 1:n
        h = _cr_total_hazard(tally, v, layers, weights, node_state, rm)
        node_hazard[v] = h
        tree[_has_leaf(v, N_tree)] = h
    end
    for k in (N_tree - 1):-1:1
        @inbounds tree[k] = tree[2k] + tree[2k + 1]
    end

    t_now = spec.tspan[1]
    t_end = spec.tspan[2]
    rec = _Recorder(keep)
    _record!(rec, t_now, state.counts)
    sched = _Schedule(updates, plan, t_now)
    on_change = touched ->
        _has_after_intervention!(tree, node_hazard, N_tree, tally, touched, layers, weights,
                                 node_state, rm, n)
    on_update = upd -> for v in (upd.src, upd.dst)
        _has_refresh!(tree, node_hazard, N_tree, tally, v, layers, weights, node_state, rm)
    end
    _check_thresholds!(sched, state, rm, n, rng, on_change) && _record!(rec, t_now, state.counts)

    while true
        λ_p = _proc_rate(proc)       # graph-process rate (0 without a process)
        Λ = tree[1] + λ_p
        t_next = Λ > 0.0 ? t_now + randexp(rng) / Λ : Inf

        # --- scheduled network updates and interventions come first when due (M4, M5) ---
        t_sched = _next_scheduled_time(sched)
        if t_sched <= t_next
            (isfinite(t_sched) && t_sched <= t_end) || break
            if !(Λ > 0.0) && !_interventions_pending(sched) && _no_catalysts(rm, state.counts)
                break     # absorbing: no hazard can ever become positive
            end
            t_now = t_sched
            if _apply_due!(sched, t_now, g, state, rm, n, rng, on_update, on_change)
                _record!(rec, t_now, state.counts)
                _check_thresholds!(sched, state, rm, n, rng, on_change) && _record!(rec, t_now, state.counts)
            end
            continue    # the Δt drawn under the old rates is discarded (memorylessness)
        end
        (isfinite(t_next) && t_next <= t_end) || break
        t_now = t_next

        # --- a graph-process event with probability λ_p/Λ (no draw without a process) ---
        if λ_p > 0.0 && rand(rng) * Λ < λ_p
            _epidemic_absorbing(rm, state.counts, sched) && break
            for v in _process_fire!(proc, g, node_state, rng)
                _has_refresh!(tree, node_hazard, N_tree, tally, v, layers, weights, node_state, rm)
            end
            continue
        end

        # --- firing node by tree descent, then the transition ---
        fired_node = _has_sample_node(tree, N_tree, n, rng)
        # Unreachable: parents are recomputed exactly as left + right, so a zero-sum subtree (padding) is never
        # entered while the root is positive. Fail loudly rather than silently ending the run.
        fired_node == 0 && error("HAS: sampled a zero-hazard leaf with total hazard $(tree[1]) (internal error)")
        c = node_state[fired_node]
        fired_j = if rand(rng) * node_hazard[fired_node] < rm.spont_total[c]
            _sample_spontaneous_transition(rm, c, rng)
        else
            _sample_contact_transition!(tally, fired_node, layers, weights, node_state, rm, rng)
        end

        _fire!(state, rm, fired_node, fired_j)
        _record!(rec, t_now, state.counts)
        _log_event!(rec, t_now, fired_j, fired_node)

        _has_refresh_node!(tree, node_hazard, N_tree, tally, fired_node, layers, weights, node_state, rm)
        _check_thresholds!(sched, state, rm, n, rng, on_change) && _record!(rec, t_now, state.counts)
    end

    return _trajectory(rec, rm, state, t_end, seed, :HAS)
end
