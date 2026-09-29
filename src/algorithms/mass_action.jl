# Owner: WP26 (DESIGN_NetworkEpiCore.md §G.2 WP26, §C.3, decision H25).
#=
algorithms/mass_action.jl

MassActionSSA: the exact count-level sampler for a `WellMixed(κ)` population of N nodes, and
`simulate(model, net::WellMixed; …)` on it (design §C.3, H25: "MassActionSSA by default; K_N only for the M13 lumping
test").

WellMixed(κ) means κ fleeting contacts per node per unit time with uniformly random partners, so a node in the
recipient compartment s of contact transition j is hit at rate κ τ_j × (fraction of the other nodes in via_j). On N
nodes this is the complete graph K_N with the per-edge rate κτ_j/(N − 1), and its node-level CTMC lumps exactly to the
counts (M13): every node of s has the same hazard

    h_j = rates[j] · κ/(N − 1) · Σ_{c ∈ via_j} (n_c − [c = s]),

(the other nodes: a node is not its own contact), so transition j fires at total rate n_s h_j, and the node that
moves is uniform in s. The sampler is Gillespie's direct method on the transitions (O(T + C) per event, independent
of N). It also tracks which node moves (a uniform member of s, by a swap-and-pop membership list), so the trajectory
has per-node infection counts and event logs exactly as on K_N, and node-level interventions work.

The per-contact rate τ of the model is converted with the rate convention on WellMixed(κ) (NetworkEpiCore
`per_contact_rates`): mass action with β_j = κτ_j (design §C.2).
=#

export MassActionSSA

"""
    MassActionSSA <: OutbreakAlgorithm

The exact count-level sampler of a well-mixed population (NetworkEpiCore `WellMixed(κ)`, design §C.3): Gillespie's
direct method on the compartment counts, O(T + C) per event whatever the population size N (T transitions, C
compartments).

A contact `s + J → X + J` with per-contact rate τ fires at total rate `τ κ n_s n_J / (N − 1)` (with `n_J − 1` when
`J = s`): the lumping of the process on the complete graph K_N with per-edge rate κτ/(N − 1), which it equals in
distribution for every N (Kurtz; design §D.5 M13). As N → ∞ the fractions follow the mass-action ODE with β = κτ.
The moving node is drawn uniformly from its compartment, so per-node infection counts (`final_size`,
`reinfection_histogram`), event logs (`keep = :events`) and interventions work as in the network samplers.

The spec's network must be `SampledNetwork(WellMixed(κ), N)`, which is what
`simulate(model, WellMixed(κ); N, …)` uses (by default with this sampler).

```julia
ens = simulate(sir_model(), WellMixed(5); N = 10_000, p = Dict(:τ => 1/10, :γ => 1/4),
               initial = SeedFraction(:I => 0.01), tspan = (0.0, 60.0), nsims = 100, seed = 1)
```
"""
struct MassActionSSA <: OutbreakAlgorithm end

_supports_interventions(::MassActionSSA) = true

function _simulate_impl(::MassActionSSA, spec::OutbreakSpec, rng::AbstractRNG, seed::_RecordedSeed, keep::Symbol,
                        plan::InterventionPlan = InterventionPlan())
    net = spec.network
    (net isa SampledNetwork && net.descriptor isa WellMixed) || throw(ArgumentError(
        "MassActionSSA simulates a well-mixed population: the network of the spec must be " *
        "SampledNetwork(WellMixed(κ), N) (use simulate(model, WellMixed(κ); N, …)); got a $(nameof(typeof(net))). " *
        "For a contact graph use NextReaction, HAS, DirectSSA or CompositionRejection"))
    return _ma_run(spec, rng, seed, keep, plan, net.N, _numeric_κ(net.descriptor))
end

# κ of a WellMixed descriptor as a number (it may be symbolic in NetworkEpiCore; WellMixed checks κ > 0 when numeric).
function _numeric_κ(net::WellMixed)
    κ = net.κ
    κ isa Union{AbstractFloat, Integer, Rational} ||
        throw(ArgumentError("WellMixed(κ) needs a numeric κ to be simulated; got $(κ)"))
    return Float64(κ)
end

# Nodes of each compartment, for a uniform draw in O(1) and O(1) moves (swap-and-pop).
struct _Membership
    members::Vector{Vector{Int}}   # nodes in each compartment, in no particular order
    pos::Vector{Int}               # position of each node in its compartment's list
    comp::Vector{Int}              # the compartment each node is listed under
end

function _Membership(node_state::Vector{Int}, C::Int)
    members = [Int[] for _ in 1:C]
    pos = zeros(Int, length(node_state))
    for (v, c) in pairs(node_state)
        push!(members[c], v)
        pos[v] = length(members[c])
    end
    return _Membership(members, pos, copy(node_state))
end

function _move_member!(m::_Membership, v::Int, new::Int)
    old = m.comp[v]
    old == new && return m
    list = m.members[old]
    p = m.pos[v]
    last_v = list[end]
    list[p] = last_v
    m.pos[last_v] = p
    pop!(list)
    push!(m.members[new], v)
    m.pos[v] = length(m.members[new])
    m.comp[v] = new
    return m
end

_random_member(m::_Membership, c::Int, rng::AbstractRNG) = m.members[c][rand(rng, 1:length(m.members[c]))]

# After an intervention: relist the moved nodes (rate changes move none).
_sync_members!(m::_Membership, node_state::Vector{Int}, ::Nothing) = m
function _sync_members!(m::_Membership, node_state::Vector{Int}, touched)
    for v in touched
        _move_member!(m, Int(v), node_state[v])
    end
    return m
end

function _ma_run(spec::OutbreakSpec, rng::AbstractRNG, seed::_RecordedSeed, keep::Symbol, plan::InterventionPlan,
                 n::Int, κ::Float64)
    rm = _RunModel(spec.model, spec.network)
    C = rm.C
    node_state = initial_state(spec, rng)
    state = _initial_outbreak_state(rm, node_state)
    pool = _Membership(node_state, C)

    # Contact transitions: recipient compartment and catalyst compartments. On one node there are no contacts.
    scale = n > 1 ? κ / (n - 1) : 0.0
    contacts = findall(rm.is_contact)
    recipient = Int[rm.from_idx[j] for j in contacts]
    catalysts = [findall(rm.via_mask[j]) for j in contacts]
    hazard = zeros(Float64, length(contacts))

    t_now = spec.tspan[1]
    t_end = spec.tspan[2]
    rec = _Recorder(keep)
    _record!(rec, t_now, state.counts)
    sched = _Schedule(nothing, plan, t_now)
    noop(_) = nothing
    on_change = touched -> _sync_members!(pool, node_state, touched)
    _check_thresholds!(sched, state, rm, n, rng, on_change) && _record!(rec, t_now, state.counts)

    counts = state.counts
    while true
        # --- propensities from the counts ---
        contact_rate = 0.0
        @inbounds for k in eachindex(contacts)
            s = recipient[k]
            h = 0.0
            if counts[s] > 0
                others = 0
                for c in catalysts[k]
                    others += counts[c] - (c == s)
                end
                h = rm.rates[contacts[k]] * scale * counts[s] * others
            end
            hazard[k] = h
            contact_rate += h
        end
        spontaneous_rate = 0.0
        @inbounds for c in 1:C
            spontaneous_rate += counts[c] * rm.spont_total[c]
        end
        total_rate = contact_rate + spontaneous_rate

        # --- next event time, merged with the scheduled interventions (M4, M5) ---
        t_next = total_rate > 0 ? t_now + randexp(rng) / total_rate : Inf
        t_sched = _next_scheduled_time(sched)
        if t_sched <= t_next
            (isfinite(t_sched) && t_sched <= t_end) || break
            if total_rate == 0 && !_interventions_pending(sched) && _no_catalysts(rm, counts)
                break
            end
            t_now = t_sched
            if _apply_due!(sched, t_now, nothing, state, rm, n, rng, noop, on_change)
                _record!(rec, t_now, counts)
                _check_thresholds!(sched, state, rm, n, rng, on_change) && _record!(rec, t_now, counts)
            end
            continue    # the rates are constant between scheduled times, so redrawing from t_now is exact
        end
        (isfinite(t_next) && t_next <= t_end) || break
        t_now = t_next

        # --- choose the transition, then a uniform node of its source compartment ---
        u = rand(rng) * total_rate
        local fired_j::Int
        local fired_node::Int
        if u < contact_rate
            k = _pick_weighted(hazard, u)
            fired_j = contacts[k]
            fired_node = _random_member(pool, recipient[k], rng)
        else
            picked_c = 0
            cum = contact_rate
            last_c = 0
            @inbounds for c in 1:C
                w = counts[c] * rm.spont_total[c]
                w > 0 || continue
                last_c = c
                cum += w
                if u < cum
                    picked_c = c
                    break
                end
            end
            picked_c == 0 && (picked_c = last_c)       # floating-point guard
            fired_node = _random_member(pool, picked_c, rng)
            fired_j = _sample_spontaneous_transition(rm, picked_c, rng)
        end

        _fire!(state, rm, fired_node, fired_j)
        _move_member!(pool, fired_node, rm.to_idx[fired_j])
        _record!(rec, t_now, counts)
        _log_event!(rec, t_now, fired_j, fired_node)
        _check_thresholds!(sched, state, rm, n, rng, on_change) && _record!(rec, t_now, counts)
    end

    return _trajectory(rec, rm, state, t_end, seed, :MassActionSSA)
end

# The index k with cumulative weight first exceeding `u` (u < Σ w); the last positive weight guards rounding.
function _pick_weighted(w::Vector{Float64}, u::Float64)
    cum = 0.0
    last_k = 0
    @inbounds for k in eachindex(w)
        w[k] > 0 || continue
        last_k = k
        cum += w[k]
        u < cum && return k
    end
    return last_k
end

# ---------------------------------------------------------------------------------------------------------------
# simulate(model, net::WellMixed; …)
# ---------------------------------------------------------------------------------------------------------------

"""
    simulate(model, net::WellMixed; N, p = Dict(), initial, tspan, nsims = 1, algorithm = MassActionSSA(),
             seed = nothing, rng = nothing, tgrid = nothing, keep = :grid, interventions = InterventionPlan(),
             parallel = false, graphs = :per_run) -> OutbreakEnsemble

Simulate `nsims` runs of `model` in a well-mixed population of `N` nodes (NetworkEpiCore `WellMixed(κ)`: a node in
the recipient compartment of a contact with per-contact rate τ is infected at rate κτ × the fraction of the other
nodes in the catalyst compartment; mass action with β = κτ).

- `algorithm = MassActionSSA()` (the default, design H25): the exact count-level sampler, O(1) in N per event. The
  ensemble's `spec.network` is `SampledNetwork(net, N)`.
- A graph sampler (`NextReaction()`, `HAS()`, `DirectSSA()`): the same process on the complete graph K_N with the
  per-edge rates κτ/(N − 1), whose counts lump exactly to the `MassActionSSA` process (design §D.5 M13). The graph
  is [`sample_graph`](@ref)`(net, N)`, a one-layer `MultiplexGraph` of K_N with layer rate κ/(N − 1), so the model
  keeps its per-contact τ. It costs O(N) per event and exists for that lumping test; the ensemble's `spec` holds
  that graph.

Run r uses the stream `NetworkOutbreaks.stable_rng(b + 2^32 + r)` for the base seed `b` (design §J.7), so
`traj.seed` reproduces it with `simulate(spec; seed = traj.seed, algorithm)`. `graphs` is accepted for signature
compatibility with the other descriptors and ignored (a well-mixed population has no graph to redraw). The other
keywords are those of `simulate(model, net::NetworkDescriptor; …)`.
"""
function simulate(model, net::WellMixed;
                  N::Integer,
                  p::AbstractDict = Dict{Symbol, Float64}(),
                  initial::SeedSpec,
                  tspan::Tuple{<:Real, <:Real},
                  nsims::Integer = 1,
                  graphs = :per_run,
                  algorithm::OutbreakAlgorithm = MassActionSSA(),
                  seed::Union{Nothing, Integer} = nothing,
                  rng::Union{Nothing, AbstractRNG} = nothing,
                  tgrid = nothing,
                  keep::Symbol = :grid,
                  interventions::InterventionPlan = InterventionPlan(),
                  parallel::Bool = false)
    N >= 1 || throw(ArgumentError("simulate: N must be ≥ 1; got $(N)"))
    nsims >= 1 || throw(ArgumentError("simulate: nsims must be ≥ 1; got $(nsims)"))
    _check_graphs_mode(graphs; allow_fixed = true)
    ts = (Float64(tspan[1]), Float64(tspan[2]))
    om = _outbreak_model(model, p, net)
    grid = _grid(tgrid, ts, keep)
    base = _base_seed(seed, rng)
    network = algorithm isa MassActionSSA ? SampledNetwork(net, N) : first(sample_graph(net, Int(N)))
    trajs = _run_all(r -> _one_run(om, network, initial, ts, algorithm, _run_seed(base, r), keep, grid,
                                   interventions),
                     Int(nsims), parallel)
    return OutbreakEnsemble(OutbreakSpec(om, network, initial, ts), trajs)
end
