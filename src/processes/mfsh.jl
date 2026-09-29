# Owner: WP36c (DESIGN_NetworkEpiCore.md §C.3, §D.5 Λ3, §J.7, §K WP36c).
#=
processes/mfsh.jl

The fleeting-contact process of mean-field social heterogeneity, the stochastic model behind NetworkEpiCore's
`MFSHNetwork(d)` (Miller–Slim–Volz Part II §3.2.2, actual-degree formulation):

- `FleetingContacts(k)`: the contact structure of one run: node v has k[v] stubs and no persistent edges;
- `sample_graph(net::MFSHNetwork, N; rng)`: N iid stub counts from the degree distribution (design §J.7 graph stream);
- `FleetingContactSSA`: the exact sampler, Gillespie's direct method on compartment × degree-class counts;
- `simulate(model, net::MFSHNetwork; N, …)`: a fresh stub sequence per run, run with `FleetingContactSSA`.

The process. Every stub of node v is, at every instant, joined to a stub drawn uniformly from the M − k_v stubs of the
other nodes (M = Σ_u k_u), so contacts are fleeting: the partner of a stub at one time says nothing about its partner
at another. A contact transition j (`s + J → X + J`, per-contact rate τ_j, catalysts `via_j`) therefore fires at a node
v in s at rate

    h_j(v) = τ_j · k_v · Σ_{c ∈ via_j} (K_c − [c = s]·k_v) / (M − k_v),

where K_c is the number of stubs of the nodes in c (v's own stubs are not its partners). The hazard depends on v only
through its stub count, so nodes of equal degree in a compartment are exchangeable: the sampler keeps the counts
n[c, i] of nodes in compartment c with the i-th distinct degree d_i, and draws the transition from the totals
H_j = τ_j Σ_i n[s, i]·d_i(K_via − [s ∈ via]·d_i)/(M − d_i), the degree class of the node with the residual of the
same uniform number (so no extra random number is used), and the node uniformly within its (compartment, class) list.
Spontaneous transitions act on a uniform node of their compartment, as in every other sampler. Each event costs
O(T·D + C) for T transitions, D distinct degrees and C compartments, whatever N.

As N → ∞ the fractions follow the MFSH edge-based ODE (EdgeBasedModels `edge_based(cm, MFSHNetwork(d))`): a stub meets
a partner in J with probability π_J, the fraction of all stubs in J. On a κ-regular degree sequence every node has
h_j = τ_j κ Σ_{c ∈ via}(n_c − [c = s])/(N − 1), the hazard of `MassActionSSA` on `WellMixed(κ)`: the two processes are
the same Markov chain (design: `MFSHNetwork(RegularDegree(κ))` is `WellMixed(κ)`), and since this sampler then draws
the same random numbers in the same order, it reproduces `MassActionSSA` event by event from the same seed.
=#

export FleetingContacts, FleetingContactSSA

# ---------------------------------------------------------------------------------------------------------------
# The contact structure of a run
# ---------------------------------------------------------------------------------------------------------------

"""
    FleetingContacts(degrees::AbstractVector{<:Integer})

The contact structure of one run of mean-field social heterogeneity (NetworkEpiCore `MFSHNetwork`): node v has
`degrees[v] ≥ 0` stubs, and at every instant each of them is joined to a stub drawn uniformly from the stubs of the
other nodes, so there are no persistent edges (fleeting contacts with heterogeneous activity). A node in the recipient
compartment s of a contact with per-contact rate τ and catalysts `via` is hit at rate
τ·k_v·Σ_{c ∈ via}(K_c − [c = s]·k_v)/(M − k_v), where K_c counts the stubs in compartment c and M all stubs.

It is the `network` of an [`OutbreakSpec`](@ref) run by [`FleetingContactSSA`](@ref);
[`sample_graph`](@ref)`(net::MFSHNetwork, N; rng)` draws one. It has no contact graph: `nv` is the number of nodes,
`ne` is 0 (no persistent edges, as `GraphInfo.edges`), and the graph samplers (`DirectSSA`, `NextReaction`,
`CompositionRejection`, `HAS`) refuse it with an `ArgumentError`. The vector is copied.

```julia
spec = OutbreakSpec(OutbreakModel(sir_model(), Dict(:τ => 1/12, :γ => 1/4)), FleetingContacts(rand(1:9, 1000)),
                    SeedFraction(:I => 0.01), (0.0, 100.0))
traj = simulate(spec; algorithm = FleetingContactSSA(), seed = 1)
```
"""
struct FleetingContacts <: AbstractContactNetwork
    degrees::Vector{Int}
    function FleetingContacts(degrees::AbstractVector{<:Integer})
        isempty(degrees) && throw(ArgumentError("FleetingContacts: the population needs at least one node"))
        all(>=(0), degrees) || throw(ArgumentError(
            "FleetingContacts: stub counts must be ≥ 0; got $(minimum(degrees))"))
        return new(collect(Int, degrees))
    end
end

Graphs.nv(net::FleetingContacts) = length(net.degrees)
Graphs.ne(::FleetingContacts) = 0             # fleeting contacts: no persistent edges

# The graph samplers read the graph of a network through `_outbreak_graph`; a population of fleeting contacts has none.
_outbreak_graph(net::FleetingContacts) = throw(ArgumentError(
    "FleetingContacts (mean-field social heterogeneity, $(length(net.degrees)) nodes) has no contact graph: its " *
    "contacts are fleeting. Simulate it with FleetingContactSSA (the default of simulate(model, MFSHNetwork(d); " *
    "N, …))"))

function Base.show(io::IO, net::FleetingContacts)
    n = length(net.degrees)
    M = sum(net.degrees)
    print(io, "FleetingContacts(", n, " nodes, ", M, " stubs, mean degree ", _gen_fmt4(M / n), ")")
end

"""
    sample_graph(net::MFSHNetwork, N::Integer; rng = Random.default_rng()) -> (FleetingContacts, info::GraphInfo)

The contact structure of one run on the mean-field social heterogeneity descriptor `net`: N stub counts drawn iid
from `net.degrees` with `rng`, as a [`FleetingContacts`](@ref) population (fleeting contacts have no persistent
graph, so no stubs are paired and none are erased; the degree sum may be odd). `info.method == :fleeting`, with the
realised `mean_degree` and `excess_degree` and `info.stubs` (M = Σk) and `info.max_degree` in its details;
`info.edges == 0`. It is the "graph" of run j of `simulate(model, net; N, seed = b)`, drawn from
`NetworkOutbreaks.stable_rng(b + j)` (design §J.7).
"""
function sample_graph(net::MFSHNetwork, N::Integer; rng::AbstractRNG = Random.default_rng())
    N = _gen_check_N(N)
    k = rand(rng, net.degrees, N)
    fc = FleetingContacts(k)
    info = _gen_info(net, :fleeting, N, 0, fc.degrees, 0, 0;
                     details = (stubs = sum(fc.degrees), max_degree = maximum(fc.degrees)))
    return fc, info
end

# ---------------------------------------------------------------------------------------------------------------
# The sampler
# ---------------------------------------------------------------------------------------------------------------

"""
    FleetingContactSSA <: OutbreakAlgorithm

The exact sampler of mean-field social heterogeneity (NetworkEpiCore `MFSHNetwork(d)`; Miller–Slim–Volz Part II
§3.2.2): Gillespie's direct method on the counts of nodes per compartment and degree class, for a
[`FleetingContacts`](@ref) population. Each stub of node v is joined at every instant to a uniform stub of the other
nodes, so a contact `s + J → X + J` with per-contact rate τ fires at a node v in s at rate
τ·k_v·Σ_{c ∈ via}(K_c − [c = s]·k_v)/(M − k_v) (K_c: stubs in c; M: all stubs). The transition is drawn from the
compartment totals, the degree class of the moving node from the residual of the same uniform number and the node
uniformly within its compartment and class; spontaneous transitions move a uniform node of their compartment. The
cost per event is O(T·D + C) (T transitions, D distinct degrees, C compartments), independent of N. Per-node
infection counts, event logs (`keep = :events`) and interventions work as in the network samplers.

As N → ∞ the fractions follow EdgeBasedModels' MFSH edge-based ODE. On a κ-regular population the hazards are those
of [`MassActionSSA`](@ref) on `WellMixed(κ)`, and the two samplers give the same trajectory from the same seed.

```julia
ens = simulate(sir_model(), MFSHNetwork(PoissonDegree(5)); N = 10_000, p = Dict(:τ => 1/12, :γ => 1/4),
               initial = SeedFraction(:I => 0.01), tspan = (0.0, 100.0), nsims = 20, seed = 1)
```
"""
struct FleetingContactSSA <: OutbreakAlgorithm end

_supports_interventions(::FleetingContactSSA) = true

function _simulate_impl(::FleetingContactSSA, spec::OutbreakSpec, rng::AbstractRNG, seed::_RecordedSeed, keep::Symbol,
                        plan::InterventionPlan = InterventionPlan())
    net = spec.network
    net isa FleetingContacts || throw(ArgumentError(
        "FleetingContactSSA simulates fleeting contacts (mean-field social heterogeneity): the network of the spec " *
        "must be a FleetingContacts population (use simulate(model, MFSHNetwork(d); N, …) or " *
        "sample_graph(MFSHNetwork(d), N)); got a $(nameof(typeof(net)))"))
    return _fleeting_run(spec, rng, seed, keep, plan, net.degrees)
end

# Node lists keyed by an integer (a compartment, or a compartment × degree class), for a uniform draw in O(1) and
# O(1) moves (swap-and-pop). The lists are built in node order.
struct _FleetingLists
    members::Vector{Vector{Int}}
    pos::Vector{Int}
    key::Vector{Int}
end

function _FleetingLists(keys::Vector{Int}, nkeys::Int)
    members = [Int[] for _ in 1:nkeys]
    pos = zeros(Int, length(keys))
    for (v, c) in pairs(keys)
        push!(members[c], v)
        pos[v] = length(members[c])
    end
    return _FleetingLists(members, pos, copy(keys))
end

function _fleeting_move!(l::_FleetingLists, v::Int, new::Int)
    old = l.key[v]
    old == new && return l
    list = l.members[old]
    p = l.pos[v]
    last_v = list[end]
    list[p] = last_v
    l.pos[last_v] = p
    pop!(list)
    push!(l.members[new], v)
    l.pos[v] = length(l.members[new])
    l.key[v] = new
    return l
end

_fleeting_member(l::_FleetingLists, c::Int, rng::AbstractRNG) = l.members[c][rand(rng, 1:length(l.members[c]))]

# The per-run state of the population: degree classes, counts per compartment × class, stubs per compartment and
# the two node lists (by compartment, for spontaneous transitions; by compartment × class, for contacts).
struct _FleetingPopulation
    k::Vector{Int}                 # stubs of each node
    class::Vector{Int}             # degree class of each node
    dval::Vector{Int}              # the distinct degrees, sorted
    ω::Vector{Float64}             # d_i/(M − d_i), 0 when a node of degree d_i holds every stub
    counts::Matrix{Int}            # n[c, i]: nodes in compartment c with degree dval[i]
    stubs::Vector{Int}             # K_c
    by_comp::_FleetingLists
    by_class::_FleetingLists
end

function _FleetingPopulation(k::Vector{Int}, node_state::Vector{Int}, C::Int)
    dval = sort!(unique(k))
    D = length(dval)
    class = Int[searchsortedfirst(dval, kv) for kv in k]
    M = sum(k)
    ω = Float64[M > d ? d / (M - d) : 0.0 for d in dval]
    counts = zeros(Int, C, D)
    stubs = zeros(Int, C)
    for v in eachindex(k)
        counts[node_state[v], class[v]] += 1
        stubs[node_state[v]] += k[v]
    end
    by_comp = _FleetingLists(node_state, C)
    by_class = _FleetingLists(Int[_fleeting_key(C, node_state[v], class[v]) for v in eachindex(k)], C * D)
    return _FleetingPopulation(k, class, dval, ω, counts, stubs, by_comp, by_class)
end

_fleeting_key(C::Int, c::Int, i::Int) = (i - 1) * C + c

# Node v has moved from compartment `old` to `new` (already recorded in the OutbreakState).
function _fleeting_moved!(pop::_FleetingPopulation, v::Int, old::Int, new::Int)
    old == new && return pop
    C = size(pop.counts, 1)
    i = pop.class[v]
    pop.counts[old, i] -= 1
    pop.counts[new, i] += 1
    pop.stubs[old] -= pop.k[v]
    pop.stubs[new] += pop.k[v]
    _fleeting_move!(pop.by_comp, v, new)
    _fleeting_move!(pop.by_class, v, _fleeting_key(C, new, i))
    return pop
end

# After an intervention: relist the moved nodes (rate changes move none).
_fleeting_sync!(pop::_FleetingPopulation, node_state::Vector{Int}, ::Nothing) = pop
function _fleeting_sync!(pop::_FleetingPopulation, node_state::Vector{Int}, touched)
    for v in touched
        v = Int(v)
        _fleeting_moved!(pop, v, pop.by_comp.key[v], node_state[v])
    end
    return pop
end

# Σ_i n[s, i]·ω_i·(K_via − [s ∈ via]·d_i): the contact hazard of compartment s per unit rate.
function _fleeting_weight(pop::_FleetingPopulation, s::Int, Kvia::Int, self::Bool)
    w = 0.0
    @inbounds for i in eachindex(pop.dval)
        m = pop.counts[s, i]
        m > 0 || continue
        w += m * pop.ω[i] * (Kvia - (self ? pop.dval[i] : 0))
    end
    return w
end

# The degree class whose cumulative weight first exceeds `r` (0 ≤ r < total); the last positive weight guards rounding.
function _fleeting_class(pop::_FleetingPopulation, s::Int, Kvia::Int, self::Bool, rate::Float64, r::Float64)
    cum = 0.0
    last_i = 0
    @inbounds for i in eachindex(pop.dval)
        m = pop.counts[s, i]
        m > 0 || continue
        w = rate * (m * pop.ω[i] * (Kvia - (self ? pop.dval[i] : 0)))
        w > 0 || continue
        last_i = i
        cum += w
        r < cum && return i
    end
    return last_i
end

function _fleeting_run(spec::OutbreakSpec, rng::AbstractRNG, seed::_RecordedSeed, keep::Symbol, plan::InterventionPlan,
                       k::Vector{Int})
    rm = _RunModel(spec.model, spec.network)
    C = rm.C
    n = length(k)
    node_state = initial_state(spec, rng)
    state = _initial_outbreak_state(rm, node_state)
    pop = _FleetingPopulation(k, node_state, C)

    # Contact transitions: recipient compartment, catalyst compartments and whether the recipient is a catalyst.
    contacts = findall(rm.is_contact)
    recipient = Int[rm.from_idx[j] for j in contacts]
    catalysts = [findall(rm.via_mask[j]) for j in contacts]
    self = Bool[rm.via_mask[j][rm.from_idx[j]] for j in contacts]
    hazard = zeros(Float64, length(contacts))
    kvia(kk) = sum((pop.stubs[c] for c in catalysts[kk]); init = 0)

    t_now = spec.tspan[1]
    t_end = spec.tspan[2]
    rec = _Recorder(keep)
    _record!(rec, t_now, state.counts)
    sched = _Schedule(nothing, plan, t_now)
    noop(_) = nothing
    on_change = touched -> _fleeting_sync!(pop, node_state, touched)
    _check_thresholds!(sched, state, rm, n, rng, on_change) && _record!(rec, t_now, state.counts)

    counts = state.counts
    while true
        # --- propensities from the counts ---
        contact_rate = 0.0
        @inbounds for kk in eachindex(contacts)
            s = recipient[kk]
            h = 0.0
            if counts[s] > 0
                Kv = kvia(kk)
                Kv > 0 && (h = rm.rates[contacts[kk]] * _fleeting_weight(pop, s, Kv, self[kk]))
            end
            hazard[kk] = h
            contact_rate += h
        end
        spontaneous_rate = 0.0
        @inbounds for c in 1:C
            spontaneous_rate += counts[c] * rm.spont_total[c]
        end
        total_rate = contact_rate + spontaneous_rate

        # --- next event time, merged with the scheduled interventions ---
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

        # --- choose the transition, then the node ---
        u = rand(rng) * total_rate
        local fired_j::Int
        local fired_node::Int
        if u < contact_rate
            # the contact (the first whose cumulative hazard exceeds u; the last positive one guards rounding), and
            # the residual of u, uniform on [0, hazard[kk]), for the degree class
            kk = 0
            before = 0.0
            cum = 0.0
            @inbounds for q in eachindex(hazard)
                hazard[q] > 0 || continue
                kk = q
                before = cum
                cum += hazard[q]
                u < cum && break
            end
            r = max(u - before, 0.0)
            fired_j = contacts[kk]
            s = recipient[kk]
            i = _fleeting_class(pop, s, kvia(kk), self[kk], rm.rates[fired_j], r)
            fired_node = _fleeting_member(pop.by_class, _fleeting_key(C, s, i), rng)
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
            fired_node = _fleeting_member(pop.by_comp, picked_c, rng)
            fired_j = _sample_spontaneous_transition(rm, picked_c, rng)
        end

        old = _fire!(state, rm, fired_node, fired_j)
        _fleeting_moved!(pop, fired_node, old, rm.to_idx[fired_j])
        _record!(rec, t_now, counts)
        _log_event!(rec, t_now, fired_j, fired_node)
        _check_thresholds!(sched, state, rm, n, rng, on_change) && _record!(rec, t_now, counts)
    end

    return _trajectory(rec, rm, state, t_end, seed, :FleetingContactSSA)
end

# ---------------------------------------------------------------------------------------------------------------
# simulate(model, net::MFSHNetwork; …)
# ---------------------------------------------------------------------------------------------------------------

"""
    simulate(model, net::MFSHNetwork; N, p = Dict(), initial, tspan, nsims = 1, graphs = :per_run,
             algorithm = FleetingContactSSA(), seed = nothing, rng = nothing, tgrid = nothing, keep = :grid,
             interventions = InterventionPlan(), parallel = false) -> OutbreakEnsemble

Simulate `nsims` runs of `model` with mean-field social heterogeneity (NetworkEpiCore `MFSHNetwork(d)`, Miller–Slim–Volz
Part II §3.2.2): N nodes with stub counts drawn iid from `net.degrees`, whose stubs make fleeting contacts with
uniformly drawn stubs of the other nodes (see [`FleetingContacts`](@ref)); the per-contact rate τ is converted with the
nominal `mean_degree(net)` as on every descriptor. Run r draws its own stub counts,
[`sample_graph`](@ref)`(net, N; rng = NetworkOutbreaks.stable_rng(b + r))` (the counts of run 1 for every run with
`graphs = :fixed`, run `mod1(r, G)` of a pool with `(:pool, G)`), and is sampled by [`FleetingContactSSA`](@ref) with
the stream `stable_rng(b + 2^32 + r)` (design §J.7). `algorithm` must be `FleetingContactSSA()` (the graph samplers have
no graph to run on); the other keywords are those of `simulate(model, net::NetworkDescriptor; …)`.

As N → ∞ the runs follow EdgeBasedModels' `edge_based(cm, MFSHNetwork(d))`; on `RegularDegree(κ)` the process is the
well-mixed one of `simulate(model, WellMixed(κ); N, …)`.

```julia
ens = simulate(sir_model(), MFSHNetwork(PoissonDegree(5)); N = 10_000, p = Dict(:τ => 1/12, :γ => 1/4),
               initial = SeedFraction(:I => 0.01), tspan = (0.0, 100.0), nsims = 20, seed = 1)
```
"""
function simulate(model, net::MFSHNetwork; algorithm::OutbreakAlgorithm = FleetingContactSSA(), kw...)
    algorithm isa FleetingContactSSA || throw(ArgumentError(
        "simulate: an MFSHNetwork (fleeting contacts, no contact graph) runs under FleetingContactSSA; got " *
        "$(nameof(typeof(algorithm)))"))
    return invoke(simulate, Tuple{Any, NetworkDescriptor}, model, net; algorithm, kw...)
end
