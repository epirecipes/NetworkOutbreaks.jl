# Tests for WP36b (DESIGN_NetworkEpiCore.md §K WP36b, §C.3, §J.7; verified issues E07 and N06): the dormant-contact
# (DC) stub process of Miller, Slim & Volz (2012), Part II §3.2.4, `DormantContactProcess`, and the route
# `simulate(model, DynamicNetwork(base, DormantContacts(η_form, η_break)); N, …)`.
#
# Every expected value comes from the definition of the process or from an independent calculation written here:
#   - bookkeeping: the edge list and the dormant-stub pool always equal the graph and max_degrees − degree;
#   - the two-state law of a stub: active with probability A = η₁/(η₁ + η₂) at stationarity (the initial state of
#     `sample_graph`), relaxation A(1 − e^{−(η₁+η₂)t}) from all-dormant, edge lifetimes Exp(η₂), activations at rate
#     η₁ per dormant stub, Binomial(k_m, A) active degrees, partners in proportion to stubs;
#   - η₁ = η₂ = 0 is the static network, event by event, under NextReaction and HAS;
#   - NextReaction and HAS agree in distribution; the streams of design §J.7;
#   - the law of large numbers: MSV's DC equations (PDF p. 10, with the ratios π_X/π), integrated here by RK4 in MSV's
#     own variables, against the scenario ensembles of :sir_dormant_msv and :sir_dormant_dvd (and the E07 verifier's
#     printed values for its scenario C check the transcription).
#
# Every ensemble is seeded, so the suite is deterministic; statistics are recorded in RESULTS.

using NetworkOutbreaks
import NetworkOutbreaks as NO
using Graphs
using Test
using StableRNGs
using Statistics

const RESULTS = Dict{String, Any}()

sir(τ, γ) = OutbreakModel([:S, :I, :R], [false, true, false],
    [OutbreakTransition(:S, :I, τ, :infection), OutbreakTransition(:I, :R, γ, :spontaneous)]; name = :SIR)

same_run(a::OutbreakTrajectory, b::OutbreakTrajectory) =
    a.times == b.times && a.counts == b.counts && a.final_infection_counts == b.final_infection_counts

edge_set(g) = Set((src(e), dst(e)) for e in edges(g))

const MSV_DEGREES = EmpiricalDegree(2 => 0.5, 8 => 0.5)            # MSV Part II Fig. 5: ψ(x) = (x² + x⁸)/2
dcnet(d, a, b) = DynamicNetwork(d, DormantContacts(a, b))

# Two-sample Kolmogorov–Smirnov test (as in processes.jl / regressions.jl).
function kolmogorov_sf(λ::Real)
    λ < 0.2 && return 1.0
    s = 0.0
    for j in 1:200
        s += (isodd(j) ? 2.0 : -2.0) * exp(-2 * j^2 * λ^2)
    end
    return clamp(s, 0.0, 1.0)
end
function ks_2sample(x::AbstractVector, y::AbstractVector)
    xs = sort(x); ys = sort(y); n1 = length(xs); n2 = length(ys)
    i = j = 0; D = 0.0
    while i < n1 && j < n2
        v = min(xs[i + 1], ys[j + 1])
        while i < n1 && xs[i + 1] == v; i += 1; end
        while j < n2 && ys[j + 1] == v; j += 1; end
        D = max(D, abs(i / n1 - j / n2))
    end
    ne_ = n1 * n2 / (n1 + n2)
    return D, kolmogorov_sf((sqrt(ne_) + 0.12 + 0.11 / sqrt(ne_)) * D)
end
welch_z(x, y) = (mean(x) - mean(y)) / sqrt(var(x) / length(x) + var(y) / length(y))

# The invariants of a per-run state on its graph: the edge list is the edge set, the pool holds max_degrees − degree
# stubs of every node, no node exceeds its stubs, the graph is simple.
function consistent(ps, g, km)
    Set(minmax(a, b) for (a, b) in zip(ps.src, ps.dst)) == edge_set(g) || return false
    length(ps.src) == ne(g) || return false
    counts = zeros(Int, nv(g))
    for v in ps.pool
        counts[v] += 1
    end
    counts == km .- degree(g) || return false
    return all(degree(g) .<= km) && !has_self_loops(g)
end

# ---------------------------------------------------------------------------------------------------------------
# MSV's dormant-contact equations, transcribed by hand (Part II §3.2.4, PDF p. 10), with the explicit seed q = 1 − ρ,
# integrated by classical RK4. Variables θ, φ_S, φ_I, φ_D, ξ_R, π_R, R; ξ = η₁/(η₁+η₂), π = η₂/(η₁+η₂);
# ξ_S = q(θ − φ_D)ψ'(θ)/ψ'(1), π_S = qφ_Dψ'(θ)/ψ'(1), ξ_I = ξ − ξ_S − ξ_R, π_I = π − π_S − π_R, S = qψ(θ):
#   θ̇ = −τφ_I,  φ̇_S = −τφ_Iφ_Sψ''/ψ' + η₁(π_S/π)φ_D − η₂φ_S,  φ̇_I = τφ_Iφ_Sψ''/ψ' + η₁(π_I/π)φ_D − (η₂+τ+γ)φ_I,
#   φ̇_D = η₂(θ − φ_D) − η₁φ_D,  ξ̇_R = −η₂ξ_R + η₁π_R + γξ_I,  π̇_R = η₂ξ_R − η₁π_R + γπ_I,  Ṙ = γ(1 − S − R),
# from θ = 1, φ_S = ξq, φ_I = ξρ, φ_D = π, ξ_R = π_R = R = 0 (uniform seeds, stubs at stationarity).
# Returns the rows (S, I, R) at t = 0, every·h, 2every·h, …, T.
# ---------------------------------------------------------------------------------------------------------------

function empirical_pgf(pairs)
    ψ(x) = sum(p * x^k for (k, p) in pairs)
    ψ1(x) = sum(p * k * x^(k - 1) for (k, p) in pairs)
    ψ2(x) = sum(p * k * (k - 1) * x^(k - 2) for (k, p) in pairs if k >= 2; init = 0.0)
    return ψ, ψ1, ψ2
end
function poisson_pgf(μ)
    return (x -> exp(μ * (x - 1)), x -> μ * exp(μ * (x - 1)), x -> μ^2 * exp(μ * (x - 1)))
end

function msv_dc_rk4(ψ, ψ1, ψ2, τ, γ, η1, η2, ρ, T; h = 1e-3, every = 100)
    q = 1 - ρ
    ξ, π = η1 / (η1 + η2), η2 / (η1 + η2)
    k̄ = ψ1(1.0)
    function f(u)
        θ, φS, φI, φD, XR, PR, R = u
        ξS = q * (θ - φD) * ψ1(θ) / k̄
        πS = q * φD * ψ1(θ) / k̄
        ξI, πI = ξ - ξS - XR, π - πS - PR
        new = τ * φI * φS * ψ2(θ) / ψ1(θ)
        S = q * ψ(θ)
        return [-τ * φI,
                -new + η1 * (πS / π) * φD - η2 * φS,
                new + η1 * (πI / π) * φD - (η2 + τ + γ) * φI,
                η2 * (θ - φD) - η1 * φD,
                -η2 * XR + η1 * PR + γ * ξI,
                η2 * XR - η1 * PR + γ * πI,
                γ * (1 - S - R)]
    end
    sir_row(u) = (S = q * ψ(u[1]); [S, 1 - S - u[7], u[7]])
    u = [1.0, ξ * q, ξ * ρ, π, 0.0, 0.0, 0.0]
    n = round(Int, T / h)
    out = [sir_row(u)]
    for s in 1:n
        k1 = f(u); k2 = f(u .+ h / 2 .* k1); k3 = f(u .+ h / 2 .* k2); k4 = f(u .+ h .* k3)
        u = u .+ h / 6 .* (k1 .+ 2k2 .+ 2k3 .+ k4)
        s % every == 0 && push!(out, sir_row(u))
    end
    return permutedims(reduce(hcat, out))
end

# ---------------------------------------------------------------------------------------------------------------
# Construction and errors
# ---------------------------------------------------------------------------------------------------------------

@testset "DormantContactProcess: construction, conversion and errors" begin
    p = DormantContactProcess(1, 1 // 2, [2, 3, 0])
    @test p.η_form === 1.0 && p.η_break === 0.5 && p.max_degrees == [2, 3, 0]
    @test DormantContactProcess(DormantContacts(2.0, 0.25), [1, 1]) == DormantContactProcess(2.0, 0.25, [1, 1])
    @test hash(DormantContactProcess(1.0, 0.5, [2])) == hash(DormantContactProcess(1.0, 0.5, [2]))
    km = [2, 3]
    q = DormantContactProcess(1.0, 0.5, km)
    km[1] = 7
    @test q.max_degrees == [2, 3]                                       # copied
    @test occursin("DormantContactProcess(η_form = 1.0, η_break = 0.5, 2 nodes, 5 stubs)", sprint(show, q))
    for bad in (-1.0, Inf, NaN)
        @test_throws ArgumentError DormantContactProcess(bad, 1.0, [1])
        @test_throws ArgumentError DormantContactProcess(1.0, bad, [1])
    end
    @test_throws ArgumentError DormantContactProcess(:η, 1.0, [1])     # a symbolic rate needs a value
    @test_throws ArgumentError DormantContactProcess(1.0, 1.0, [1, -1])
    # the graph must fit the stubs
    g = path_graph(3)                                                   # degrees 1, 2, 1
    @test DynamicGraph(g, DormantContactProcess(1.0, 1.0, [1, 2, 3])).process.max_degrees == [1, 2, 3]
    @test_throws ArgumentError DynamicGraph(g, DormantContactProcess(1.0, 1.0, [1, 1, 1]))     # node 2 has 2 edges
    @test_throws ArgumentError DynamicGraph(g, DormantContactProcess(1.0, 1.0, [1, 2]))        # wrong length
    loop = copy(g); add_edge!(loop, 1, 1)
    @test_throws ArgumentError DynamicGraph(loop, DormantContactProcess(1.0, 1.0, [3, 2, 1]))
    # a NetworkEpiCore descriptor does not say how many stubs each node of a given graph has
    err = try DynamicGraph(g, DormantContacts(1.0, 0.5)) catch e e end
    @test err isa ArgumentError && occursin("DormantContactProcess", sprint(showerror, err))
    # DormantContacts itself refuses η_form + η_break = 0 (symbolic rates need Symbolics, not loaded here; sample_graph
    # refuses them through _dc_rates)
    @test_throws ArgumentError DormantContacts(0, 0)
    @test_throws ArgumentError sample_graph(dcnet(MSV_DEGREES, 1.0, 1.0), 0)
    # only NextReaction and HAS run graph processes
    net = dcnet(MSV_DEGREES, 1.0, 1.0)
    kw = (N = 100, p = Dict(:τ => 0.5, :γ => 0.5), initial = SeedFraction(:I => 0.05), tspan = (0.0, 5.0))
    @test_throws ArgumentError simulate(sir_model(), net; kw..., algorithm = DirectSSA())
    @test_throws ArgumentError simulate(sir_model(), net; kw..., algorithm = CompositionRejection())
    dg, _ = sample_graph(net, 100; rng = NO.stable_rng(1))
    spec = OutbreakSpec(model = sir(0.5, 0.5), network = dg, initial = SeedNodes(:I => [1]), tspan = (0.0, 5.0))
    @test_throws ArgumentError simulate(spec; algorithm = DirectSSA(), seed = 1)
    @test_throws ArgumentError evolve_graph!(copy(dg.graph), dg.process, -1.0)
end

# ---------------------------------------------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------------------------------------------

@testset "the per-run state stays equal to the graph and the stubs, event by event" begin
    net = dcnet(MSV_DEGREES, 1.0, 0.5)
    dg, _ = sample_graph(net, 400; rng = NO.stable_rng(11))
    km = dg.process.max_degrees
    g = copy(dg.graph)
    E0 = ne(g)
    ps = NO._process_state(dg.process, g, StableRNG(12))
    @test consistent(ps, g, km)
    rng = StableRNG(13)
    ok = true
    touched_ok = true
    for k in 1:30_000
        before = edge_set(g)
        t = copy(NO._process_fire!(ps, g, Int[], rng))
        after = edge_set(g)
        # an event changes at most one edge, and the touched nodes are its two ends
        Δ = symdiff(before, after)
        touched_ok &= (isempty(Δ) && isempty(t)) || (length(Δ) == 1 && Set(t) == Set(only(Δ)))
        k % 1000 == 0 && (ok &= consistent(ps, g, km))
    end
    @test ok && touched_ok && consistent(ps, g, km)
    s = NO._process_summary(ps)
    @test s.events == 30_000 == s.broken + s.formed + s.rejected
    @test s.formed - s.broken == ne(g) - E0
    @test s.rejected < 0.02 * s.events                                   # pairings are rarely self-loops or repeats
    # the process state is built on (and only mutates) the graph it is given
    @test edge_set(dg.graph) != edge_set(g)
    # evolve_graph! runs the same process
    h1 = copy(dg.graph)
    st1 = evolve_graph!(h1, dg.process, 2.0; rng = StableRNG(14))
    @test st1.events > 0 && keys(st1) == (:events, :broken, :formed, :rejected)
    @test all(degree(h1) .<= km)
end

# A process that wraps DormantContactProcess and checks the stub bound on the graph of the run itself.
struct CheckedDC <: NO.GraphProcess
    inner::DormantContactProcess
    states::Vector{Any}
end
mutable struct CheckedDCState
    inner::Any
    g::SimpleGraph{Int}
    km::Vector{Int}
    fired::Int
    ok::Bool
end
function NO._process_state(p::CheckedDC, g::SimpleGraph{Int}, rng)
    s = CheckedDCState(NO._process_state(p.inner, g, rng), g, p.inner.max_degrees, 0, true)
    push!(p.states, s)
    return s
end
NO._process_rate(s::CheckedDCState) = NO._process_rate(s.inner)
function NO._process_fire!(s::CheckedDCState, g, node_state, rng)
    touched = NO._process_fire!(s.inner, g, node_state, rng)
    s.fired += 1
    s.ok &= g === s.g && all(v -> degree(g, v) <= s.km[v], touched)
    return touched
end

@testset "NextReaction and HAS run the process on the run's own graph, within the stubs" begin
    dg, _ = sample_graph(dcnet(MSV_DEGREES, 1.0, 1.0), 1000; rng = NO.stable_rng(21))
    g0 = copy(dg.graph)
    for alg in (NextReaction(), HAS())
        p = CheckedDC(dg.process, Any[])
        tr = simulate(OutbreakSpec(model = sir(1.0, 1.0), network = DynamicGraph(dg.graph, p),
                                   initial = SeedFraction(:I => 0.02), tspan = (0.0, 10.0)); algorithm = alg, seed = 3)
        s = only(p.states)
        @test s.fired > 5_000 && s.ok
        @test consistent(s.inner, s.g, dg.process.max_degrees)
        @test edge_set(dg.graph) == edge_set(g0)                         # the spec's graph is never mutated
        @test length(intersect(edge_set(s.g), edge_set(g0))) < 0.1 * ne(g0)
        @test sum(tr.counts[:, end]) == 1000
    end
end

# ---------------------------------------------------------------------------------------------------------------
# The law of one stub
# ---------------------------------------------------------------------------------------------------------------

@testset "sample_graph: the stationary state (active w.p. A, Binomial(k_m, A) degrees, partners ∝ stubs)" begin
    N = 20_000
    for (a, b) in ((1.0, 1.0), (1.0, 3.0), (5.0, 0.5))
        A = a / (a + b)
        dg, info = sample_graph(dcnet(MSV_DEGREES, a, b), N; rng = NO.stable_rng(31))
        km = dg.process.max_degrees
        g = dg.graph
        M = sum(km)
        @test dg isa DynamicGraph && dg.process == DormantContactProcess(a, b, km)
        @test info.method === :dormant_contacts && info.N == N && info.edges == ne(g)
        @test sort(unique(km)) == [2, 8] && abs(mean(km .== 8) - 0.5) < 4 * sqrt(0.25 / N)
        @test info.mean_max_degree == M / N && info.target_active_fraction == A
        @test info.active_fraction == 2ne(g) / M && info.dormant_stubs == M - 2ne(g)
        @test info.mean_degree ≈ 2ne(g) / N
        # the active fraction (edges carry two stubs: the SE of the fraction is at most √(2A(1−A)/M))
        se = sqrt(2A * (1 - A) / M)
        RESULTS["stationary A $(a),$(b)"] = (A = A, active = info.active_fraction, se = se, erased = info.erased)
        @test abs(info.active_fraction - A) < 4se
        @test info.erased < 0.01 * ne(g)
        @test all(degree(g) .<= km) && !has_self_loops(g)
        # the active degree of a node with k_m = 8 is Binomial(8, A)
        d8 = degree(g)[km .== 8]
        @test abs(mean(d8) - 8A) < 4 * sqrt(8A * (1 - A) / length(d8))
        @test var(d8) ≈ 8A * (1 - A) rtol = 0.06
        # a stub's partner is a uniform active stub: 8/(2 + 8) of the edge ends are at k_m = 8 nodes
        ends8 = sum(d8) / 2ne(g)
        @test abs(ends8 - 0.8) < 4 * sqrt(2 * 0.16 / 2ne(g))
    end
    # reproducible for a given stream
    a1, _ = sample_graph(dcnet(MSV_DEGREES, 1.0, 1.0), 500; rng = NO.stable_rng(5))
    a2, _ = sample_graph(dcnet(MSV_DEGREES, 1.0, 1.0), 500; rng = NO.stable_rng(5))
    @test edge_set(a1.graph) == edge_set(a2.graph) && a1.process == a2.process
    # η_break = 0: every stub is active (up to erasures)
    s0, i0 = sample_graph(dcnet(RegularDegree(4), 1.0, 0.0), 2000; rng = NO.stable_rng(6))
    @test i0.active_fraction > 0.99 && i0.target_active_fraction == 1
end

@testset "evolve_graph!: relaxation A(1 − e^{−(η₁+η₂)t}) from all-dormant, rates η₂ and η₁, stationarity kept" begin
    N = 20_000
    km = fill(4, N)
    M = 4N
    a, b = 1.0, 0.5
    A = a / (a + b)
    for (k, t) in enumerate((0.3, 1.0, 3.0))
        g = SimpleGraph(N)                                              # every stub dormant
        st = evolve_graph!(g, DormantContactProcess(a, b, km), t; rng = StableRNG(40 + k))
        pt = A * (1 - exp(-(a + b) * t))
        frac = 2ne(g) / M
        se = sqrt(2pt * (1 - pt) / M)
        RESULTS["relaxation t=$(t)"] = (expected = pt, observed = frac, se = se, rejected = st.rejected)
        @test abs(frac - pt) < 4se
        @test all(degree(g) .<= km)
    end
    # from the stationary state: break events at rate η₂ per active edge, formations at rate η₁D/2 (each dormant stub
    # activates at rate η₁), the active fraction stays A, and an initial edge survives Δt with probability e^{−η₂Δt}
    for (a, b) in ((1.0, 0.5), (0.4, 2.0))
        A = a / (a + b)
        dg, _ = sample_graph(dcnet(PoissonDegree(5), a, b), N; rng = NO.stable_rng(50))
        g = copy(dg.graph)
        km = dg.process.max_degrees
        M = sum(km)
        E0 = ne(g)
        Δt = 1.0
        st = evolve_graph!(g, dg.process, Δt; rng = StableRNG(51))
        Ē = (E0 + ne(g)) / 2                                           # the active edges stay ≈ constant
        D̄ = M - 2Ē
        z_break = (st.broken - b * Ē * Δt) / sqrt(b * Ē * Δt)
        z_form = (st.formed + st.rejected - a * D̄ / 2 * Δt) / sqrt(a * D̄ / 2 * Δt)
        frac = 2ne(g) / M
        se = sqrt(2A * (1 - A) / M)
        p̂ = length(intersect(edge_set(dg.graph), edge_set(g))) / E0
        η̂ = -log(p̂) / Δt
        se_η = sqrt(p̂ * (1 - p̂) / E0) / (p̂ * Δt)
        RESULTS["rates $(a),$(b)"] = (z_break = z_break, z_form = z_form, active = frac, A = A, η2_hat = η̂, se_η2 = se_η,
                                      rejected = st.rejected / max(st.formed + st.rejected, 1))
        @test abs(z_break) < 4 && abs(z_form) < 4
        @test abs(frac - A) < 4se
        @test abs(η̂ - b) < 4se_η
        @test st.rejected < 0.01 * (st.formed + st.rejected)
    end
end

# ---------------------------------------------------------------------------------------------------------------
# The epidemic on the process
# ---------------------------------------------------------------------------------------------------------------

@testset "η_form = η_break = 0 is the static network, event by event" begin
    dg, _ = sample_graph(dcnet(MSV_DEGREES, 1.0, 1.0), 1000; rng = NO.stable_rng(61))
    g = dg.graph
    frozen = DormantContactProcess(0.0, 0.0, dg.process.max_degrees)
    model = sir(1.0, 0.5)
    for alg in (NextReaction(), HAS()), s in (1, 2)
        run(net) = simulate(OutbreakSpec(model = model, network = net, initial = SeedFraction(:I => 0.01),
                                         tspan = (0.0, 100.0)); algorithm = alg, seed = s, keep = :events)
        static = run(g)
        dyn = run(DynamicGraph(g, frozen))
        @test same_run(dyn, static)
        @test [(e.time, e.transition_index, e.node) for e in events(dyn)] ==
              [(e.time, e.transition_index, e.node) for e in events(static)]
    end
    # and a live process changes the outcome
    run2(net) = simulate(OutbreakSpec(model = model, network = net, initial = SeedFraction(:I => 0.01),
                                      tspan = (0.0, 100.0)); algorithm = NextReaction(), seed = 1)
    @test !same_run(run2(dg), run2(g))
end

@testset "simulate(model, DynamicNetwork(base, DormantContacts)): streams (§J.7), graphs modes, the scenario route" begin
    net = dcnet(MSV_DEGREES, 1.0, 1.0)
    kw = (N = 300, p = Dict(:τ => 1.0, :γ => 1.0), initial = SeedFraction(:I => 0.05), tspan = (0.0, 15.0))
    a = simulate(sir_model(), net; kw..., nsims = 3, seed = 9)
    b = simulate(sir_model(), net; kw..., nsims = 3, seed = 9)
    @test all(same_run(x, y) for (x, y) in zip(a.trajectories, b.trajectories))
    @test [t.seed for t in a.trajectories] == [UInt64(9) + (UInt64(1) << 32) + r for r in 1:3]
    @test a.spec.network isa SampledNetwork && a.spec.network.descriptor == net
    # run r is simulate(spec_r; seed = b + 2³² + r) on sample_graph(net, N; rng = stable_rng(b + r)): the stationary
    # stub state is the process's own initial state (not a sample of the base)
    dg2 = first(sample_graph(net, 300; rng = NO.stable_rng(9 + 2)))
    @test dg2.process isa DormantContactProcess
    tr2 = simulate(OutbreakSpec(a.spec.model, dg2, kw.initial, kw.tspan); algorithm = NextReaction(),
                   seed = a.trajectories[2].seed)
    @test NO._on_grid(tr2, a.trajectories[2].times).counts == a.trajectories[2].counts
    @test NO._initial_network(net, 300, NO.stable_rng(4)).graph == first(sample_graph(net, 300; rng = NO.stable_rng(4))).graph
    f = simulate(sir_model(), net; kw..., nsims = 2, seed = 9, graphs = :fixed, algorithm = HAS())
    @test f.spec.network isa DynamicGraph && f.spec.network.process isa DormantContactProcess
    @test f.trajectories[1].algorithm === :HAS
    @test length(simulate(sir_model(), net; kw..., nsims = 2, seed = 9, graphs = (:pool, 1))) == 2
    # the scenario route (the runner draws the stationary state through _initial_network)
    sc = derive(scenario(:sir_dormant_msv); nsims = 2)
    ens = simulate(sc)
    @test length(ens) == 2 && all(t -> t.algorithm === :NextReaction, ens.trajectories)
    n1, _ = NO._scenario_network(sc, 1)
    @test n1 isa DynamicGraph && n1.process isa DormantContactProcess && nv(n1.graph) == sc.sim.N
end

@testset "NextReaction and HAS agree in distribution" begin
    net = dcnet(MSV_DEGREES, 1.0, 1.0)
    p = Dict(:τ => 1.0, :γ => 1.0)
    fs = Dict(nameof(typeof(alg)) => final_size(simulate(sir_model(), net; N = 1000, p, initial = SeedFraction(:I => 0.02),
                                                         tspan = (0.0, 40.0), nsims = 150, seed = 900 + k,
                                                         algorithm = alg, parallel = true))
              for (k, alg) in enumerate((NextReaction(), HAS())))
    D, pval = ks_2sample(fs[:NextReaction], fs[:HAS])
    RESULTS["DC KS NextReaction vs HAS"] = (D = D, p = pval)
    @test pval > 1e-3
    @test abs(welch_z(fs[:NextReaction], fs[:HAS])) < 4
end

@testset "law of large numbers: MSV's DC equations (Part II §3.2.4) on :sir_dormant_msv and :sir_dormant_dvd" begin
    # The transcription reproduces the E07 verifier's DC ODE values for its scenario C (Poisson(3), τ = 2, γ = 1,
    # η₂ = 0.5, 0.1% seeded): R∞ = 0.6258 (η₁ = 1) and 0.7993 (η₁ = 5); its exact stub SSA gave 0.6264 ± 0.0011 and
    # 0.7992 ± 0.0005.
    for (η1, ref) in ((1.0, 0.6258), (5.0, 0.7993))
        x = msv_dc_rk4(poisson_pgf(3.0)..., 2.0, 1.0, η1, 0.5, 1e-3, 400.0; h = 0.01, every = 40_000)
        @test x[end, 3] ≈ ref atol = 2e-4
    end
    # The scenarios' own ensembles: N = 10⁴, 200 runs, 1% seeded, a fresh stationary stub state per run, runs
    # conditioned on a major outbreak (every run is one with 100 seeds).
    for id in (:sir_dormant_msv, :sir_dormant_dvd)
        sc = scenario(id)
        N = sc.sim.N
        pk = sc.network.base.degrees.p                                  # p[k + 1] = P(k_m = k)
        pairs = [(k - 1, pk[k]) for k in eachindex(pk) if pk[k] > 0]
        τ, γ = sc.params[:τ], sc.params[:γ]
        η1, η2 = Float64(sc.network.process.η_form), Float64(sc.network.process.η_break)
        ρ = only(last.(seed_fractions(sc.initial)))
        step_ = Float64(step(sc.tgrid))
        ode = msv_dc_rk4(empirical_pgf(pairs)..., τ, γ, η1, η2, ρ, sc.tspan[2]; h = step_ / 100, every = 100)
        @test size(ode, 1) == length(sc.tgrid)
        ens = simulate(sc.model, sc.network; N, p = sc.params, initial = sc.initial, tspan = sc.tspan,
                       nsims = sc.sim.nsims, seed = sc.sim.base_seed, tgrid = sc.tgrid, parallel = true)
        nseed = round(Int, ρ * N)
        major = [tr for tr in ens.trajectories if final_size(tr) * N - nseed >= 0.05 * N]
        @test length(major) == sc.sim.nsims
        ix = [ens.spec.model.index_of[X] for X in (:S, :I, :R)]
        A = cat([Float64.(permutedims(tr.counts[ix, :])) ./ N for tr in major]...; dims = 3)   # time × (S, I, R) × run
        μ = dropdims(mean(A; dims = 3); dims = 3)
        se = dropdims(std(A; dims = 3); dims = 3) ./ sqrt(length(major))
        D = vec(maximum(abs.(μ .- ode); dims = 1))
        fs = final_size.(major)
        zR = (mean(fs) - ode[end, 3]) / (std(fs) / sqrt(length(fs)))
        RESULTS["LLN $(id)"] = (D∞ = D, se_max = vec(maximum(se; dims = 1)), R∞ = mean(fs), ref = ode[end, 3], z = zR)
        @info "WP36b DormantContactProcess vs MSV's DC ODE" scenario = id N runs = length(major) D∞_SIR = join(round.(D; sigdigits = 3), ", ") R∞ = mean(fs) R∞_ODE = ode[end, 3] z_R∞ = zR
        @test all(<(0.01), D)
        @test D[2] < 0.005                                              # prevalence, the design's §E.2 criterion
        @test abs(zR) < 4
    end
end
