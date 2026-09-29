# Tests for WP26 (DESIGN_NetworkEpiCore.md §G.2 WP26, §C.2, §C.3, H25; verified issues N06 and E05):
#
#   - multiplex networks in NextReaction and HAS: bit-for-bit equal to the static sampler on one-layer (and
#     empty- or zero-weight-layer) multiplexes, and equal in distribution to DirectSSA (KS on final size);
#   - NeighbourExchangeProcess: the degree sequence is preserved exactly (also on the graph of an actual run), each
#     edge rewires at rate η within 3 SE, η = 0 is the static run event by event, NextReaction and HAS agree, and the
#     SIR final size matches the Miller–Slim–Volz dynamic fixed-degree (DFD) model (an RK4 solution written here);
#   - MassActionSSA: equal in distribution to the SSA on the complete graph K_N with rates κτ/(N − 1) (KS, N = 500),
#     the n_J − 1 of a catalyst in the recipient's own compartment, the mean final size within 3 SE of the
#     mass-action final-size equation, interventions, the WellMixed/DynamicNetwork/Scenario routes and the errors.
#
# Every ensemble is seeded (NetworkOutbreaks.stable_rng streams), so the suite is deterministic; the p-values and
# z-scores are recorded in RESULTS for reporting.

using NetworkOutbreaks
import NetworkOutbreaks as NO
using Graphs
using Test
using StableRNGs
using Statistics

const RESULTS = Dict{String, Any}()

sir(τ, γ) = OutbreakModel([:S, :I, :R], [false, true, false],
    [OutbreakTransition(:S, :I, τ, :infection), OutbreakTransition(:I, :R, γ, :spontaneous)]; name = :SIR)

# ---------------------------------------------------------------------------------------------------------------
# Statistical helpers (the same as in regressions.jl, which validates them against scipy)
# ---------------------------------------------------------------------------------------------------------------

function kolmogorov_sf(λ::Real)
    λ < 0.2 && return 1.0
    s = 0.0
    for j in 1:200
        s += (isodd(j) ? 2.0 : -2.0) * exp(-2 * j^2 * λ^2)
    end
    return clamp(s, 0.0, 1.0)
end

# Two-sample Kolmogorov–Smirnov test: (D, asymptotic p-value with Stephens' small-sample correction).
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

# Welch z-score of the difference of two sample means.
welch_z(x, y) = (mean(x) - mean(y)) / sqrt(var(x) / length(x) + var(y) / length(y))

same_run(a::OutbreakTrajectory, b::OutbreakTrajectory) =
    a.times == b.times && a.counts == b.counts && a.final_infection_counts == b.final_infection_counts

# ---------------------------------------------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------------------------------------------

# Miller–Slim–Volz dynamic fixed-degree (DFD) SIR (arXiv 1106.6320 §3.2), with uniformly random seeds ρ (q = 1 − ρ,
# verified issue E05): θ' = −τφ_I, φ_S' = −τφ_Iφ_Sψ''/ψ' + ηθπ_S − ηφ_S, φ_I' = τφ_Iφ_Sψ''/ψ' + ηθπ_I − (τ+γ+η)φ_I,
# π_R' = γπ_I, π_S = qθψ'(θ)/ψ'(1), π_I = 1 − π_R − π_S, R' = γI, S = qψ(θ), I = 1 − S − R. Returns R∞ = 1 − S(∞)
# (ever infected, seeds included). Classical RK4 with step h. It reproduces the E05 values 0.859871 (Poisson(5),
# τ = 1/6, γ = 1/4, η = 1/2, ρ = 0.01) and 0.473831 (Poisson(3), τ = 0.6, γ = 1, η = 1, ρ = 10⁻³) to 10⁻⁶.
function dfd_final_size(ψ, dψ, d2ψ, τ, γ, η, ρ; T = 400.0, h = 0.01)
    q = 1 - ρ
    ψ1 = dψ(1.0)
    function f(u)
        θ, φS, φI, πR, R = u
        πS = q * θ * dψ(θ) / ψ1
        πI = 1 - πR - πS
        I = 1 - q * ψ(θ) - R
        flux = τ * φI * φS * d2ψ(θ) / dψ(θ)
        return (-τ * φI, -flux + η * θ * πS - η * φS, flux + η * θ * πI - (τ + γ + η) * φI, γ * πI, γ * I)
    end
    u = (1.0, q, ρ, 0.0, 0.0)
    for _ in 1:round(Int, T / h)
        k1 = f(u); k2 = f(u .+ (h / 2) .* k1); k3 = f(u .+ (h / 2) .* k2); k4 = f(u .+ h .* k3)
        u = u .+ (h / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
    end
    return 1 - q * ψ(u[1])
end
regular_pgf(k) = (x -> x^k, x -> k * x^(k - 1), x -> k * (k - 1) * x^(k - 2))
poisson_pgf(μ) = (x -> exp(μ * (x - 1)), x -> μ * exp(μ * (x - 1)), x -> μ^2 * exp(μ * (x - 1)))

# Mass-action SIR final size with seeds ρ in I: S∞ = (1 − ρ) exp(−R₀ (1 − S∞)), R∞ = 1 − S∞ (bisection).
function ma_final_size(R0, ρ)
    f(S) = S - (1 - ρ) * exp(-R0 * (1 - S))
    lo, hi = 0.0, 1 - ρ - 1e-12          # f(lo) < 0 < f(hi) for R₀ > 1
    for _ in 1:200
        mid = (lo + hi) / 2
        f(mid) < 0 ? (lo = mid) : (hi = mid)
    end
    return 1 - (lo + hi) / 2
end

@testset "references reproduce the verified values" begin
    @test dfd_final_size(poisson_pgf(5)..., 1 / 6, 1 / 4, 0.5, 0.01) ≈ 0.859871 atol = 1e-6       # E05
    @test dfd_final_size(poisson_pgf(3)..., 0.6, 1.0, 1.0, 1e-3) ≈ 0.473831 atol = 1e-6           # E05 case A
    @test dfd_final_size(poisson_pgf(3)..., 0.6, 1.0, 1.0, 0.05) ≈ 0.534375 atol = 1e-6           # E05 case B
    # η = 0 is the static edge-based model: R∞ = 0.533 on the 6-regular :sir_ne_reg6 base (design §E.3)
    @test dfd_final_size(regular_pgf(6)..., 1 / 12, 1 / 4, 0.0, 0.01) ≈ 0.5326 atol = 1e-3
    @test ma_final_size(2.0, 0.01) ≈ final_size(sir_model(), WellMixed(5), Dict(:τ => 0.1, :γ => 0.25);
                                                initial = SeedFraction(:I => 0.01)) atol = 1e-9
end

# ---------------------------------------------------------------------------------------------------------------
# Multiplex networks in NextReaction and HAS (N06)
# ---------------------------------------------------------------------------------------------------------------

@testset "multiplex: NextReaction and HAS equal the static sampler on degenerate multiplexes" begin
    N = 400
    g = erdos_renyi(N, 4 / (N - 1); rng = StableRNG(11))
    g2 = erdos_renyi(N, 3 / (N - 1); rng = StableRNG(12))
    model = sir(0.3, 0.2)
    run(net, alg, s) = simulate(OutbreakSpec(model = model, network = net, initial = SeedFraction(:I => 0.02),
                                             tspan = (0.0, 200.0)); algorithm = alg, seed = s, keep = :events)
    sizes = Float64[]
    for alg in (NextReaction(), HAS()), s in (1, 2, 3)
        static = run(g, alg, s)
        push!(sizes, final_size(static))
        # one layer of weight 1
        @test same_run(run(MultiplexGraph([g], [1.0]), alg, s), static)
        # the contacts only in the second layer (the first is empty): every neighbour must be reached through
        # layer 2 (a refresh over the first layer alone would leave stale clocks)
        @test same_run(run(MultiplexGraph([SimpleGraph(N), g], [1.0, 1.0]), alg, s), static)
        # a zero-weight layer contributes nothing
        @test same_run(run(MultiplexGraph([g, g2], [1.0, 0.0]), alg, s), static)
        @test same_run(run(MultiplexGraph([g2, g], [0.0, 1.0]), alg, s), static)
    end
    @test count(>(0.1), sizes) >= 3                      # major outbreaks among them: the comparisons are not vacuous
    # the event log of a two-layer run replays: every event moves a node out of the compartment it is in, and every
    # infected node has a neighbour in I (in a layer of positive weight) at the time of its infection
    for alg in (NextReaction(), HAS())
        tr = simulate(OutbreakSpec(model = model, network = MultiplexGraph([g, g2], [0.6, 0.4]),
                                   initial = SeedNodes(:I => collect(1:8)), tspan = (0.0, 200.0));
                      algorithm = alg, seed = 5, keep = :events)
        state = fill(1, N); state[1:8] .= 2
        ok = true
        for e in events(tr)
            t = model.transitions[e.transition_index]
            ok &= state[e.node] == model.index_of[t.from]
            if t.type === :infection
                ok &= any(u -> state[u] == 2, neighbors(g, e.node)) || any(u -> state[u] == 2, neighbors(g2, e.node))
            end
            state[e.node] = model.index_of[t.to]
        end
        @test ok && length(events(tr)) > 50
        @test [count(==(c), state) for c in 1:3] == tr.counts[:, end]
    end
end

@testset "multiplex: NextReaction and HAS agree with DirectSSA in distribution" begin
    # N06: two Erdős–Rényi layers (mean degrees 3 and 2) with layer rates 0.6 and 0.4, SIR τ = γ = 1, 1% seeds.
    N = 1000
    g1 = erdos_renyi(N, 3 / (N - 1); rng = StableRNG(1))
    g2 = erdos_renyi(N, 2 / (N - 1); rng = StableRNG(2))
    net = MultiplexGraph([g1, g2], [0.6, 0.4])
    nr = 300
    spec = OutbreakSpec(model = sir(1.0, 1.0), network = net, initial = SeedFraction(:I => 0.01),
                        tspan = (0.0, 500.0))
    fs = Dict(nameof(typeof(alg)) => final_size(simulate_ensemble(spec; nsims = nr, seed = 600 + k, algorithm = alg,
                                                                  parallel = true))
              for (k, alg) in enumerate((DirectSSA(), NextReaction(), HAS())))
    @test count(>(0.1), fs[:DirectSSA]) > 100                 # enough major outbreaks to compare
    for alg in (:NextReaction, :HAS)
        D, pval = ks_2sample(fs[alg], fs[:DirectSSA])
        RESULTS["multiplex SIR KS $(alg) vs DirectSSA"] = (D = D, p = pval)
        @test pval > 1e-3
        @test abs(welch_z(fs[alg], fs[:DirectSSA])) < 4
    end

    # Two contact transitions out of S (the choice at a fired node uses the layer-weighted catalyst tally):
    # S → I via I at 0.6, S → A via {I, A} at 0.5; the statistic is the fraction of infections that went to A.
    two = OutbreakModel([:S, :I, :A, :R], [false, true, true, false],
        [OutbreakTransition(:S, :I, 0.6, :infection; via = [:I]), OutbreakTransition(:S, :A, 0.5, :infection; via = [:I, :A]),
         OutbreakTransition(:I, :R, 1.0, :spontaneous), OutbreakTransition(:A, :R, 1.0, :spontaneous)]; name = :two)
    spec2 = OutbreakSpec(model = two, network = net, initial = SeedFraction(:I => 0.02), tspan = (0.0, 500.0))
    fracA(tr) = (e = events(tr); nA = count(x -> x.transition_index == 2, e); nI = count(x -> x.transition_index == 1, e);
                 nA + nI == 0 ? missing : nA / (nA + nI))
    stat = Dict(nameof(typeof(alg)) => collect(skipmissing(fracA.(simulate_ensemble(spec2; nsims = nr, seed = 700 + k,
                                                                                     algorithm = alg, keep = :events,
                                                                                     parallel = true))))
                for (k, alg) in enumerate((DirectSSA(), NextReaction(), HAS())))
    for alg in (:NextReaction, :HAS)
        D, pval = ks_2sample(stat[alg], stat[:DirectSSA])
        RESULTS["multiplex two-transition KS $(alg) vs DirectSSA"] = (D = D, p = pval)
        @test pval > 1e-3
        @test abs(welch_z(stat[alg], stat[:DirectSSA])) < 4
    end
    # CompositionRejection still refuses multiplex networks (not a WP26 sampler)
    @test_throws ArgumentError simulate(spec; algorithm = CompositionRejection(), seed = 1)
end

@testset "multiplex: NextReaction and HAS final size matches the multiplex edge-based model (N = 10⁴)" begin
    # N06 (C): Erdős–Rényi layers with mean degrees 3 and 2 (≈ Poisson), per-layer rates 0.6 and 0.4 (γ = 1), 1% seeds.
    # For Poisson layers the multiplex EBCM (Miller–Volz) final size solves r = 1 − (1 − ρ) exp(−Σ_ℓ κ_ℓ T_ℓ r),
    # T_ℓ = τ_ℓ/(τ_ℓ + γ): r = 0.69602 (N06's independent fixed point; NetworkEpiCore's multiplex final_size agrees).
    ref = ma_final_size(3 * 0.6 / 1.6 + 2 * 0.4 / 1.4, 0.01)
    @test ref ≈ 0.69602 atol = 1e-5
    cm = ContactModel(:mpx; contacts = [Contact(:S, :I, :I, 0.6; layer = :a), Contact(:S, :I, :I, 0.4; layer = :b)],
                      transitions = [NodeTransition(:I, :R, 1.0)])
    @test final_size(cm, NO.NetworkEpiCore.MultiplexNetwork(:a => PoissonDegree(3), :b => PoissonDegree(2)),
                     Dict{Symbol, Float64}(); initial = SeedFraction(:I => 0.01)) ≈ ref atol = 1e-8
    N = 10_000
    for (k, alg) in enumerate((NextReaction(), HAS()))
        fs = map(1:40) do r
            g1 = erdos_renyi(N, 3 / (N - 1); rng = NO.stable_rng(1000k + r))
            g2 = erdos_renyi(N, 2 / (N - 1); rng = NO.stable_rng(2000k + r))
            spec = OutbreakSpec(model = sir(1.0, 1.0), network = MultiplexGraph([g1, g2], [0.6, 0.4]),
                                initial = SeedFraction(:I => 0.01), tspan = (0.0, 500.0))
            final_size(simulate(spec; algorithm = alg, seed = 3000k + r))
        end
        @test all(>(0.3), fs)                         # 100 seeds: every run is a major outbreak
        se = std(fs) / sqrt(length(fs))
        RESULTS["multiplex EB final size $(nameof(typeof(alg)))"] = (mean = mean(fs), se = se, ref = ref,
                                                                      z = (mean(fs) - ref) / se)
        @test abs(mean(fs) - ref) < 3 * se
    end
end

# ---------------------------------------------------------------------------------------------------------------
# NeighbourExchangeProcess
# ---------------------------------------------------------------------------------------------------------------

edge_set(g) = Set((src(e), dst(e)) for e in edges(g))

@testset "neighbour exchange: construction and errors" begin
    @test NeighbourExchangeProcess(1).η === 1.0
    @test NeighbourExchangeProcess(NeighbourExchange(0.5)).η == 0.5
    @test_throws ArgumentError NeighbourExchangeProcess(-1.0)
    @test_throws ArgumentError NeighbourExchangeProcess(Inf)
    @test_throws ArgumentError NeighbourExchangeProcess(NaN)
    g = random_regular_graph(50, 4; rng = StableRNG(1))
    @test DynamicGraph(g, NeighbourExchange(2.0)).process == NeighbourExchangeProcess(2.0)
    @test nv(DynamicGraph(g, NeighbourExchangeProcess(1.0))) == 50
    @test_throws ArgumentError DynamicGraph(SimpleDiGraph(g), NeighbourExchangeProcess(1.0))
    loop = copy(g); add_edge!(loop, 1, 1)
    @test_throws ArgumentError DynamicGraph(loop, NeighbourExchangeProcess(1.0))
    @test_throws ArgumentError DynamicGraph(g, DormantContacts(1.0, 0.5))     # no process yet (WP36b)
    @test_throws ArgumentError evolve_graph!(copy(g), NeighbourExchangeProcess(1.0), -1.0)
    # only NextReaction and HAS run graph processes: DirectSSA and CompositionRejection refuse (instead of silently
    # simulating the initial graph as a static one)
    spec = OutbreakSpec(model = sir(0.5, 0.2), network = DynamicGraph(g, NeighbourExchangeProcess(1.0)),
                        initial = SeedNodes(:I => [1]), tspan = (0.0, 10.0))
    @test_throws ArgumentError simulate(spec; algorithm = DirectSSA(), seed = 1)
    @test_throws ArgumentError simulate(spec; algorithm = CompositionRejection(), seed = 1)
    @test_throws ArgumentError simulate(sir_model(), DynamicNetwork(RegularDegree(4), NeighbourExchange(1.0)); N = 50,
                                        p = Dict(:τ => 0.5, :γ => 0.2), initial = SeedFraction(:I => 0.1),
                                        tspan = (0.0, 5.0), algorithm = DirectSSA())
    @test occursin("NextReaction or HAS", sprint(showerror, try simulate(spec; algorithm = DirectSSA(), seed = 1)
                                                            catch err; err end))
end

@testset "neighbour exchange: the degree sequence is preserved exactly" begin
    graphs = [random_regular_graph(2000, 6; rng = StableRNG(21)),
              erdos_renyi(2000, 5 / 1999; rng = StableRNG(22)),
              barabasi_albert(2000, 3; rng = StableRNG(23))]           # heavy-tailed: many rejections at the hubs
    for g in graphs
        h = copy(g)
        st = evolve_graph!(h, NeighbourExchangeProcess(1.5), 5.0; rng = StableRNG(24))
        @test degree(h) == degree(g)
        @test ne(h) == ne(g) && !has_self_loops(h) && !is_directed(h)
        @test st.events > 0.9 * 1.5 * 5.0 * ne(g) / 2 && st.rewired <= st.events
        @test st.rewired > 0.9 * st.events                             # rejections are rare on sparse graphs
        # after ≈ 7.5 rewirings per edge an original edge survives with probability exp(−7.5) ≈ 5.5e-4, but a pair
        # {u, v} is also joined afresh, with probability ≈ k_u k_v/(2E) in a uniform graph with these degrees (large
        # for the hubs of the Barabási–Albert graph). The overlap with the initial graph is that of an independent
        # draw: its count is within 4 Poisson SE of the expectation.
        d = degree(g); E = ne(g)
        expected = sum(min(1.0, d[src(e)] * d[dst(e)] / (2E)) for e in edges(g)) + E * exp(-7.5)
        overlap = length(intersect(edge_set(g), edge_set(h)))
        RESULTS["NE overlap $(maximum(d))"] = (overlap = overlap, expected = expected)
        @test abs(overlap - expected) < 4 * sqrt(expected)
        @test overlap < 0.02 * E
    end
    # the process's own edge list stays equal to the graph's edge set
    g = random_regular_graph(300, 4; rng = StableRNG(25))
    h = copy(g)
    ps = NO._process_state(NeighbourExchangeProcess(1.0), h, StableRNG(26))
    rng = StableRNG(27)
    for _ in 1:20_000
        NO._process_fire!(ps, h, Int[], rng)
    end
    @test Set(minmax(a, b) for (a, b) in zip(ps.src, ps.dst)) == edge_set(h)
    @test degree(h) == degree(g)
    @test NO._process_summary(ps).events == 20_000
end

# A process that wraps neighbour exchange and checks the degrees of the touched nodes on the graph of the run
# itself (the sampler's own copy): the samplers must run the process on that graph, and never on the spec's.
struct CheckedNE <: NO.GraphProcess
    inner::NeighbourExchangeProcess
    states::Vector{Any}
end
mutable struct CheckedState
    inner::Any
    g::SimpleGraph{Int}
    degrees::Vector{Int}
    fired::Int
    ok::Bool
end
function NO._process_state(p::CheckedNE, g::SimpleGraph{Int}, rng)
    s = CheckedState(NO._process_state(p.inner, g, rng), g, degree(g), 0, true)
    push!(p.states, s)
    return s
end
NO._process_rate(s::CheckedState) = NO._process_rate(s.inner)
function NO._process_fire!(s::CheckedState, g, node_state, rng)
    touched = NO._process_fire!(s.inner, g, node_state, rng)
    s.fired += 1
    s.ok &= g === s.g && all(v -> degree(g, v) == s.degrees[v], touched)
    return touched
end

@testset "neighbour exchange: degrees preserved on the graph of a NextReaction/HAS run" begin
    g = random_regular_graph(1000, 6; rng = StableRNG(31))
    g0 = copy(g)
    for alg in (NextReaction(), HAS())
        p = CheckedNE(NeighbourExchangeProcess(2.0), Any[])
        tr = simulate(OutbreakSpec(model = sir(1 / 12, 1 / 4), network = DynamicGraph(g, p),
                                   initial = SeedFraction(:I => 0.02), tspan = (0.0, 30.0)); algorithm = alg, seed = 3)
        s = only(p.states)
        @test s.fired > 10_000 && s.ok
        @test degree(s.g) == degree(g0)
        @test s.g !== g && edge_set(g) == edge_set(g0)             # the spec's graph is never mutated
        @test length(intersect(edge_set(s.g), edge_set(g0))) < 0.1 * ne(g0)   # and the run's graph did change
        @test sum(tr.counts[:, end]) == 1000
    end
end

@testset "neighbour exchange: each edge rewires at rate η (within 3 SE)" begin
    # 4-regular, N = 20000 (E = 40000), η = 0.7 over Δt = 1: about 14000 swaps. The rejection probability is
    # O(k/N) ≈ 5e-4, far below the standard errors (≈ 1%).
    η, Δt = 0.7, 1.0
    for (label, g) in (("4-regular", random_regular_graph(20_000, 4; rng = StableRNG(41))),
                       ("Poisson(5)", erdos_renyi(20_000, 5 / 19_999; rng = StableRNG(42))))
        E = ne(g)
        h = copy(g)
        st = evolve_graph!(h, NeighbourExchangeProcess(η), Δt; rng = StableRNG(43))
        # (a) swap events: Poisson with mean ηEΔt/2, so η̂ = 2·events/(EΔt) with SE √(2η/(EΔt))
        se_ev = sqrt(2η / (E * Δt))
        η_events = 2 * st.events / (E * Δt)
        η_rewired = 2 * st.rewired / (E * Δt)
        # (b) per edge: an original edge survives Δt with probability exp(−ηΔt); η̂ = −log(p̂)/Δt
        p̂ = length(intersect(edge_set(g), edge_set(h))) / E
        η_edges = -log(p̂) / Δt
        se_edges = sqrt(p̂ * (1 - p̂) / E) / (p̂ * Δt)
        RESULTS["NE rate $(label)"] = (events = η_events, rewired = η_rewired, se_events = se_ev,
                                       per_edge = η_edges, se_per_edge = se_edges,
                                       rejected = 1 - st.rewired / st.events)
        @test abs(η_events - η) < 3 * se_ev
        @test abs(η_rewired - η) < 3 * se_ev
        @test abs(η_edges - η) < 3 * se_edges
        @test degree(h) == degree(g)
    end
end

@testset "neighbour exchange: η = 0 is the static network, event by event" begin
    g = random_regular_graph(1000, 6; rng = StableRNG(51))
    model = sir(1 / 6, 1 / 4)
    for alg in (NextReaction(), HAS()), s in (1, 2)
        run(net) = simulate(OutbreakSpec(model = model, network = net, initial = SeedFraction(:I => 0.01),
                                         tspan = (0.0, 300.0)); algorithm = alg, seed = s, keep = :events)
        static = run(g)
        dyn = run(DynamicGraph(g, NeighbourExchangeProcess(0.0)))
        @test same_run(dyn, static)
        @test [(e.time, e.transition_index, e.node) for e in events(dyn)] ==
              [(e.time, e.transition_index, e.node) for e in events(static)]
        @test final_size(static) > 0.5                              # a major outbreak: the check is not vacuous
    end
    # and η > 0 changes the outcome (the process is really run)
    dyn = simulate(OutbreakSpec(model = model, network = DynamicGraph(g, NeighbourExchangeProcess(1.0)),
                                initial = SeedFraction(:I => 0.01), tspan = (0.0, 300.0)); algorithm = NextReaction(),
                   seed = 1)
    static = simulate(OutbreakSpec(model = model, network = g, initial = SeedFraction(:I => 0.01),
                                   tspan = (0.0, 300.0)); algorithm = NextReaction(), seed = 1)
    @test !same_run(dyn, static)
end

@testset "neighbour exchange: NextReaction and HAS agree in distribution" begin
    net = DynamicNetwork(RegularDegree(6), NeighbourExchange(1.0))
    p = Dict(:τ => 1 / 12, :γ => 1 / 4)
    fs = Dict(nameof(typeof(alg)) => final_size(simulate(sir_model(), net; N = 1000, p, initial = SeedFraction(:I => 0.01),
                                                         tspan = (0.0, 300.0), nsims = 150, seed = 800 + k,
                                                         algorithm = alg, parallel = true))
              for (k, alg) in enumerate((NextReaction(), HAS())))
    D, pval = ks_2sample(fs[:NextReaction], fs[:HAS])
    RESULTS["NE KS NextReaction vs HAS"] = (D = D, p = pval)
    @test pval > 1e-3
    @test abs(welch_z(fs[:NextReaction], fs[:HAS])) < 4
end

@testset "neighbour exchange: SIR final size matches the Miller–Slim–Volz DFD model" begin
    # :sir_ne_reg6_eta1 (design §E.3): 6-regular base, τ = 1/12, γ = 1/4, η = 1, 1% seeds, N = 5000. DFD R∞ = 0.74694
    # (static: 0.5326; η → ∞: mass action MA(1/2, 1/4), 0.8002). All runs are major with 50 seeds (R₀ = 1.81).
    ref = dfd_final_size(regular_pgf(6)..., 1 / 12, 1 / 4, 1.0, 0.01)
    @test ref ≈ 0.74694 atol = 1e-4
    net = DynamicNetwork(RegularDegree(6), NeighbourExchange(1.0))
    ens = simulate(sir_model(), net; N = 5000, p = Dict(:τ => 1 / 12, :γ => 1 / 4), initial = SeedFraction(:I => 0.01),
                   tspan = (0.0, 300.0), nsims = 40, seed = 20260926, parallel = true)
    fs = final_size(ens)
    @test all(>(0.3), fs)
    se = std(fs) / sqrt(length(fs))
    RESULTS["NE DFD η=1 N=5000"] = (mean = mean(fs), se = se, ref = ref, z = (mean(fs) - ref) / se)
    @test abs(mean(fs) - ref) < 3 * se
    # the static network is far below (0.533): the process matters
    @test mean(fs) - dfd_final_size(regular_pgf(6)..., 1 / 12, 1 / 4, 0.0, 0.01) > 0.15
end

@testset "neighbour exchange: simulate(model, DynamicNetwork) streams and graphs" begin
    net = DynamicNetwork(RegularDegree(4), NeighbourExchange(0.5))
    kw = (N = 200, p = Dict(:τ => 0.3, :γ => 0.25), initial = SeedFraction(:I => 0.05), tspan = (0.0, 30.0))
    a = simulate(sir_model(), net; kw..., nsims = 3, seed = 9)
    b = simulate(sir_model(), net; kw..., nsims = 3, seed = 9)
    @test all(same_run(x, y) for (x, y) in zip(a.trajectories, b.trajectories))       # reproducible
    @test [t.seed for t in a.trajectories] == [UInt64(9) + (UInt64(1) << 32) + r for r in 1:3]
    @test a.spec.network isa SampledNetwork && a.spec.network.descriptor == net
    # run r is simulate(spec_r; seed = b + 2^32 + r) on DynamicGraph(sample_graph(base, N; rng = stable_rng(b + r)))
    g2 = first(sample_graph(net.base, 200; rng = NO.stable_rng(9 + 2)))
    spec2 = OutbreakSpec(a.spec.model, DynamicGraph(g2, NeighbourExchangeProcess(0.5)), kw.initial, kw.tspan)
    tr2 = simulate(spec2; algorithm = NextReaction(), seed = a.trajectories[2].seed)
    @test NO._on_grid(tr2, a.trajectories[2].times).counts == a.trajectories[2].counts
    # graphs = :fixed: one initial graph for every run (each run evolves its own copy)
    f = simulate(sir_model(), net; kw..., nsims = 2, seed = 9, graphs = :fixed, algorithm = HAS())
    @test f.spec.network isa DynamicGraph && f.trajectories[1].algorithm === :HAS
    @test length(simulate(sir_model(), net; kw..., nsims = 2, seed = 9, graphs = (:pool, 1))) == 2
    # the scenario route (design §E.3; ALGORITHM :next_reaction)
    sc = scenario(:sir_ne_reg6_eta1)
    ens = simulate(sc; nsims = 2)
    @test length(ens) == 2 && all(t -> t.algorithm === :NextReaction, ens.trajectories)
    @test ens.spec.network isa SampledNetwork && ens.spec.network.N == 5000
end

@testset "neighbour exchange: interventions on a dynamic graph" begin
    g = random_regular_graph(500, 6; rng = StableRNG(61))
    spec = OutbreakSpec(model = sir(0.3, 0.25), network = DynamicGraph(g, NeighbourExchangeProcess(1.0)),
                        initial = SeedFraction(:I => 0.02), tspan = (0.0, 50.0))
    plan = InterventionPlan([ScheduledRateChange(5.0, :S, :I, :infection, 0.0)])
    for alg in (NextReaction(), HAS())
        tr = simulate(spec; algorithm = alg, seed = 4, keep = :events, interventions = plan)
        @test all(e -> e.transition_index != 1 || e.time <= 5.0, events(tr))    # no infection after t = 5
        @test any(e -> e.transition_index == 1, events(tr))
        @test sum(tr.counts[:, end]) == 500
    end
    # a run ends early once the epidemic is absorbing (no catalyst, no spontaneous transition): the final snapshot
    # is still at the end of the time span
    tr = simulate(OutbreakSpec(model = sir(0.0, 1.0), network = DynamicGraph(g, NeighbourExchangeProcess(50.0)),
                               initial = SeedNodes(:I => [1]), tspan = (0.0, 1e4)); algorithm = NextReaction(), seed = 1)
    @test tr.times[end] == 1e4 && tr.counts[:, end] == [499, 0, 1]
end

# ---------------------------------------------------------------------------------------------------------------
# MassActionSSA
# ---------------------------------------------------------------------------------------------------------------

@testset "MassActionSSA agrees with the SSA on the complete graph (KS, N = 500)" begin
    # WellMixed(5), τ = 1/10 (β = 1/2), γ = 1/4 (R₀ = 2), 1% seeds. The K_N process with per-edge rate κτ/(N − 1)
    # lumps exactly to the MassActionSSA counts (design §D.5 M13).
    p = Dict(:τ => 0.1, :γ => 0.25)
    kw = (N = 500, p, initial = SeedFraction(:I => 0.01), tspan = (0.0, 400.0), nsims = 300, parallel = true)
    ma = simulate(sir_model(), WellMixed(5); kw..., seed = 901)
    kn = simulate(sir_model(), WellMixed(5); kw..., seed = 902, algorithm = DirectSSA())
    @test all(t -> t.algorithm === :MassActionSSA, ma.trajectories)
    @test kn.spec.network isa MultiplexGraph && ne(kn.spec.network) == 500 * 499 ÷ 2      # K_500
    @test kn.spec.network.layer_rates ≈ [5 / 499] && kn.spec.model.transitions[1].rate == 0.1  # rates κτ/(N − 1)
    D, pval = ks_2sample(final_size(ma), final_size(kn))
    RESULTS["MassActionSSA vs K_500 KS final size"] = (D = D, p = pval)
    @test pval > 1e-3
    # also the peak prevalence, at the grid resolution
    peak(e) = [maximum(compartment_series(t, :I)) for t in e.trajectories]
    D2, p2 = ks_2sample(peak(ma), peak(kn))
    RESULTS["MassActionSSA vs K_500 KS peak"] = (D = D2, p = p2)
    @test p2 > 1e-3

    # A two-infector, branching model with a non-infectious (awareness) contact: S → E via I (0.25) and via A (0.125),
    # awareness S → Sa via {Sa, I} (0.05), E → I (0.12), E → A (0.08), I → R and A → R (0.25). κ = 4, N = 200.
    br = OutbreakModel([:S, :Sa, :E, :I, :A, :R], [false, false, false, true, true, false],
        [OutbreakTransition(:S, :E, 0.25, :infection; via = [:I]), OutbreakTransition(:S, :E, 0.125, :infection; via = [:A]),
         OutbreakTransition(:S, :Sa, 0.05, :contact_trace; via = [:Sa, :I]),
         OutbreakTransition(:E, :I, 0.12, :spontaneous), OutbreakTransition(:E, :A, 0.08, :spontaneous),
         OutbreakTransition(:I, :R, 0.25, :spontaneous), OutbreakTransition(:A, :R, 0.25, :spontaneous)]; name = :branching)
    kwb = (N = 200, initial = SeedCount(:I => 4, :Sa => 10), tspan = (0.0, 500.0), nsims = 400, keep = :counts,
           parallel = true)
    mab = simulate(br, WellMixed(4); kwb..., seed = 903)
    knb = simulate(br, WellMixed(4); kwb..., seed = 904, algorithm = NextReaction())
    for (label, stat) in (("final size", e -> final_size(e)),
                          ("final Sa", e -> [compartment_series(t, :Sa)[end] for t in e.trajectories]))
        D, pval = ks_2sample(stat(mab), stat(knb))
        RESULTS["MassActionSSA vs K_200 branching $(label)"] = (D = D, p = pval)
        @test pval > 1e-3
        @test abs(welch_z(stat(mab), stat(knb))) < 4
    end
end

@testset "MassActionSSA: a catalyst in the recipient's own compartment counts the other nodes" begin
    # Two nodes, both in I; I → J via I at rate 1 on WellMixed(1): each node sees the other one only, so the first
    # event comes at rate 2 (mean 1/2), exactly as on K_2 with per-edge rate 1. Counting the node itself would
    # give rate 4 (mean 1/4).
    m = OutbreakModel([:I, :J], [true, false], [OutbreakTransition(:I, :J, 1.0, :contact_trace; via = [:I])]; name = :self)
    first_time(e) = [first(events(t)).time for t in e.trajectories]
    kw = (N = 2, initial = SeedCount(:I => 2), tspan = (0.0, 1e3), nsims = 4000, keep = :events)
    t_ma = first_time(simulate(m, WellMixed(1.0); kw..., seed = 905))
    t_kn = first_time(simulate(m, WellMixed(1.0); kw..., seed = 906, algorithm = NextReaction()))
    se = 0.5 / sqrt(4000)
    RESULTS["self-catalyst mean first time"] = (ma = mean(t_ma), kn = mean(t_kn), se = se)
    @test abs(mean(t_ma) - 0.5) < 3 * se
    @test abs(mean(t_kn) - 0.5) < 3 * se
    # after the first event the remaining I has no other I: the run is absorbing with one event
    tr = simulate(OutbreakSpec(model = m, network = SampledNetwork(WellMixed(1.0), 2), initial = SeedCount(:I => 2),
                               tspan = (0.0, 1e3)); algorithm = MassActionSSA(), seed = 1, keep = :events)
    @test length(events(tr)) == 1 && tr.counts[:, end] == [1, 1]
    # a single node has no contacts
    tr1 = simulate(OutbreakSpec(model = sir(1.0, 0.0), network = SampledNetwork(WellMixed(5.0), 1),
                                initial = SeedCount(:I => 1), tspan = (0.0, 10.0)); algorithm = MassActionSSA(), seed = 1)
    @test tr1.counts[:, end] == [0, 1, 0]
end

@testset "MassActionSSA: mean final size within 3 SE of the mass-action final-size equation" begin
    # :sir_wm5 (design §E.3): WellMixed(5), τ = 1/10, γ = 1/4 (R₀ = 2), 1% seeds, N = 10⁴, 200 runs.
    sc = scenario(:sir_wm5)
    @test sc.sim.algorithm === :mass_action
    ens = simulate(sc; nsims = 200)
    @test all(t -> t.algorithm === :MassActionSSA, ens.trajectories)
    @test ens.spec.network isa SampledNetwork && ens.spec.network.N == sc.sim.N == 10_000
    fs = final_size(ens)
    @test all(>(0.3), fs)                            # 100 seeds: every run is a major outbreak
    ref = ma_final_size(2.0, 0.01)                    # 0.80020
    ref_nec = final_size(sc.model, sc.network, sc.params; initial = sc.initial)
    @test ref ≈ ref_nec atol = 1e-9
    se = std(fs) / sqrt(length(fs))
    RESULTS["MassActionSSA R∞ N=1e4"] = (mean = mean(fs), se = se, ref = ref, z = (mean(fs) - ref) / se)
    @test abs(mean(fs) - ref) < 3 * se
    # SEIR on WellMixed: the same final-size equation (it depends on the infectious period only through R₀)
    seir = simulate(seir_model(), WellMixed(5); N = 10_000, p = Dict(:τ => 0.1, :σ => 0.2, :γ => 0.25),
                    initial = SeedFraction(:E => 0.01), tspan = (0.0, 1000.0), nsims = 100, seed = 907)
    fse = final_size(seir)
    se2 = std(fse) / sqrt(length(fse))
    RESULTS["MassActionSSA SEIR R∞ N=1e4"] = (mean = mean(fse), se = se2, ref = ref, z = (mean(fse) - ref) / se2)
    @test abs(mean(fse) - ref) < 3 * se2
end

@testset "MassActionSSA: node bookkeeping, events and interventions" begin
    N = 300
    m = sir(0.2, 0.25)
    spec = OutbreakSpec(m, SampledNetwork(WellMixed(4.0), N), SeedNodes(:I => [1, 2, 3]), (0.0, 200.0))
    tr = simulate(spec; algorithm = MassActionSSA(), seed = 11, keep = :events)
    # replay the event log from the known seeding: every event moves a node out of its current compartment
    state = fill(1, N); state[1:3] .= 2
    ok = true
    for e in events(tr)
        t = m.transitions[e.transition_index]
        ok &= state[e.node] == m.index_of[t.from]
        state[e.node] = m.index_of[t.to]
    end
    @test ok
    @test [count(==(c), state) for c in 1:3] == tr.counts[:, end]
    @test count(>(0), tr.final_infection_counts) == N - tr.counts[1, end]
    @test simulate(spec; algorithm = MassActionSSA(), seed = 11, keep = :events).counts == tr.counts   # reproducible
    @test simulate(spec; algorithm = MassActionSSA(), rng = NO.stable_rng(11)).counts == tr.counts

    # interventions: a state change moves half of S to R at t = 5; later infections must still take S nodes (the
    # membership lists follow the moved nodes: a stale list would move vaccinated R nodes to I without lowering S)
    plan = InterventionPlan([ScheduledStateChange(5.0, :R, 0.5; from = [:S], basis = :eligible)])
    tr = simulate(OutbreakSpec(m, SampledNetwork(WellMixed(4.0), N), SeedFraction(:I => 0.02), (0.0, 200.0));
                  algorithm = MassActionSSA(), seed = 12, keep = :events, interventions = plan)
    S = compartment_series(tr, :S)
    k5 = findfirst(>=(5.0), tr.times)
    moved = round(Int, 0.5 * S[k5 - 1], RoundNearestTiesAway)
    @test tr.times[k5] == 5.0 && S[k5] == S[k5 - 1] - moved && moved > 50       # the intervention snapshot
    infections = count(e -> e.transition_index == 1, events(tr))
    @test count(e -> e.transition_index == 1 && e.time > 5.0, events(tr)) > 20      # infections after the pulse
    @test S[end] == S[1] - infections - moved                                        # each infection took an S node
    @test count(>(0), tr.final_infection_counts) == infections + tr.counts[2, 1]
    # a rate change stops transmission
    plan2 = InterventionPlan([ScheduledRateChange(5.0, :S, :I, :infection, 0.0)])
    tr2 = simulate(OutbreakSpec(m, SampledNetwork(WellMixed(4.0), N), SeedFraction(:I => 0.02), (0.0, 200.0));
                   algorithm = MassActionSSA(), seed = 13, keep = :events, interventions = plan2)
    @test all(e -> e.transition_index != 1 || e.time <= 5.0, events(tr2))
    @test any(e -> e.transition_index == 1, events(tr2))
    # threshold interventions fire on the counts
    plan3 = InterventionPlan([ThresholdIntervention(:I, :above, 20, ScheduledStateChange(0.0, :R, 1.0; from = [:S]))])
    tr3 = simulate(OutbreakSpec(m, SampledNetwork(WellMixed(4.0), N), SeedFraction(:I => 0.02), (0.0, 200.0));
                   algorithm = MassActionSSA(), seed = 14, interventions = plan3)
    @test compartment_series(tr3, :S)[end] == 0 && maximum(compartment_series(tr3, :I)) >= 20
end

@testset "MassActionSSA: errors" begin
    m = sir(0.2, 0.25)
    g = random_regular_graph(100, 4; rng = StableRNG(1))
    @test_throws ArgumentError simulate(OutbreakSpec(m, g, SeedFraction(:I => 0.05), (0.0, 10.0));
                                        algorithm = MassActionSSA(), seed = 1)
    @test_throws ArgumentError simulate(OutbreakSpec(m, SampledNetwork(ConfigurationNetwork(PoissonDegree(4)), 100),
                                                     SeedFraction(:I => 0.05), (0.0, 10.0));
                                        algorithm = MassActionSSA(), seed = 1)
    @test_throws ArgumentError simulate(sir_model(), WellMixed(5); N = 0, p = Dict(:τ => 0.1, :γ => 0.25),
                                        initial = SeedFraction(:I => 0.01), tspan = (0.0, 1.0))
    @test NO._algorithm(:mass_action) === MassActionSSA()
    @test MassActionSSA() isa OutbreakAlgorithm
end

# The statistics behind the tests above (p-values, z-scores, estimates), for the test log.
for (k, v) in sort!(collect(RESULTS); by = first)
    println("  processes: ", k, " => ", v)
end
