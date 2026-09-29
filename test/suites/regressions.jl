# Regression tests for the NetworkOutbreaks prerequisite fixes (DESIGN_NetworkEpiCore.md §A.5, §G.2 WP3):
# M1–M6, m3, m5, rng threading, ALGORITHM_REVISION, and the cross-algorithm χ² and KS acceptance tests.
# References are exact probabilities, an independent SSA written here, or EBCM values computed with scipy
# (quoted inline). See VERIFIED_ISSUES.md N03, N04, N05 for the original reproducers.

using NetworkOutbreaks
import NetworkOutbreaks as NO
using Graphs
using Test
using StableRNGs
using Statistics
using Random: randexp

const ALGS = (DirectSSA(), NextReaction(), CompositionRejection(), HAS())
# p-values of the acceptance tests, kept for reporting (read by scripts that include this suite)
const PVALUES = Dict{String, Float64}()
const ALGS_SCHED = (DirectSSA(), NextReaction(), HAS())        # time-varying networks and interventions
algname(a) = nameof(typeof(a))

sir_model(β, γ) = OutbreakModel([:S, :I, :R], [false, true, false],
    [OutbreakTransition(:S, :I, β, :infection), OutbreakTransition(:I, :R, γ, :spontaneous)]; name = :SIR)
si_model(β) = OutbreakModel([:S, :I], [false, true], [OutbreakTransition(:S, :I, β, :infection)]; name = :SI)

# Binomial z-test helper: |p̂ − p| within `z` standard errors.
within(p̂, p, n; z = 4) = abs(p̂ - p) <= z * sqrt(p * (1 - p) / n)

# ---------------------------------------------------------------------------------------------------------------
# Statistical helpers (Base only; validated against scipy below)
# ---------------------------------------------------------------------------------------------------------------

# Upper tail of the χ² distribution with an even number of degrees of freedom 2m:
# Q = e^{-x/2} Σ_{i<m} (x/2)^i / i!  (exact).
function chisq_sf_even(x::Real, df::Integer)
    iseven(df) && df > 0 || throw(ArgumentError("df must be even and positive"))
    m = df ÷ 2
    h = x / 2
    term = 1.0
    s = 1.0
    for i in 1:(m - 1)
        term *= h / i
        s += term
    end
    return exp(-h) * s
end

# Kolmogorov distribution tail Q_KS(λ) = 2 Σ_{j≥1} (−1)^{j−1} exp(−2 j² λ²).
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
    ne = n1 * n2 / (n1 + n2)
    return D, kolmogorov_sf((sqrt(ne) + 0.12 + 0.11 / sqrt(ne)) * D)
end

# Pearson χ² goodness of fit of `observed` counts to probabilities `p`.
function chisq_gof(observed::AbstractVector{<:Integer}, p::AbstractVector{<:Real})
    n = sum(observed)
    x = sum((observed[k] - n * p[k])^2 / (n * p[k]) for k in eachindex(p))
    return x, chisq_sf_even(x, length(p) - 1)
end

# Pearson χ² test of homogeneity for a (groups × categories) table of counts.
function chisq_homogeneity(table::AbstractMatrix{<:Integer})
    r = vec(sum(table; dims = 2)); c = vec(sum(table; dims = 1)); n = sum(table)
    x = 0.0
    for a in axes(table, 1), k in axes(table, 2)
        e = r[a] * c[k] / n
        x += (table[a, k] - e)^2 / e
    end
    return x, chisq_sf_even(x, (size(table, 1) - 1) * (size(table, 2) - 1))
end

@testset "statistical helpers agree with scipy" begin
    # scipy.stats.chi2.sf and scipy.special.kolmogorov
    @test chisq_sf_even(10.0, 6) ≈ 0.12465201948308109 rtol = 1e-12
    @test chisq_sf_even(30.0, 18) ≈ 0.037446493479672875 rtol = 1e-12
    @test chisq_sf_even(40.0, 18) ≈ 0.002087259049135014 rtol = 1e-12
    @test chisq_sf_even(5.0, 2) ≈ 0.0820849986238988 rtol = 1e-12
    @test kolmogorov_sf(0.5) ≈ 0.9639452436648751 rtol = 1e-10
    @test kolmogorov_sf(1.0) ≈ 0.26999967167735456 rtol = 1e-10
    @test kolmogorov_sf(2.0) ≈ 0.0006709252557796953 rtol = 1e-8
    D, p = ks_2sample([1, 2, 3, 4], [1, 2, 3, 4]); @test D == 0 && p == 1
    D, _ = ks_2sample(collect(1:10), collect(11:20)); @test D == 1
end

# ---------------------------------------------------------------------------------------------------------------
# ALGORITHM_REVISION and RNG threading
# ---------------------------------------------------------------------------------------------------------------

@testset "ALGORITHM_REVISION" begin
    @test NetworkOutbreaks.ALGORITHM_REVISION isa String
    @test ALGORITHM_REVISION === NetworkOutbreaks.ALGORITHM_REVISION     # exported
    @test ALGORITHM_REVISION == "2"
end

@testset "rng threading: seed builds a mixed StableRNG, rng is used as given" begin
    g = random_regular_graph(300, 4; rng = StableRNG(1))
    spec = OutbreakSpec(model = sir_model(0.5, 0.5), network = g, initial = SeedFraction(:I => 0.02),
                        tspan = (0.0, 30.0))
    for alg in ALGS
        a = simulate(spec; algorithm = alg, seed = 17)
        b = simulate(spec; algorithm = alg, rng = NO.stable_rng(17))
        @test a.times == b.times && a.counts == b.counts
        @test a.seed == 17 && b.seed === nothing      # an explicit rng records no seed (src/events.jl)
        @test NO.stable_rng(17) isa StableRNG
        # an explicit rng is advanced: two calls differ, and replaying the generator reproduces both
        r = StableRNG(5)
        c1 = simulate(spec; algorithm = alg, rng = r); c2 = simulate(spec; algorithm = alg, rng = r)
        r = StableRNG(5)
        d1 = simulate(spec; algorithm = alg, rng = r); d2 = simulate(spec; algorithm = alg, rng = r)
        @test c1.counts == d1.counts && c2.counts == d2.counts && c1.times != c2.times
    end
    @test_throws ArgumentError simulate(spec; seed = 1, rng = StableRNG(1))
    @test_throws ArgumentError simulate(spec; seed = -1)
    # Ensembles: child k is simulate(spec; seed = child seed k), with rng-derived child seeds too.
    e1 = simulate_ensemble(spec; nsims = 4, rng = StableRNG(9), algorithm = NextReaction())
    rr = StableRNG(9); kids = [rand(rr, UInt64) for _ in 1:4]
    @test [t.seed for t in e1] == kids
    @test all(e1[k].counts == simulate(spec; algorithm = NextReaction(), seed = kids[k]).counts for k in 1:4)
    e2 = simulate_ensemble(spec; nsims = 4, seed = 3, algorithm = NextReaction())
    @test [t.seed for t in e2] == [NO._ensemble_child_seed(UInt64(3), k) for k in 1:4]
    @test_throws ArgumentError simulate_ensemble(spec; nsims = 2, seed = 1, rng = StableRNG(1))
    # Seed mixing: raw StableRNG(s) and StableRNG(s + 1) give first draws with correlation ≈ -0.43;
    # the mixed generators must not.
    x = [rand(NO.stable_rng(s)) for s in 1:20_000]
    y = [rand(NO.stable_rng(s + 1)) for s in 1:20_000]
    @test abs(cor(x, y)) < 4 / sqrt(20_000)
    xr = [rand(StableRNG(s)) for s in 1:20_000]; yr = [rand(StableRNG(s + 1)) for s in 1:20_000]
    @test cor(xr, yr) < -0.3        # documents why the mixing is needed
end

# ---------------------------------------------------------------------------------------------------------------
# M1: interventions work on a per-run copy of the rates (N04 #2)
# ---------------------------------------------------------------------------------------------------------------

@testset "M1: rate changes never mutate the spec; ensemble runs are independent" begin
    m = sir_model(1.0, 1.0)
    g = random_regular_graph(500, 4; rng = StableRNG(1))
    spec = OutbreakSpec(model = m, network = g, initial = SeedFraction(:I => 0.02), tspan = (0.0, 20.0))
    plan = InterventionPlan([ScheduledRateChange(1.0, :S, :I, :infection, 0.1)])
    for alg in ALGS_SCHED
        ens = simulate_ensemble(spec; nsims = 8, seed = 3, algorithm = alg, interventions = plan)
        @test spec.model.transitions[1].rate == 1.0
        @test all(tr.model === spec.model for tr in ens)
        ref = [simulate(OutbreakSpec(deepcopy(m), spec.network, spec.initial, spec.tspan);
                        algorithm = alg, seed = NO._ensemble_child_seed(UInt64(3), k), interventions = plan)
               for k in 1:8]
        @test all(ens[k].counts == ref[k].counts && ens[k].times == ref[k].times for k in 1:8)
        ensp = simulate_ensemble(spec; nsims = 8, seed = 3, algorithm = alg, interventions = plan, parallel = true)
        @test all(ensp[k].counts == ens[k].counts for k in 1:8)
        # a later plain run uses the original rate again
        plain = simulate(spec; algorithm = alg, seed = 11)
        again = simulate(OutbreakSpec(deepcopy(m), spec.network, spec.initial, spec.tspan); algorithm = alg, seed = 11)
        @test plain.counts == again.counts
    end
    # the rate change takes effect exactly at its time: rate 0 after t = 1 ⇒ no infection event after t = 1
    plan0 = InterventionPlan([ScheduledRateChange(1.0, :S, :I, :infection, 0.0)])
    for alg in ALGS_SCHED
        tr = simulate(spec; algorithm = alg, seed = 4, keep = :events, interventions = plan0)
        @test !any(e -> e.transition_index == 1 && e.time > 1.0, tr.events)
        @test any(e -> e.transition_index == 1 && e.time < 1.0, tr.events)
    end
    # a threshold-triggered rate change does not mutate the spec either
    thr = InterventionPlan([ThresholdIntervention(:I, :above, 20,
                                                  ScheduledRateChange(NaN, :S, :I, :infection, 0.05))])
    simulate_ensemble(spec; nsims = 3, seed = 1, interventions = thr)
    @test spec.model.transitions[1].rate == 1.0
    # ambiguous rate changes are rejected; `via` selects one transition
    two = OutbreakModel([:S, :E, :A, :I], [false, false, true, true],
        [OutbreakTransition(:S, :E, 1.0, :infection; via = [:I]),
         OutbreakTransition(:S, :E, 1.0, :infection; via = [:A])])
    spec2 = OutbreakSpec(model = two, network = g, initial = SeedFraction(:I => 0.05, :A => 0.05), tspan = (0.0, 5.0))
    @test_throws ArgumentError simulate(spec2; interventions = InterventionPlan([ScheduledRateChange(1.0, :S, :E, :infection, 0.0)]))
    @test_throws ArgumentError simulate(spec2; interventions = InterventionPlan([ScheduledRateChange(1.0, :S, :R, :infection, 0.0)]))
    only_A = InterventionPlan([ScheduledRateChange(1.0, :S, :E, :infection, 0.0; via = [:A])])
    for alg in ALGS_SCHED
        tr = simulate(spec2; algorithm = alg, seed = 2, keep = :events, interventions = only_A)
        @test !any(e -> e.transition_index == 2 && e.time > 1.0, tr.events)
        @test any(e -> e.transition_index == 1 && e.time > 1.0, tr.events)
    end
    # an empty model `via` means the infectious compartments, so it is matched by their names (was: unmatched)
    @test NO._rate_change_target(ScheduledRateChange(1.0, :S, :I, :infection, 0.0; via = [:I]), spec.model) == 1
    @test_throws ArgumentError NO._rate_change_target(ScheduledRateChange(1.0, :S, :I, :infection, 0.0; via = [:R]),
                                                      spec.model)
    # every constructor validates (the positional 6-argument default constructor bypassed the checks)
    @test_throws ArgumentError ScheduledRateChange(1.0, :S, :I, :infection, -1.0)
    @test_throws ArgumentError ScheduledRateChange(1.0, :S, :I, :bogus, 1.0)
    @test_throws MethodError ScheduledRateChange(1.0, :S, :I, :infection, -1.0, nothing)
    @test_throws ArgumentError simulate(spec; algorithm = CompositionRejection(), interventions = plan)
end

# ---------------------------------------------------------------------------------------------------------------
# M2: one `via` semantics in every algorithm, including DirectSSA's choice among several transitions (N03)
# ---------------------------------------------------------------------------------------------------------------

# Independent direct-method SSA for per-node, via-catalysed models (written from scratch: explicit neighbour
# counts, no NetworkOutbreaks internals). Returns the final compartment counts.
function reference_ssa(g, comps, infectious, trs, state0, tmax, rng)
    idx = Dict(c => i for (i, c) in enumerate(comps)); n = nv(g)
    state = copy(state0); t = 0.0
    via(tr) = isempty(tr.via) ? findall(infectious) : [idx[c] for c in tr.via]
    vias = [via(tr) for tr in trs]
    while true
        rates = Float64[]; who = Tuple{Int, Int}[]
        for v in 1:n, (j, tr) in enumerate(trs)
            state[v] == idx[tr.from] || continue
            h = tr.type === :spontaneous ? tr.rate : tr.rate * count(u -> state[u] in vias[j], neighbors(g, v))
            h > 0 || continue
            push!(rates, h); push!(who, (v, j))
        end
        total = sum(rates; init = 0.0)
        total > 0 || break
        t += randexp(rng) / total
        t > tmax && break
        u = rand(rng) * total; k = findfirst(>(u), cumsum(rates))
        k === nothing && (k = length(rates))
        v, j = who[k]; state[v] = idx[trs[j].to]
    end
    return [count(==(i), state) for i in eachindex(comps)]
end

@testset "M2: via catalysts need not be infectious; all algorithms agree" begin
    comps = [:S, :I, :D, :X]; inf = [false, true, false, false]
    # (a) centre S, leaves I and D; S→I at 1 via I, S→X at 3 via D. Exact: E[T] = 1/4, P(X) = 3/4.
    model = OutbreakModel(comps, inf,
        [OutbreakTransition(:S, :I, 1.0, :infection; via = [:I]),
         OutbreakTransition(:S, :X, 3.0, :infection; via = [:D])])
    spec = OutbreakSpec(model = model, network = star_graph(3),
                        initial = SeedNodes(:I => [2], :D => [3]; default = :S), tspan = (0.0, 50.0))
    nrep = 4000
    # a different seed per algorithm, so that the algorithms are independent checks against the exact values
    for (a, alg) in enumerate(ALGS)
        trajs = simulate_ensemble(spec; nsims = nrep, seed = 100 + a, algorithm = alg, keep = :events).trajectories
        @test all(length(t.events) == 1 for t in trajs)
        T = [t.events[1].time for t in trajs]
        @test abs(mean(T) - 1 / 4) < 4 * (1 / 4) / sqrt(nrep)                 # sd of Exp(4) is 1/4
        @test within(mean(t.events[1].transition_index == 2 for t in trajs), 3 / 4, nrep)
    end
    # (b) the only catalyst is non-infectious: S→X at 1 via D on a path. Exact: E[T] = 1.
    m1 = OutbreakModel(comps, inf, [OutbreakTransition(:S, :X, 1.0, :infection; via = [:D])])
    spec1 = OutbreakSpec(model = m1, network = path_graph(2),
                         initial = SeedNodes(:D => [2]; default = :S), tspan = (0.0, 50.0))
    for (a, alg) in enumerate(ALGS)
        trajs = simulate_ensemble(spec1; nsims = 2000, seed = 200 + a, algorithm = alg, keep = :events).trajectories
        @test all(length(t.events) == 1 for t in trajs)
        @test abs(mean(t.events[1].time for t in trajs) - 1.0) < 4 / sqrt(2000)
    end
    # (c) peer "awareness" A (not infectious) spreads alongside SIR: every algorithm agrees with an independent SSA
    compsA = [:S, :I, :R, :A]; infA = [false, true, false, false]
    trsA = [OutbreakTransition(:S, :I, 0.4, :infection; via = [:I]),
            OutbreakTransition(:S, :A, 0.15, :infection; via = [:A]),
            OutbreakTransition(:I, :R, 1.0, :spontaneous)]
    mA = OutbreakModel(compsA, infA, trsA)
    gA = erdos_renyi(600, 6 / 599; rng = StableRNG(7))
    specA = OutbreakSpec(model = mA, network = gA,
                         initial = SeedNodes(:I => collect(1:15), :A => collect(16:30); default = :S),
                         tspan = (0.0, 200.0))
    nr = 60
    rrng = StableRNG(2024)
    s0 = [v <= 15 ? 2 : v <= 30 ? 4 : 1 for v in 1:600]
    refA = [reference_ssa(gA, compsA, infA, trsA, s0, 200.0, rrng)[4] / 600 for _ in 1:nr]
    for alg in ALGS
        ens = simulate_ensemble(specA; nsims = nr, seed = 300, algorithm = alg)
        a = [t.counts[4, end] / 600 for t in ens]
        se = sqrt(var(a) / nr + var(refA) / nr)
        @test abs(mean(a) - mean(refA)) < 4 * se
        @test mean(a) > 0.2          # the 0.1 DirectSSA gave ≈ 0.01 (A never spread)
    end
end

# ---------------------------------------------------------------------------------------------------------------
# M3: :contact_trace in NextReaction, CompositionRejection and HAS (N05)
# ---------------------------------------------------------------------------------------------------------------

@testset "M3: contact tracing runs in every algorithm and matches the EBCM" begin
    comps = [:S, :I, :D, :X]; inf = [false, true, false, false]
    m2 = OutbreakModel(comps, inf, [OutbreakTransition(:S, :X, 1.0, :contact_trace; via = [:D])])
    spec2 = OutbreakSpec(model = m2, network = path_graph(2),
                         initial = SeedNodes(:D => [2]; default = :S), tspan = (0.0, 50.0))
    for (a, alg) in enumerate(ALGS)                                     # independent seeds per algorithm
        trajs = simulate_ensemble(spec2; nsims = 1000, seed = 5 + a, algorithm = alg, keep = :events).trajectories
        @test all(length(t.events) == 1 for t in trajs)              # was KeyError in NR/CR/HAS
        @test abs(mean(t.events[1].time for t in trajs) - 1.0) < 4 / sqrt(1000)
    end
    # SIRQ on a 5-regular graph, N = 2000, 1% seeds; β = 0.5, tracing α = 0.2, γ = 0.5.
    # EBCM with competing contacts (scipy, rtol 1e-11): Q∞ = 0.247466, R∞ = 0.628665, S∞ = 0.123869.
    N = 2000
    G = random_regular_graph(N, 5; rng = StableRNG(1))
    sirq = OutbreakModel([:S, :I, :R, :Q], [false, true, false, false],
        [OutbreakTransition(:S, :I, 0.5, :infection; via = [:I]),
         OutbreakTransition(:I, :R, 0.5, :spontaneous),
         OutbreakTransition(:S, :Q, 0.2, :contact_trace; via = [:I])])
    spec = OutbreakSpec(model = sirq, network = G, initial = SeedFraction(:I => 0.01), tspan = (0.0, 80.0))
    for alg in ALGS
        ens = simulate_ensemble(spec; nsims = 20, seed = 1, algorithm = alg)
        @test isapprox(mean(t.counts[4, end] for t in ens) / N, 0.247466; atol = 0.01)
        @test isapprox(mean(t.counts[3, end] for t in ens) / N, 0.628665; atol = 0.015)
    end
    # :contact_trace is exactly an edge-mediated contact: relabelling it :infection changes nothing, bit for bit
    # (so mapping every Contact to :infection, as OutbreakModel(::ContactModel) will, preserves results).
    relabel = OutbreakModel([:S, :I, :R, :Q], [false, true, false, false],
        [OutbreakTransition(:S, :I, 0.5, :infection; via = [:I]),
         OutbreakTransition(:I, :R, 0.5, :spontaneous),
         OutbreakTransition(:S, :Q, 0.2, :infection; via = [:I])])
    specR = OutbreakSpec(model = relabel, network = G, initial = SeedFraction(:I => 0.01), tspan = (0.0, 80.0))
    for alg in ALGS
        a = simulate(spec; algorithm = alg, seed = 9, keep = :events)
        b = simulate(specR; algorithm = alg, seed = 9, keep = :events)
        @test a.times == b.times && a.counts == b.counts && a.final_infection_counts == b.final_infection_counts
        @test [(e.time, e.transition_index, e.node) for e in a.events] == [(e.time, e.transition_index, e.node) for e in b.events]
    end
end

# ---------------------------------------------------------------------------------------------------------------
# M4: no premature stop while scheduled events are pending (N04 #5)
# ---------------------------------------------------------------------------------------------------------------

@testset "M4: zero total rate does not end a run with scheduled work pending" begin
    # 2-node SI, no edge until t = 1, rate 1: P(node 2 infected by t) = 1 − e^{−(t−1)}.
    tvn = TimeVaryingNetwork(SimpleGraph(2), [(t = 1.0, src = 1, dst = 2, action = :add)])
    spec = OutbreakSpec(model = si_model(1.0), network = tvn, initial = SeedNodes(:I => [1]), tspan = (0.0, 2.0))
    nrep = 2000
    for alg in ALGS_SCHED
        ens = simulate_ensemble(spec; nsims = nrep, seed = 1, algorithm = alg, keep = :events)
        p̂ = mean(state_at(t, 2.0)[2] == 2 for t in ens)
        @test within(p̂, 1 - exp(-1), nrep)                              # was 0.0
        @test all(e -> e.time > 1.0, reduce(vcat, [t.events for t in ens]))
    end
    # rate raised from 0 at t = 1
    spec0 = OutbreakSpec(model = si_model(0.0), network = path_graph(2), initial = SeedNodes(:I => [1]),
                         tspan = (0.0, 2.0))
    plan = InterventionPlan([ScheduledRateChange(1.0, :S, :I, :infection, 1.0)])
    for alg in ALGS_SCHED
        ens = simulate_ensemble(spec0; nsims = nrep, seed = 2, algorithm = alg, interventions = plan)
        @test within(mean(state_at(t, 2.0)[2] == 2 for t in ens), 1 - exp(-1), nrep)
    end
    # importation into an all-susceptible population
    specS = OutbreakSpec(model = sir_model(1.0, 1.0), network = random_regular_graph(500, 5; rng = StableRNG(2)),
                         initial = SeedFraction(:S => 1.0), tspan = (0.0, 30.0))
    imp = InterventionPlan([ScheduledStateChange(1.0, :I, 0.02; from = [:S])])
    for alg in ALGS_SCHED
        tr = simulate(specS; algorithm = alg, seed = 1, interventions = imp)
        @test state_at(tr, 30.0)[3] >= 10                                  # was 0: the run stopped at t = 0
        @test state_at(tr, 0.999) == [500, 0, 0]
    end
    # an infinite horizon runs to absorption (no BoundsError), with and without pending network updates
    specInf = OutbreakSpec(model = sir_model(0.5, 1.0), network = random_regular_graph(200, 4; rng = StableRNG(3)),
                           initial = SeedFraction(:I => 0.05), tspan = (0.0, Inf))
    for alg in ALGS
        tr = simulate(specInf; algorithm = alg, seed = 1)
        @test tr.counts[2, end] == 0 && isfinite(tr.times[end])
    end
    specInf2 = OutbreakSpec(model = si_model(1.0), network = tvn, initial = SeedNodes(:I => [1]), tspan = (0.0, Inf))
    for alg in ALGS_SCHED
        tr = simulate(specInf2; algorithm = alg, seed = 1)
        @test tr.counts[:, end] == [0, 2] && tr.times[end] > 1.0
    end
    # absorbing state with network updates still pending ends early and exactly (no catalysts left)
    late = TimeVaryingNetwork(path_graph(3), [(t = 50.0, src = 1, dst = 3, action = :add)])
    specL = OutbreakSpec(model = sir_model(0.0, 1.0), network = late, initial = SeedNodes(:I => [2]),
                         tspan = (0.0, 100.0))
    for alg in ALGS_SCHED
        tr = simulate(specL; algorithm = alg, seed = 1)
        @test tr.counts[:, end] == [2, 0, 1] && tr.times[end] == 100.0
    end
end

# ---------------------------------------------------------------------------------------------------------------
# M5: one merged, time-ordered queue of network updates and interventions (N04 #6)
# ---------------------------------------------------------------------------------------------------------------

@testset "M5: network updates and interventions are applied in time order" begin
    # Rate raised 0 → 2 at t = 1, edge 1–2 added at t = 5, horizon 5.5; a slow clock node keeps the rate positive.
    # Node 2 can only be infected after t = 5: P = 1 − e^{−2·0.5}.
    m = OutbreakModel([:S, :I, :X], [false, true, false],
        [OutbreakTransition(:S, :I, 0.0, :infection), OutbreakTransition(:X, :X, 0.01, :spontaneous)])
    tvn = TimeVaryingNetwork(SimpleGraph(3), [(t = 5.0, src = 1, dst = 2, action = :add)])
    plan = InterventionPlan([ScheduledRateChange(1.0, :S, :I, :infection, 2.0)])
    spec = OutbreakSpec(model = m, network = tvn, initial = SeedNodes(:I => [1], :X => [3]), tspan = (0.0, 5.5))
    nrep = 2000
    for alg in ALGS_SCHED
        ens = simulate_ensemble(spec; nsims = nrep, seed = 7, algorithm = alg, keep = :events, interventions = plan)
        early = count(t -> any(e -> e.node == 2 && e.time < 5.0, t.events), ens.trajectories)
        @test early == 0                                                   # was ≈ 96%
        @test within(mean(state_at(t, 5.5)[2] == 2 for t in ens), 1 - exp(-1), nrep)
    end
    # Pure ordering: edge present until t = 3, a no-op rate change at t = 1. Exact P = 1 − e^{−3}
    # (the bug gave (1 − e^{−1}) + e^{−1}(1 − e^{−2})² ≈ 0.907).
    tvn2 = TimeVaryingNetwork(path_graph(2), [(t = 3.0, src = 1, dst = 2, action = :remove)])
    spec2 = OutbreakSpec(model = si_model(1.0), network = tvn2, initial = SeedNodes(:I => [1]), tspan = (0.0, 10.0))
    noop = InterventionPlan([ScheduledRateChange(1.0, :S, :I, :infection, 1.0)])
    for alg in ALGS_SCHED
        ens = simulate_ensemble(spec2; nsims = 4000, seed = 8, algorithm = alg, interventions = noop)
        @test within(mean(state_at(t, 10.0)[2] == 2 for t in ens), 1 - exp(-3), 4000)
    end
    # Threshold interventions are checked at the start and after scheduled interventions (not only after events).
    g = random_regular_graph(200, 4; rng = StableRNG(4))
    mv = OutbreakModel([:S, :I, :R, :V], [false, true, false, false],
        [OutbreakTransition(:S, :I, 1.0, :infection), OutbreakTransition(:I, :R, 1.0, :spontaneous)])
    specV = OutbreakSpec(model = mv, network = g, initial = SeedFraction(:I => 0.05), tspan = (0.0, 10.0))
    at_start = InterventionPlan([ThresholdIntervention(:I, :above, 5, ScheduledStateChange(NaN, :V, 1.0; from = [:S]))])
    for alg in ALGS_SCHED
        tr = simulate(specV; algorithm = alg, seed = 1, interventions = at_start)
        @test tr.times[2] == 0.0 && tr.counts[:, 2] == [0, 10, 0, 190]    # fired at t = 0, before any event
        @test final_size(tr) == 10 / 200
        # curves are right-continuous: at t = 0 they show the state after the t = 0 intervention
        ens = simulate_ensemble(specV; nsims = 2, seed = 1, algorithm = alg, interventions = at_start)
        @test mean_curve(ens, :V; tgrid = [0.0])[2] == [190.0]
    end
    # ... and after a scheduled intervention, with no SSA event at all: the t = 1 pulse moves 5 of 9 S to V, which
    # crosses V ≥ 3, so the threshold moves the remaining 4 S to R at t = 1 (not never).
    m0 = OutbreakModel([:S, :I, :R, :V], [false, true, false, false], [OutbreakTransition(:S, :I, 0.0, :infection)])
    spec0 = OutbreakSpec(model = m0, network = SimpleGraph(10), initial = SeedNodes(:I => [1]), tspan = (0.0, 5.0))
    after_pulse = InterventionPlan([ScheduledStateChange(1.0, :V, 0.5; from = [:S]),
                                    ThresholdIntervention(:V, :above, 3, ScheduledStateChange(NaN, :R, 1.0; from = [:S]))])
    for alg in ALGS_SCHED
        tr = simulate(spec0; algorithm = alg, seed = 1, interventions = after_pulse)
        @test tr.times == [0.0, 1.0, 1.0, 5.0]
        @test tr.counts[:, 2] == [4, 1, 0, 5] && tr.counts[:, end] == [0, 1, 4, 5]
    end
    # TVN updates that refer to missing nodes are rejected
    bad = TimeVaryingNetwork(SimpleGraph(2), [(t = 1.0, src = 1, dst = 5, action = :add)])
    @test_throws ArgumentError simulate(OutbreakSpec(model = si_model(1.0), network = bad,
                                                     initial = SeedNodes(:I => [1]), tspan = (0.0, 2.0)))
    # interventions before the start time are rejected
    @test_throws ArgumentError simulate(specV; interventions = InterventionPlan([ScheduledStateChange(-1.0, :V, 0.1)]))
end

# ---------------------------------------------------------------------------------------------------------------
# M6: ScheduledStateChange(from = …) and importations counted (N04 #7, N05)
# ---------------------------------------------------------------------------------------------------------------

@testset "M6: ScheduledStateChange from filter and basis" begin
    m = OutbreakModel([:S, :I, :R, :V], [false, true, false, false],
        [OutbreakTransition(:S, :I, 0.5, :infection), OutbreakTransition(:I, :R, 1e-12, :spontaneous)])
    spec = OutbreakSpec(model = m, network = SimpleGraph(1000), initial = SeedFraction(:I => 0.5, :R => 0.2),
                        tspan = (0.0, 2.0))
    for alg in ALGS_SCHED
        c = state_at(simulate(spec; algorithm = alg, seed = 1,
                              interventions = InterventionPlan([ScheduledStateChange(1.0, :V, 0.3; from = [:S])])), 1.5)
        @test c == [0, 500, 200, 300]                                      # only S nodes are vaccinated
        c2 = state_at(simulate(spec; algorithm = alg, seed = 1,
                               interventions = InterventionPlan([ScheduledStateChange(1.0, :V, 0.5; from = [:S],
                                                                                      basis = :eligible)])), 1.5)
        @test c2 == [150, 500, 200, 150]
        # the default (no `from`) keeps the 0.1 behaviour: 300 nodes drawn from every other compartment
        c3 = state_at(simulate(spec; algorithm = alg, seed = 1,
                               interventions = InterventionPlan([ScheduledStateChange(1.0, :V, 0.3)])), 1.5)
        @test c3[4] == 300 && c3[2] < 500 && c3[3] < 200
    end
    # an importation into an infectious compartment counts in final_size (was 0)
    spec3 = OutbreakSpec(model = m, network = SimpleGraph(1000), initial = SeedFraction(:S => 1.0), tspan = (0.0, 2.0))
    for alg in ALGS_SCHED
        tr = simulate(spec3; algorithm = alg, seed = 1,
                      interventions = InterventionPlan([ScheduledStateChange(1.0, :I, 0.05; from = [:S])]))
        @test final_size(tr) == 0.05
        @test reinfection_histogram(tr; L = 1) == [950, 50]
    end
    @test_throws ArgumentError ScheduledStateChange(1.0, :V, 1.5)
    @test_throws ArgumentError ScheduledStateChange(1.0, :V, 0.5; basis = :nodes)
    @test_throws ArgumentError simulate(spec; interventions = InterventionPlan([ScheduledStateChange(1.0, :V, 0.3; from = [:Z])]))
    @test_throws ArgumentError simulate(spec; interventions = InterventionPlan([ScheduledStateChange(1.0, :Z, 0.3)]))
end

# ---------------------------------------------------------------------------------------------------------------
# m3: directed graphs (and self-loops) are rejected (N05)
# ---------------------------------------------------------------------------------------------------------------

@testset "m3: directed contact graphs are rejected" begin
    dg = SimpleDiGraph(50)
    for v in 1:49
        add_edge!(dg, v, v + 1)
    end
    spec = OutbreakSpec(model = si_model(1.0), network = dg, initial = SeedNodes(:I => [1]), tspan = (0.0, 100.0))
    for alg in ALGS
        @test_throws ArgumentError simulate(spec; algorithm = alg, seed = 1)
    end
    tvn = TimeVaryingNetwork(dg, [(t = 1.0, src = 1, dst = 3, action = :add)])
    for alg in ALGS_SCHED
        @test_throws ArgumentError simulate(OutbreakSpec(model = si_model(1.0), network = tvn,
                                                         initial = SeedNodes(:I => [1]), tspan = (0.0, 5.0)); algorithm = alg)
    end
    mpx = MultiplexGraph([SimpleDiGraph(path_graph(50)), SimpleDiGraph(path_graph(50))], [1.0, 1.0])
    for alg in (DirectSSA(), NextReaction(), HAS())                # the samplers that accept a MultiplexGraph
        @test_throws ArgumentError simulate(OutbreakSpec(model = si_model(1.0), network = mpx,
                                                         initial = SeedNodes(:I => [1]), tspan = (0.0, 5.0));
                                            algorithm = alg, seed = 1)
    end
    loop = path_graph(4); add_edge!(loop, 2, 2)
    @test_throws ArgumentError simulate(OutbreakSpec(model = si_model(1.0), network = loop,
                                                     initial = SeedNodes(:I => [1]), tspan = (0.0, 5.0)))
    # the undirected version of the same path is fine: the whole path is infected
    und = SimpleGraph(dg)
    for alg in ALGS
        tr = simulate(OutbreakSpec(model = si_model(1.0), network = und, initial = SeedNodes(:I => [50]),
                                   tspan = (0.0, 1e3)); algorithm = alg, seed = 1)
        @test tr.counts[2, end] == 50
    end
end

# ---------------------------------------------------------------------------------------------------------------
# m5: final_size counts every node that was ever infected, including latent seeds (N05, E28)
# ---------------------------------------------------------------------------------------------------------------

@testset "m5: final_size counts latent seeds and ignores vaccination and tracing" begin
    seir = OutbreakModel([:S, :E, :I, :R], [false, false, true, false],
        [OutbreakTransition(:S, :E, 0.5, :infection), OutbreakTransition(:E, :I, 0.01, :spontaneous),
         OutbreakTransition(:I, :R, 0.25, :spontaneous)])
    spec = OutbreakSpec(model = seir, network = SimpleGraph(1000), initial = SeedFraction(:E => 0.1),
                        tspan = (0.0, 1.0))
    for alg in ALGS
        tr = simulate(spec; algorithm = alg, seed = 1)
        @test final_size(tr) == 0.1                                         # was 1 − S/N only after E → I (≈ 0.001)
        @test NO._initially_infected(tr) == 100
        @test reinfection_histogram(tr) == [900, 100]
    end
    # E is infected, R is not: an E → I → R history counts once
    @test NO._infected_mask(seir) == BitVector([false, true, true, false])
    # a full SEIR epidemic with E seeds: final size = 1 − S(∞)/N when every node that left S was infected
    specE = OutbreakSpec(model = OutbreakModel([:S, :E, :I, :R], [false, false, true, false],
                             [OutbreakTransition(:S, :E, 0.5, :infection), OutbreakTransition(:E, :I, 0.5, :spontaneous),
                              OutbreakTransition(:I, :R, 0.5, :spontaneous)]),
                         network = random_regular_graph(400, 4; rng = StableRNG(9)),
                         initial = SeedFraction(:E => 0.05), tspan = (0.0, 200.0))
    for alg in ALGS
        tr = simulate(specE; algorithm = alg, seed = 3)
        @test final_size(tr) ≈ 1 - tr.counts[1, end] / 400
        @test final_size(tr; recovered = :R) == tr.counts[4, end] / 400
    end
    # vaccination S → V (node-local exit) is not an infection
    sirv = OutbreakModel([:S, :I, :R, :V], [false, true, false, false],
        [OutbreakTransition(:S, :I, 0.5, :infection), OutbreakTransition(:I, :R, 1.0, :spontaneous),
         OutbreakTransition(:S, :V, 1.0, :spontaneous)])
    specV = OutbreakSpec(model = sirv, network = SimpleGraph(200), initial = SeedFraction(:I => 0.05),
                         tspan = (0.0, 20.0))
    for alg in ALGS
        tr = simulate(specV; algorithm = alg, seed = 2)
        @test tr.counts[4, end] > 150
        @test final_size(tr) == 0.05
    end
    # a traced node (S → Q by :contact_trace) is not infected
    sirq = OutbreakModel([:S, :I, :R, :Q], [false, true, false, false],
        [OutbreakTransition(:S, :Q, 1.0, :contact_trace; via = [:I])])
    @test NO._infected_mask(sirq) == BitVector([false, true, false, false])
    for alg in ALGS
        tr = simulate(OutbreakSpec(model = sirq, network = path_graph(2), initial = SeedNodes(:I => [1]),
                                   tspan = (0.0, 100.0)); algorithm = alg, seed = 1)
        @test tr.counts[4, end] == 1 && final_size(tr) == 0.5
    end
    # Tracing of latent contacts (E → Q by :contact_trace): E is a contact recipient and still infected, so E seeds
    # count at t0 and every S → E infection counts, also when the node is traced before it becomes infectious
    # (the first mask excluded every contact recipient: {I} only, final_size 0.258 instead of 0.387 on RRG(2000, 5)).
    seirq = OutbreakModel([:S, :E, :I, :R, :Q], [false, false, true, false, false],
        [OutbreakTransition(:S, :E, 0.4, :infection; via = [:I]), OutbreakTransition(:E, :I, 0.2, :spontaneous),
         OutbreakTransition(:I, :R, 0.25, :spontaneous),
         OutbreakTransition(:S, :Q, 0.2, :contact_trace; via = [:I]),
         OutbreakTransition(:E, :Q, 0.2, :contact_trace; via = [:I])])
    @test NO._infected_mask(seirq) == BitVector([false, true, true, false, false])
    specQ = OutbreakSpec(model = seirq, network = random_regular_graph(1000, 5; rng = StableRNG(1)),
                         initial = SeedFraction(:E => 0.02), tspan = (0.0, 300.0))
    for alg in ALGS
        tr = simulate(specQ; algorithm = alg, seed = 1, keep = :events)
        @test tr.counts[2, 1] == 20 && NO._initially_infected(tr) == 20
        nSE = count(e -> e.transition_index == 1, tr.events)
        traced_latent = [e.node for e in tr.events if e.transition_index == 5]
        @test nSE > 100 && !isempty(traced_latent)
        @test final_size(tr) == (20 + nSE) / 1000
        @test reinfection_histogram(tr) == [1000 - 20 - nSE, 20 + nSE]
        @test all(tr.final_infection_counts[v] == 1 for v in traced_latent)
        tr0 = simulate(OutbreakSpec(model = seirq, network = SimpleGraph(100), initial = SeedFraction(:E => 0.1),
                                    tspan = (0.0, 1.0)); algorithm = alg, seed = 1)
        @test final_size(tr0) == 0.1                                        # was 0.0
    end
    # Tracing of infectious nodes (I → Q via diagnosed D): I is a contact recipient, E is still infected
    # (the first mask dropped E, so 10% E seeds gave final_size 0.0).
    seidrq = OutbreakModel([:S, :E, :I, :D, :R, :Q], [false, false, true, true, false, false],
        [OutbreakTransition(:S, :E, 0.4, :infection; via = [:I, :D]), OutbreakTransition(:E, :I, 0.2, :spontaneous),
         OutbreakTransition(:I, :D, 0.1, :spontaneous), OutbreakTransition(:I, :R, 0.2, :spontaneous),
         OutbreakTransition(:D, :R, 0.5, :spontaneous),
         OutbreakTransition(:I, :Q, 0.3, :contact_trace; via = [:D])])
    @test NO._infected_mask(seidrq) == BitVector([false, true, true, true, false, false])
    specD = OutbreakSpec(model = seidrq, network = random_regular_graph(1000, 5; rng = StableRNG(2)),
                         initial = SeedFraction(:E => 0.02), tspan = (0.0, 300.0))
    for alg in ALGS
        tr0 = simulate(OutbreakSpec(model = seidrq, network = SimpleGraph(100), initial = SeedFraction(:E => 0.1),
                                    tspan = (0.0, 0.001)); algorithm = alg, seed = 1)
        @test final_size(tr0) == 0.1                                        # was 0.0
        tr = simulate(specD; algorithm = alg, seed = 1, keep = :events)
        nSE = count(e -> e.transition_index == 1, tr.events)
        @test nSE > 100 && any(e -> e.transition_index == 6, tr.events)
        @test final_size(tr) == (20 + nSE) / 1000
    end
    # Superinfection I1 → I2 (via I2): I1 is a contact recipient, E1 is still infected (the first mask dropped E1, so
    # E1 seeds gave final_size 0.0); I1 → I2 is not a new infection.
    superinf = OutbreakModel([:S, :E1, :I1, :I2, :R], [false, false, true, true, false],
        [OutbreakTransition(:S, :E1, 0.5, :infection; via = [:I1]), OutbreakTransition(:E1, :I1, 1.0, :spontaneous),
         OutbreakTransition(:I1, :R, 0.25, :spontaneous), OutbreakTransition(:I1, :I2, 2.0, :infection; via = [:I2]),
         OutbreakTransition(:I2, :R, 0.5, :spontaneous)])
    @test NO._infected_mask(superinf) == BitVector([false, true, true, true, false])
    specS = OutbreakSpec(model = superinf, network = random_regular_graph(500, 6; rng = StableRNG(3)),
                         initial = SeedNodes(:E1 => 1:10, :I2 => 11:60), tspan = (0.0, 100.0))
    for alg in ALGS
        tr0 = simulate(OutbreakSpec(model = superinf, network = SimpleGraph(100), initial = SeedNodes(:E1 => 1:10),
                                    tspan = (0.0, 0.001)); algorithm = alg, seed = 1)
        @test final_size(tr0) == 0.1                                        # was 0.0
        tr = simulate(specS; algorithm = alg, seed = 2, keep = :events)
        @test NO._initially_infected(tr) == 60
        nSE = count(e -> e.transition_index == 1, tr.events)
        @test nSE > 100 && count(e -> e.transition_index == 4, tr.events) > 100
        @test final_size(tr) == (60 + nSE) / 500
        @test length(reinfection_histogram(tr)) == 2                      # nobody is counted twice
    end
    # Quarantine of exposed contacts that still progress (E → Eq by :contact_trace, Eq → Iq): tracing preserves
    # infection status, so E and Eq are infected and a traced node is counted once (the reviewer's candidate rule,
    # which let tracing contacts into Eq make E a susceptible class, dropped E).
    quar = OutbreakModel([:S, :E, :I, :R, :Eq, :Iq], [false, false, true, false, false, true],
        [OutbreakTransition(:S, :E, 0.4, :infection; via = [:I]), OutbreakTransition(:S, :E, 0.05, :infection; via = [:Iq]),
         OutbreakTransition(:E, :I, 0.2, :spontaneous), OutbreakTransition(:I, :R, 0.25, :spontaneous),
         OutbreakTransition(:E, :Eq, 0.3, :contact_trace; via = [:I]), OutbreakTransition(:Eq, :Iq, 0.2, :spontaneous),
         OutbreakTransition(:Iq, :R, 0.5, :spontaneous)])
    @test NO._infected_mask(quar) == BitVector([false, true, true, false, true, true])
    specq = OutbreakSpec(model = quar, network = random_regular_graph(1000, 5; rng = StableRNG(4)),
                         initial = SeedFraction(:E => 0.02), tspan = (0.0, 300.0))
    for alg in ALGS
        tr = simulate(specq; algorithm = alg, seed = 1, keep = :events)
        @test NO._initially_infected(tr) == 20
        nSE = count(e -> e.transition_index in (1, 2), tr.events)
        @test nSE > 100 && count(e -> e.transition_index == 5, tr.events) > 10
        @test final_size(tr) == (20 + nSE) / 1000
        @test length(reinfection_histogram(tr)) == 2                      # nobody is counted twice
    end
    # Aware susceptibles protected from contact infection but open to importation (S → Sa via Sa, Sa → E): the
    # awareness contact is not an infection (no infectious catalyst), so Sa is not infected and an importation into
    # E counts (the candidate rule counted Sa as infected).
    aware = OutbreakModel([:S, :Sa, :E, :I, :R], [false, false, false, true, false],
        [OutbreakTransition(:S, :Sa, 1.0, :infection; via = [:Sa]), OutbreakTransition(:S, :E, 0.5, :infection; via = [:I]),
         OutbreakTransition(:Sa, :E, 0.01, :spontaneous), OutbreakTransition(:E, :I, 1.0, :spontaneous),
         OutbreakTransition(:I, :R, 0.5, :spontaneous)])
    @test NO._infected_mask(aware) == BitVector([false, false, true, true, false])
    speca = OutbreakSpec(model = aware, network = random_regular_graph(500, 4; rng = StableRNG(5)),
                         initial = SeedNodes(:I => 1:5, :Sa => 6:30), tspan = (0.0, 100.0))
    for alg in ALGS
        tr = simulate(speca; algorithm = alg, seed = 1, keep = :events)
        @test NO._initially_infected(tr) == 5
        nSE = count(e -> e.transition_index == 2, tr.events)
        nimport = count(e -> e.transition_index == 3, tr.events)
        @test nimport > 10 && tr.counts[2, end] > 10                      # some aware nodes never imported
        @test final_size(tr) == (5 + nSE + nimport) / 500
    end
    # masks of further models: those without infected contact recipients are the same as under the first rule
    Tr = OutbreakTransition
    for (m, mask) in [
            OutbreakModel([:S, :I, :R], [false, true, false],
                [Tr(:S, :I, 1, :infection), Tr(:I, :R, 1, :spontaneous), Tr(:R, :S, 1, :spontaneous)]) => [0, 1, 0],
            OutbreakModel([:S, :E, :A, :I, :R], [false, false, true, true, false],
                [Tr(:S, :E, 1, :infection; via = [:I]), Tr(:S, :E, 1, :infection; via = [:A]),
                 Tr(:E, :A, 1, :spontaneous), Tr(:E, :I, 1, :spontaneous), Tr(:A, :R, 1, :spontaneous),
                 Tr(:I, :R, 1, :spontaneous)]) => [0, 1, 1, 1, 0],
            # awareness spreading S → A via A is not an infection
            OutbreakModel([:S, :I, :R, :A], [false, true, false, false],
                [Tr(:S, :I, 1, :infection; via = [:I]), Tr(:S, :A, 1, :infection; via = [:A]),
                 Tr(:I, :R, 1, :spontaneous)]) => [0, 1, 0, 0],
            # aware susceptibles Sa that can also be infected by contact stay susceptible
            OutbreakModel([:S, :Sa, :E, :I, :R], [false, false, false, true, false],
                [Tr(:S, :Sa, 1, :infection; via = [:Sa]), Tr(:S, :E, 1, :infection; via = [:I]),
                 Tr(:Sa, :E, 0.5, :infection; via = [:I]), Tr(:Sa, :E, 0.01, :spontaneous),
                 Tr(:E, :I, 1, :spontaneous), Tr(:I, :R, 1, :spontaneous)]) => [0, 0, 1, 1, 0],
            # ... also when the awareness is caught from infectious neighbours (S → Sa via I)
            OutbreakModel([:S, :Sa, :E, :I, :R], [false, false, false, true, false],
                [Tr(:S, :Sa, 1, :infection; via = [:I]), Tr(:S, :E, 1, :infection; via = [:I]),
                 Tr(:Sa, :E, 0.2, :infection; via = [:I]), Tr(:Sa, :E, 0.01, :spontaneous),
                 Tr(:E, :I, 1, :spontaneous), Tr(:I, :R, 1, :spontaneous)]) => [0, 0, 1, 1, 0],
            # ring vaccination written as an :infection into V is not an infection (V does not lead to I)
            OutbreakModel([:S, :I, :R, :V], [false, true, false, false],
                [Tr(:S, :I, 1, :infection), Tr(:S, :V, 1, :infection; via = [:I]),
                 Tr(:I, :R, 1, :spontaneous)]) => [0, 1, 0, 0],
            # a superinfected-then-relapsing strain: the latent relapse phase L after the infectious recipient I1
            OutbreakModel([:S, :E1, :I1, :I2, :L, :R], [false, false, true, true, false, false],
                [Tr(:S, :E1, 1, :infection; via = [:I1]), Tr(:E1, :I1, 1, :spontaneous),
                 Tr(:I1, :I2, 1, :infection; via = [:I2]), Tr(:I1, :L, 1, :spontaneous), Tr(:L, :I1, 1, :spontaneous),
                 Tr(:I2, :R, 1, :spontaneous)]) => [0, 1, 1, 1, 1, 0]]
        @test NO._infected_mask(m) == BitVector(mask)
    end
    # SIS: reinfections are counted per infection, final_size per node
    sis = OutbreakModel([:S, :I], [false, true],
        [OutbreakTransition(:S, :I, 1.0, :infection), OutbreakTransition(:I, :S, 1.0, :spontaneous)])
    trS = simulate(OutbreakSpec(model = sis, network = random_regular_graph(200, 4; rng = StableRNG(1)),
                                initial = SeedFraction(:I => 0.1), tspan = (0.0, 20.0)); seed = 1)
    h = reinfection_histogram(trS)
    @test sum(h) == 200 && length(h) > 2
    @test final_size(trS) ≈ 1 - h[1] / 200
    # `recovered` is honoured and validated
    tr = simulate(specE; seed = 3)
    @test final_size(tr; recovered = [:R, :I]) == (tr.counts[3, end] + tr.counts[4, end]) / 400
    @test_throws ArgumentError final_size(tr; recovered = :nonexistent)
    @test_throws ArgumentError final_size(tr; recovered = Symbol[])
    ens = simulate_ensemble(specE; nsims = 3, seed = 1)
    @test final_size(ens) == [final_size(t) for t in ens]
end

# ---------------------------------------------------------------------------------------------------------------
# CompositionRejection: hazards spanning more than 2^63 (bucket index clamped to the top bucket)
# ---------------------------------------------------------------------------------------------------------------

@testset "CR: the clamped top bucket is sampled exactly" begin
    # Hazards 1e-30, 1 and 2: the last two exceed the ceiling of bucket 64 (1e-30 · 2^64 ≈ 1.8e-11). The first event
    # is B → D with probability 2/3 exactly; rejection sampling in that bucket accepted every proposal and gave 1/2.
    m = OutbreakModel([:X, :A, :B, :D], [false, false, false, false],
        [OutbreakTransition(:X, :X, 1e-30, :spontaneous), OutbreakTransition(:A, :D, 1.0, :spontaneous),
         OutbreakTransition(:B, :D, 2.0, :spontaneous)])
    spec = OutbreakSpec(model = m, network = SimpleGraph(3), initial = SeedNodes(:X => [1], :A => [2], :B => [3]),
                        tspan = (0.0, 100.0))
    nrep = 4000
    for alg in ALGS
        ens = simulate_ensemble(spec; nsims = nrep, seed = 1, algorithm = alg, keep = :events)
        @test within(mean(t -> t.events[1].transition_index == 3, ens.trajectories), 2 / 3, nrep)
    end
end

# ---------------------------------------------------------------------------------------------------------------
# Acceptance: χ² agreement of event types for a two-infector model, across all four algorithms
# ---------------------------------------------------------------------------------------------------------------

@testset "χ²: event-type frequencies agree across DirectSSA/NR/CR/HAS (two infectors)" begin
    # S → E via I (τ₁) and via A (τ₂); E → A | I (branching); A, I → R; plus tracing S → Q via I.
    # Seven event types, so the χ² tests have even degrees of freedom (6 and 18).
    τ1, τ2, α, σ, p, γ = 0.3, 0.15, 0.1, 0.5, 0.4, 0.25
    comps = [:S, :E, :A, :I, :R, :Q]; inf = [false, false, true, true, false, false]
    trs = [OutbreakTransition(:S, :E, τ1, :infection; via = [:I]),
           OutbreakTransition(:S, :E, τ2, :infection; via = [:A]),
           OutbreakTransition(:S, :Q, α, :contact_trace; via = [:I]),
           OutbreakTransition(:E, :A, p * σ, :spontaneous),
           OutbreakTransition(:E, :I, (1 - p) * σ, :spontaneous),
           OutbreakTransition(:A, :R, γ, :spontaneous),
           OutbreakTransition(:I, :R, γ, :spontaneous)]
    model = OutbreakModel(comps, inf, trs; name = :SEAIRQ)
    K = length(trs)

    # (i) First event against exact probabilities (hazard of each type at t = 0, enumerated independently).
    g = random_regular_graph(200, 4; rng = StableRNG(5))
    I0 = collect(1:10); A0 = collect(11:20); E0 = collect(21:30)
    seedspec = SeedNodes(:I => I0, :A => A0, :E => E0; default = :S)
    st = fill(:S, 200); st[I0] .= :I; st[A0] .= :A; st[E0] .= :E
    nI = sum(count(u -> st[u] == :I, neighbors(g, v)) for v in 1:200 if st[v] == :S)
    nA = sum(count(u -> st[u] == :A, neighbors(g, v)) for v in 1:200 if st[v] == :S)
    haz = [τ1 * nI, τ2 * nA, α * nI, 10p * σ, 10(1 - p) * σ, 10γ, 10γ]
    pexact = haz ./ sum(haz)
    spec1 = OutbreakSpec(model = model, network = g, initial = seedspec, tspan = (0.0, 0.25))
    nrun = 10_000
    for (a, alg) in enumerate(ALGS)
        ens = simulate_ensemble(spec1; nsims = nrun, seed = 1000 + a, algorithm = alg, keep = :events)
        firsts = [t.events[1].transition_index for t in ens if !isempty(t.events)]
        @test length(firsts) >= 0.99nrun
        obs = [count(==(k), firsts) for k in 1:K]
        x, pval = chisq_gof(obs, pexact)
        PVALUES["chi2 first-event GOF $(algname(alg))"] = pval
        @test pval > 1e-3
    end

    # (ii) Whole epidemics: one uniformly chosen event per run (independent draws, as the χ² test requires),
    # 10⁴ runs per algorithm, homogeneity across the four algorithms.
    g2 = random_regular_graph(100, 4; rng = StableRNG(6))
    spec2 = OutbreakSpec(model = model, network = g2, initial = SeedFraction(:I => 0.05, :A => 0.05),
                         tspan = (0.0, 8.0))
    table = zeros(Int, length(ALGS), K)
    pick = StableRNG(77)
    for (a, alg) in enumerate(ALGS)
        ens = simulate_ensemble(spec2; nsims = nrun, seed = 2000 + a, algorithm = alg, keep = :events)
        for t in ens
            isempty(t.events) && continue
            table[a, t.events[rand(pick, 1:length(t.events))].transition_index] += 1
        end
    end
    @test all(>=(0.99nrun), sum(table; dims = 2))
    @test all(>(0), table)
    x, pval = chisq_homogeneity(table)
    PVALUES["chi2 homogeneity (4 algorithms x 7 types)"] = pval
    @test pval > 1e-3
end

# ---------------------------------------------------------------------------------------------------------------
# Acceptance: KS test on the SIR final size at N = 2000 across the four algorithms
# ---------------------------------------------------------------------------------------------------------------

@testset "KS: SIR final-size distributions agree across algorithms (N = 2000)" begin
    # :sir_reg6 anchors: τ = 1/6, γ = 1/4 on a 6-regular graph, 1% seeds. EBCM R∞ (incl. seeds) = 0.929510.
    N = 2000
    g = random_regular_graph(N, 6; rng = StableRNG(2026))
    spec = OutbreakSpec(model = sir_model(1 / 6, 1 / 4), network = g, initial = SeedFraction(:I => 0.01),
                        tspan = (0.0, 500.0))
    nr = 200
    fs = Dict(algname(alg) => final_size(simulate_ensemble(spec; nsims = nr, seed = 3000 + a, algorithm = alg,
                                                           parallel = true))
              for (a, alg) in enumerate(ALGS))
    names_ = collect(keys(fs))
    for i in eachindex(names_), j in (i + 1):length(names_)
        D, pval = ks_2sample(fs[names_[i]], fs[names_[j]])
        PVALUES["KS final size $(names_[i]) vs $(names_[j])"] = pval
        @test pval > 1e-3
    end
    for (k, v) in fs
        @test abs(mean(v) - 0.929510) < 0.01          # large-N limit (finite-size and graph effects ≪ 0.01)
        @test minimum(v) > 0.5                        # all runs major with 20 seeds
    end
end
