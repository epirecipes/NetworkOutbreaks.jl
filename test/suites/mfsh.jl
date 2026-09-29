# Tests for WP36c (DESIGN_NetworkEpiCore.md §K WP36c, §C.3, §D.5 Λ3, §J.7): the fleeting-contact process of
# mean-field social heterogeneity (NetworkEpiCore `MFSHNetwork`; Miller–Slim–Volz Part II §3.2.2), i.e.
# `FleetingContacts`, `sample_graph(::MFSHNetwork)`, `FleetingContactSSA` and `simulate(model, ::MFSHNetwork)`.
#
# Every expected value comes from an independent source:
#   - the hazard law τ·k_v·Σ_{c ∈ via}(K_c − [c = s]k_v)/(M − k_v), computed by hand on four- and two-node populations
#     (the target and the waiting time of the first events, 20 000 runs each);
#   - MassActionSSA on WellMixed(κ): on a κ-regular population the two processes are the same Markov chain, and the
#     samplers agree event by event from the same seed (also under interventions);
#   - the MFSH ODE of MSV Part II §3.2.2, θ̇ = −τθ + τqθ²ψ'(θ)/ψ'(1) − γθ ln θ, Ṙ = γ(1 − qψ(θ) − R), S = qψ(θ),
#     integrated by an RK4 written in this file, and NetworkEpiCore's MFSH final-size equation (the scenarios'
#     expected values), for the law of large numbers on :sir_mfsh_pois5 and :sir_mfsh_msv.
#
# Every ensemble is seeded (NetworkOutbreaks.stable_rng streams), so the suite is deterministic; the z-scores are
# recorded in RESULTS for reporting.

using NetworkOutbreaks
import NetworkOutbreaks as NO
using Graphs
using Test
using StableRNGs
using Statistics

const RESULTS = Dict{String, Any}()

sir(τ, γ) = OutbreakModel([:S, :I, :R], [false, true, false],
    [OutbreakTransition(:S, :I, τ, :infection), OutbreakTransition(:I, :R, γ, :spontaneous)]; name = :SIR)

# Same trajectory: counts, per-node infection counts and the event log; times to rounding (the two samplers compute
# the same propensities in a different order of floating-point operations).
function same_path(a::OutbreakTrajectory, b::OutbreakTrajectory)
    a.counts == b.counts || return false
    a.final_infection_counts == b.final_infection_counts || return false
    length(a.times) == length(b.times) || return false
    maximum(abs.(a.times .- b.times); init = 0.0) <= 1e-9 * max(1.0, maximum(abs, a.times)) || return false
    return [(e.transition_index, e.node) for e in events(a)] == [(e.transition_index, e.node) for e in events(b)]
end

# The MFSH SIR ODE (MSV Part II §3.2.2, explicit seed q = 1 − ρ), by RK4 with step h, sampled every `every` steps:
# columns S, I, R.
function mfsh_rk4(d, τ, γ, ρ, T; h = 1e-3, every = 100)
    q = 1 - ρ
    k̄ = mean_degree(d)
    f(u) = (-τ * u[1] + τ * q * u[1]^2 * pgf_derivative(d, u[1], 1) / k̄ - γ * u[1] * log(u[1]),
            γ * (1 - q * pgf(d, u[1]) - u[2]))
    u = (1.0, 0.0)
    out = [u]
    for n in 1:round(Int, T / h)
        k1 = f(u); k2 = f(u .+ h / 2 .* k1); k3 = f(u .+ h / 2 .* k2); k4 = f(u .+ h .* k3)
        u = u .+ h / 6 .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
        n % every == 0 && push!(out, u)
    end
    S = [q * pgf(d, x[1]) for x in out]
    R = [x[2] for x in out]
    return hcat(S, 1 .- S .- R, R)
end

# ---------------------------------------------------------------------------------------------------------------
# The contact structure
# ---------------------------------------------------------------------------------------------------------------

@testset "FleetingContacts and sample_graph(::MFSHNetwork)" begin
    k = [3, 0, 5, 1]
    fc = FleetingContacts(k)
    @test nv(fc) == 4 && fc.degrees == k
    k[1] = 100
    @test fc.degrees[1] == 3                                    # the vector is copied
    @test fc isa AbstractContactNetwork
    @test sprint(show, fc) == "FleetingContacts(4 nodes, 9 stubs, mean degree 2.25)"
    @test_throws ArgumentError FleetingContacts(Int[])
    @test_throws ArgumentError FleetingContacts([2, -1])
    @test ne(fc) == 0                                           # no persistent edges (as GraphInfo.edges)

    net = MFSHNetwork(PoissonDegree(5))
    g, info = sample_graph(net, 20_000; rng = NO.stable_rng(1))
    @test g isa FleetingContacts && nv(g) == 20_000
    @test first(sample_graph(net, 20_000; rng = NO.stable_rng(1))).degrees == g.degrees     # reproducible
    @test first(sample_graph(net, 20_000; rng = NO.stable_rng(2))).degrees != g.degrees
    @test info.method === :fleeting && info.N == 20_000 && info.edges == 0
    @test info.erased == 0 && info.candidate_edges == 0 && info.erased_fraction == 0
    @test info.stubs == sum(g.degrees) && info.max_degree == maximum(g.degrees)
    @test info.mean_degree ≈ mean(g.degrees)
    @test info.excess_degree ≈ sum(x -> x * (x - 1), g.degrees) / sum(g.degrees)
    @test abs(info.mean_degree - 5) < 4 * sqrt(5 / 20_000)                   # Poisson(5): sd of the mean
    @test abs(info.excess_degree - 5) < 0.1                                 # κ_ex = μ for Poisson
    g2, _ = sample_graph(MFSHNetwork(EmpiricalDegree(2 => 0.5, 8 => 0.5)), 1000; rng = NO.stable_rng(3))
    @test Set(g2.degrees) == Set([2, 8])                                    # an odd degree sum is allowed
    @test_throws ArgumentError sample_graph(net, 0)
end

# ---------------------------------------------------------------------------------------------------------------
# The hazard law
# ---------------------------------------------------------------------------------------------------------------

@testset "FleetingContactSSA: the hazard τ·k_v·(K_I − [v ∈ I]k_v)/(M − k_v), by hand" begin
    # Four nodes with 1, 2, 3 and 0 stubs (M = 6); node 1 is infectious, τ = 1, no recovery. Node 2 is hit at rate
    # 2·1/(6 − 2) = 1/2 and node 3 at 3·1/(6 − 3) = 1, node 4 never: the first infection comes at rate 3/2 and hits
    # node 2 with probability 1/3. (Partners among all M stubs, or M − 1, would give 2/5.) Then the other one is hit
    # at rate 2·4/4 = 2 (after node 3) or 3·3/3 = 3 (after node 2).
    spec = OutbreakSpec(sir(1.0, 0.0), FleetingContacts([1, 2, 3, 0]), SeedNodes(:I => [1]), (0.0, Inf))
    n = 20_000
    first_node = Int[]; first_time = Float64[]; second_gap = Dict(2 => Float64[], 3 => Float64[])
    nevents = Int[]; node4 = Int[]
    for s in 1:n
        tr = simulate(spec; algorithm = FleetingContactSSA(), seed = 360_000 + s, keep = :events)
        ev = events(tr)
        push!(nevents, length(ev)); push!(node4, tr.final_infection_counts[4])
        push!(first_node, ev[1].node); push!(first_time, ev[1].time)
        length(ev) >= 2 && push!(second_gap[ev[1].node], ev[2].time - ev[1].time)
    end
    @test all(==(2), nevents)                                   # nodes 2 and 3, then no hazard is left
    @test all(==(0), node4)                                     # no stubs: never infected
    p2 = count(==(2), first_node) / n
    se_p = sqrt(1 / 3 * 2 / 3 / n)
    RESULTS["hazard: P(first = node 2)"] = (p = p2, ref = 1 / 3, z = (p2 - 1 / 3) / se_p)
    @test abs(p2 - 1 / 3) < 4 * se_p
    @test abs(mean(first_time) - 2 / 3) < 4 * (2 / 3) / sqrt(n)
    @test abs(mean(second_gap[3]) - 1 / 2) < 4 * (1 / 2) / sqrt(length(second_gap[3]))
    @test abs(mean(second_gap[2]) - 1 / 3) < 4 * (1 / 3) / sqrt(length(second_gap[2]))

    # A catalyst in the recipient's own compartment (I → J via I): a node's own stubs are not its partners. Two nodes
    # with 2 and 1 stubs, both in I: node 1 is hit at 2·(3 − 2)/(3 − 2) = 2, node 2 at 1·(3 − 1)/(3 − 1) = 1.
    m = OutbreakModel([:I, :J], [true, false], [OutbreakTransition(:I, :J, 1.0, :contact_trace; via = [:I])]; name = :self)
    spec2 = OutbreakSpec(m, FleetingContacts([2, 1]), SeedCount(:I => 2), (0.0, Inf))
    nodes = Int[]; times_ = Float64[]; nevents2 = Int[]
    for s in 1:n
        ev = events(simulate(spec2; algorithm = FleetingContactSSA(), seed = 370_000 + s, keep = :events))
        push!(nodes, ev[1].node); push!(times_, ev[1].time); push!(nevents2, length(ev))
    end
    @test all(==(1), nevents2)                                  # the survivor has no partner left in I
    p1 = count(==(1), nodes) / n
    RESULTS["hazard: self-catalyst P(first = node 1)"] = (p = p1, ref = 2 / 3, z = (p1 - 2 / 3) / sqrt(2 / 9 / n))
    @test abs(p1 - 2 / 3) < 4 * sqrt(2 / 9 / n)
    @test abs(mean(times_) - 1 / 3) < 4 * (1 / 3) / sqrt(n)
    @info "WP36c hazard law (20 000 runs each)" P_first_node2 = p2 z = RESULTS["hazard: P(first = node 2)"].z mean_first_time = mean(first_time) self_P_first_node1 = p1 z_self = RESULTS["hazard: self-catalyst P(first = node 1)"].z self_mean_first_time = mean(times_)
end

# ---------------------------------------------------------------------------------------------------------------
# κ-regular populations are well mixed
# ---------------------------------------------------------------------------------------------------------------

@testset "RegularDegree(κ): the same process as MassActionSSA on WellMixed(κ), event by event" begin
    p = Dict(:τ => 0.2, :γ => 0.25, :σ => 0.3, :ν => 0.05)
    models = ((sir_model(), SeedFraction(:I => 0.1)), (seir_model(), SeedFraction(:E => 0.1)),
              (sirv_model(), SeedFraction(:I => 0.1)))
    for (cm, initial) in models, κ in (1, 3, 6), N in (40, 1000)
        kw = (; N, p, initial, tspan = (0.0, 200.0), nsims = 4, seed = 36 + κ, keep = :events)
        a = simulate(cm, MFSHNetwork(RegularDegree(κ)); kw...)
        b = simulate(cm, WellMixed(κ); kw...)
        @test all(t -> t.algorithm === :FleetingContactSSA, a.trajectories)
        @test all(r -> same_path(a.trajectories[r], b.trajectories[r]), 1:4)
        @test sum(length ∘ events, a.trajectories) > 4                      # something happened
    end
    # a catalyst in the recipient's own compartment (MassActionSSA counts n_J − 1, here k(K_J − k)/(M − k))
    m = OutbreakModel([:I, :J], [true, false], [OutbreakTransition(:I, :J, 0.5, :contact_trace; via = [:I])]; name = :self)
    for κ in (2, 5)
        sa = OutbreakSpec(m, FleetingContacts(fill(κ, 30)), SeedCount(:I => 30), (0.0, 50.0))
        sb = OutbreakSpec(m, SampledNetwork(WellMixed(κ), 30), SeedCount(:I => 30), (0.0, 50.0))
        for s in 1:5
            @test same_path(simulate(sa; algorithm = FleetingContactSSA(), seed = s, keep = :events),
                            simulate(sb; algorithm = MassActionSSA(), seed = s, keep = :events))
        end
    end
    # interventions: a vaccination pulse, a rate change and a threshold, applied by both samplers alike
    plans = (InterventionPlan([ScheduledStateChange(5.0, :R, 0.5; from = [:S], basis = :eligible)]),
             InterventionPlan([ScheduledRateChange(5.0, :S, :I, :infection, 0.0)]),
             InterventionPlan([ThresholdIntervention(:I, :above, 20, ScheduledStateChange(NaN, :R, 1.0; from = [:S]))]))
    for plan in plans
        sa = OutbreakSpec(sir(0.2, 0.25), FleetingContacts(fill(4, 300)), SeedFraction(:I => 0.02), (0.0, 200.0))
        sb = OutbreakSpec(sir(0.2, 0.25), SampledNetwork(WellMixed(4), 300), SeedFraction(:I => 0.02), (0.0, 200.0))
        a = simulate(sa; algorithm = FleetingContactSSA(), seed = 12, keep = :events, interventions = plan)
        b = simulate(sb; algorithm = MassActionSSA(), seed = 12, keep = :events, interventions = plan)
        @test same_path(a, b)
    end
end

# ---------------------------------------------------------------------------------------------------------------
# Heterogeneous populations: bookkeeping and interventions
# ---------------------------------------------------------------------------------------------------------------

@testset "FleetingContactSSA: node bookkeeping, degree-0 nodes, interventions, reproducibility" begin
    N = 400
    k = [isodd(v) ? 0 : 1 + v % 9 for v in 1:N]                        # every odd node has no stub
    m = sir(0.3, 0.25)
    spec = OutbreakSpec(m, FleetingContacts(k), SeedNodes(:I => [2, 4, 6, 8]), (0.0, 200.0))
    tr = simulate(spec; algorithm = FleetingContactSSA(), seed = 11, keep = :events)
    @test tr.algorithm === :FleetingContactSSA
    # replay the event log: every event moves a node out of its current compartment
    state = fill(1, N); state[[2, 4, 6, 8]] .= 2
    ok = true
    for e in events(tr)
        t = m.transitions[e.transition_index]
        ok &= state[e.node] == m.index_of[t.from]
        state[e.node] = m.index_of[t.to]
    end
    @test ok
    @test [count(==(c), state) for c in 1:3] == tr.counts[:, end]
    @test all(v -> tr.final_infection_counts[v] == 0, 1:2:N)            # no stub: never infected
    @test count(>(0), tr.final_infection_counts) > 50
    @test simulate(spec; algorithm = FleetingContactSSA(), seed = 11, keep = :events).counts == tr.counts
    @test simulate(spec; algorithm = FleetingContactSSA(), rng = NO.stable_rng(11)).counts == tr.counts
    # a population without stubs has no contacts: only the seeds recover
    tr0 = simulate(OutbreakSpec(m, FleetingContacts(zeros(Int, 50)), SeedCount(:I => 5), (0.0, 1e3));
                   algorithm = FleetingContactSSA(), seed = 1, keep = :events)
    @test tr0.counts[:, end] == [45, 0, 5] && length(events(tr0)) == 5

    # interventions: after a pulse moving half of S to R, later infections still take S nodes (the lists by
    # compartment and by degree class follow the moved nodes)
    plan = InterventionPlan([ScheduledStateChange(5.0, :R, 0.5; from = [:S], basis = :eligible)])
    tr = simulate(OutbreakSpec(m, FleetingContacts(k), SeedFraction(:I => 0.02), (0.0, 200.0));
                  algorithm = FleetingContactSSA(), seed = 12, keep = :events, interventions = plan)
    S = compartment_series(tr, :S)
    k5 = findfirst(>=(5.0), tr.times)
    moved = round(Int, 0.5 * S[k5 - 1], RoundNearestTiesAway)
    @test tr.times[k5] == 5.0 && S[k5] == S[k5 - 1] - moved
    infections = count(e -> e.transition_index == 1, events(tr))
    @test count(e -> e.transition_index == 1 && e.time > 5.0, events(tr)) > 5
    @test S[end] == S[1] - infections - moved
    @test all(e -> e.transition_index != 1 || isodd(e.node) == false, events(tr))    # only nodes with stubs
    plan2 = InterventionPlan([ScheduledRateChange(5.0, :S, :I, :infection, 0.0)])
    tr2 = simulate(OutbreakSpec(m, FleetingContacts(k), SeedFraction(:I => 0.02), (0.0, 200.0));
                   algorithm = FleetingContactSSA(), seed = 13, keep = :events, interventions = plan2)
    @test all(e -> e.transition_index != 1 || e.time <= 5.0, events(tr2))
end

# ---------------------------------------------------------------------------------------------------------------
# simulate(model, ::MFSHNetwork): streams and graphs modes
# ---------------------------------------------------------------------------------------------------------------

@testset "simulate(model, MFSHNetwork(d)): fresh stub counts per run, streams (§J.7), graphs modes" begin
    net = MFSHNetwork(EmpiricalDegree(2 => 0.5, 8 => 0.5))
    p = Dict(:τ => 0.5, :γ => 1.0)
    kw = (; N = 500, p, initial = SeedFraction(:I => 0.02), tspan = (0.0, 20.0), nsims = 3, keep = :counts)
    ens = simulate(sir_model(), net; kw..., seed = 99)
    @test ens.spec.network isa SampledNetwork && ens.spec.network.descriptor == net && ens.spec.network.N == 500
    # run r: the stub counts of sample_graph(net, N; rng = stable_rng(b + r)), the SSA stream stable_rng(b + 2^32 + r)
    for r in 1:3
        fc = first(sample_graph(net, 500; rng = NO.stable_rng(99 + r)))
        tr = simulate(OutbreakSpec(ens.spec.model, fc, kw.initial, kw.tspan); algorithm = FleetingContactSSA(),
                      seed = ens.trajectories[r].seed)
        @test ens.trajectories[r].seed == 99 + 2^32 + r
        @test tr.counts == ens.trajectories[r].counts && tr.times == ens.trajectories[r].times
    end
    fixed = simulate(sir_model(), net; kw..., seed = 99, graphs = :fixed)
    @test fixed.spec.network isa FleetingContacts
    @test fixed.spec.network.degrees == first(sample_graph(net, 500; rng = NO.stable_rng(100))).degrees
    pool = simulate(sir_model(), net; kw..., seed = 99, graphs = (:pool, 2))
    @test pool.spec.network isa SampledNetwork
    @test simulate(sir_model(), net; kw..., seed = 99, parallel = true).trajectories[2].counts == ens.trajectories[2].counts
end

# ---------------------------------------------------------------------------------------------------------------
# The law of large numbers: the MFSH ODE of Miller–Slim–Volz
# ---------------------------------------------------------------------------------------------------------------

@testset "law of large numbers: the MFSH ODE (MSV Part II §3.2.2) on :sir_mfsh_pois5 and :sir_mfsh_msv" begin
    # The scenarios' own ensembles: N = 10⁴, 200 runs, 1% seeded, a fresh stub sequence per run, runs conditioned on a
    # major outbreak (every run is one with 100 seeds).
    for id in (:sir_mfsh_pois5, :sir_mfsh_msv)
        sc = scenario(id)
        N = sc.sim.N
        d = sc.network.degrees
        τ, γ = sc.params[:τ], sc.params[:γ]
        ρ = only(last.(seed_fractions(sc.initial)))
        step_ = Float64(step(sc.tgrid))
        ode = mfsh_rk4(d, τ, γ, ρ, sc.tspan[2]; h = step_ / 100, every = 100)
        @test size(ode, 1) == length(sc.tgrid)
        # the RK4 reaches NetworkEpiCore's final-size equation (−ln θ = A + qB(1 − θψ'(θ)/ψ'(1)))
        @test 1 - ode[end, 1] ≈ sc.expected[:final_size] atol = 1e-4
        ens = simulate(sc.model, sc.network; N, p = sc.params, initial = sc.initial, tspan = sc.tspan,
                       nsims = sc.sim.nsims, seed = sc.sim.base_seed, tgrid = sc.tgrid)
        nseed = round(Int, ρ * N)
        major = [tr for tr in ens.trajectories if final_size(tr) * N - nseed >= 0.05 * N]
        @test length(major) == sc.sim.nsims
        ix = [ens.spec.model.index_of[X] for X in (:S, :I, :R)]
        A = cat([Float64.(permutedims(tr.counts[ix, :])) ./ N for tr in major]...; dims = 3)   # time × (S, I, R) × run
        μ = dropdims(mean(A; dims = 3); dims = 3)
        se = dropdims(std(A; dims = 3); dims = 3) ./ sqrt(length(major))
        D = vec(maximum(abs.(μ .- ode); dims = 1))
        fs = final_size.(major)
        zR = (mean(fs) - sc.expected[:final_size]) / (std(fs) / sqrt(length(fs)))
        RESULTS["LLN $(id)"] = (D∞ = D, se_max = vec(maximum(se; dims = 1)), R∞ = mean(fs), ref = sc.expected[:final_size], z = zR)
        @info "WP36c FleetingContactSSA vs the MFSH ODE" scenario = id N runs = length(major) D∞_SIR = join(round.(D; sigdigits = 3), ", ") R∞ = mean(fs) R∞_ODE = sc.expected[:final_size] z_R∞ = zR
        @test all(<(0.01), D)
        @test D[2] < 0.005                                              # prevalence, the design's §E.2 criterion
        @test abs(zR) < 3
    end
end

# ---------------------------------------------------------------------------------------------------------------
# Errors and the scenario route
# ---------------------------------------------------------------------------------------------------------------

@testset "FleetingContactSSA and MFSHNetwork: errors" begin
    m = sir(0.2, 0.25)
    fc = FleetingContacts(fill(4, 100))
    spec = OutbreakSpec(m, fc, SeedFraction(:I => 0.05), (0.0, 10.0))
    for alg in (NextReaction(), DirectSSA(), HAS(), CompositionRejection(), MassActionSSA())
        @test_throws ArgumentError simulate(spec; algorithm = alg, seed = 1)          # no contact graph
    end
    g = random_regular_graph(100, 4; rng = StableRNG(1))
    @test_throws ArgumentError simulate(OutbreakSpec(m, g, SeedFraction(:I => 0.05), (0.0, 10.0));
                                        algorithm = FleetingContactSSA(), seed = 1)
    @test_throws ArgumentError simulate(OutbreakSpec(m, SampledNetwork(WellMixed(4), 100), SeedFraction(:I => 0.05),
                                                     (0.0, 10.0)); algorithm = FleetingContactSSA(), seed = 1)
    net = MFSHNetwork(PoissonDegree(5))
    kw = (; p = Dict(:τ => 0.1, :γ => 0.25), initial = SeedFraction(:I => 0.05), tspan = (0.0, 10.0))
    @test_throws ArgumentError simulate(sir_model(), net; N = 100, kw..., algorithm = NextReaction())
    @test_throws ArgumentError simulate(sir_model(), net; N = 0, kw...)
    @test_throws ArgumentError simulate(sir_model(), net; N = 100, kw..., nsims = 0)
    err = try
        simulate(spec; algorithm = NextReaction(), seed = 1)
    catch e
        sprint(showerror, e)
    end
    @test occursin("FleetingContactSSA", err)
end

@testset "the scenario route for :sir_mfsh_*" begin
    # simulate(sc) and scenario_ensemble(sc) take the sampler from sc.sim.algorithm. NetworkOutbreaks maps :fleeting
    # to FleetingContactSSA and records the stub statistics of each run (no clustering: fleeting contacts have no
    # graph); the MFSH scenarios use it once NetworkEpiCore registers :fleeting for them (a request to the NEC
    # steward). The fleeting-contact process itself is validated above through simulate(model, net).
    sc = derive(scenario(:sir_mfsh_msv); nsims = 3)
    @test NO._algorithm(:fleeting) === FleetingContactSSA()
    @test NO._realised_names(sc) == [:excess_degree, :mean_degree]
    fc, info = NO._scenario_network(sc, 1)
    @test fc isa FleetingContacts && nv(fc) == sc.sim.N
    @test fc.degrees == first(sample_graph(sc.network, sc.sim.N; rng = NO.stable_rng(sc.sim.base_seed + 1))).degrees
    @test NO._realised_values(NO._realised_names(sc), fc, info) == [info.excess_degree, info.mean_degree]
    if sc.sim.algorithm === :fleeting
        ens = simulate(sc)
        @test ens isa OutbreakEnsemble && length(ens) == 3 && all(t -> t.algorithm === :FleetingContactSSA, ens)
        sens = scenario_ensemble(sc)
        @test sens isa ScenarioEnsemble && sens.realised_names == [:excess_degree, :mean_degree]
    else
        @test_broken sc.sim.algorithm === :fleeting          # pending the NetworkEpiCore registry
    end
end
