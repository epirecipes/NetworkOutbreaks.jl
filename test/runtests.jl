using NetworkOutbreaks
using Graphs
using Test
using StableRNGs
using Statistics: mean

@testset "NetworkOutbreaks" begin
    @testset "Direct SSA: SIR on a regular graph" begin
        # Build a simple SIR model directly via OutbreakModel.
        comps = [:S, :I, :R]
        inf   = [false, true, false]
        trs = [
            OutbreakTransition(:S, :I, 0.5, :infection),
            OutbreakTransition(:I, :R, 1.0, :spontaneous),
        ]
        model = OutbreakModel(comps, inf, trs; name = :SIR)
        g = random_regular_graph(500, 4; rng = StableRNG(11))
        spec = OutbreakSpec(model = model, network = g,
                            initial = SeedFraction(:I => 0.02),
                            tspan = (0.0, 50.0))
        traj = simulate(spec; seed = 42)

        @test length(traj.times) >= 2
        @test traj.times[1] == 0.0
        @test traj.times[end] <= 50.0 + 1e-12
        # Conservation
        S = compartment_series(traj, :S)
        I = compartment_series(traj, :I)
        R = compartment_series(traj, :R)
        @test all(S .+ I .+ R .== 500)
        # Final state has no infected (epidemic burned out) — for these
        # parameters R0 ≈ τ·k/γ = 0.5*4/1 = 2 > 1 so most runs reach R > 0.
        @test R[end] >= 1
    end

    @testset "Determinism by seed" begin
        comps = [:S, :I]
        inf = [false, true]
        trs = [
            OutbreakTransition(:S, :I, 0.5, :infection),
            OutbreakTransition(:I, :S, 1.0, :spontaneous),
        ]
        model = OutbreakModel(comps, inf, trs; name = :SIS)
        g = random_regular_graph(200, 3; rng = StableRNG(7))
        spec = OutbreakSpec(model = model, network = g,
                            initial = SeedFraction(:I => 0.05),
                            tspan = (0.0, 20.0))
        a = simulate(spec; seed = 123)
        b = simulate(spec; seed = 123)
        @test a.times == b.times
        @test a.counts == b.counts
    end

    @testset "NextReaction determinism" begin
        comps = [:S, :I]
        inf = [false, true]
        trs = [
            OutbreakTransition(:S, :I, 0.6, :infection),
            OutbreakTransition(:I, :S, 1.0, :spontaneous),
        ]
        model = OutbreakModel(comps, inf, trs; name = :SIS)
        g = random_regular_graph(200, 3; rng = StableRNG(17))
        spec = OutbreakSpec(model = model, network = g,
                            initial = SeedFraction(:I => 0.05),
                            tspan = (0.0, 20.0))
        a = simulate(spec; algorithm = NextReaction(), seed = 123, keep = :events)
        b = simulate(spec; algorithm = NextReaction(), seed = 123, keep = :events)
        @test a.times == b.times
        @test a.counts == b.counts
        @test a.final_infection_counts == b.final_infection_counts
        @test [(e.time, e.transition_index, e.node) for e in a.events] ==
              [(e.time, e.transition_index, e.node) for e in b.events]
        @test a.algorithm == :NextReaction
    end

    @testset "NextReaction agrees with DirectSSA in ensemble" begin
        comps = [:S, :I]
        inf = [false, true]
        trs = [
            OutbreakTransition(:S, :I, 0.6, :infection),
            OutbreakTransition(:I, :S, 1.0, :spontaneous),
        ]
        model = OutbreakModel(comps, inf, trs; name = :SIS)
        g = random_regular_graph(400, 4; rng = StableRNG(33))
        spec = OutbreakSpec(model = model, network = g,
                            initial = SeedFraction(:I => 0.05),
                            tspan = (0.0, 30.0))
        tgrid = collect(range(0.0, 30.0; length = 61))
        direct = simulate_ensemble(spec; nsims = 40, seed = 101,
                                   algorithm = DirectSSA())
        next = simulate_ensemble(spec; nsims = 40, seed = 202,
                                 algorithm = NextReaction())
        _, μ_direct = mean_curve(direct, :I; tgrid = tgrid)
        _, μ_next = mean_curve(next, :I; tgrid = tgrid)
        @test maximum(abs.(μ_direct .- μ_next) ./ 400) <= 0.05
    end

    @testset "Parallel ensemble determinism" begin
        comps = [:S, :I]
        inf = [false, true]
        trs = [
            OutbreakTransition(:S, :I, 0.6, :infection),
            OutbreakTransition(:I, :S, 1.0, :spontaneous),
        ]
        model = OutbreakModel(comps, inf, trs; name = :SIS)
        g = random_regular_graph(160, 3; rng = StableRNG(19))
        spec = OutbreakSpec(model = model, network = g,
                            initial = SeedFraction(:I => 0.05),
                            tspan = (0.0, 15.0))
        tgrid = collect(0.0:1.0:15.0)
        seq = simulate_ensemble(spec; nsims = 12, seed = 404,
                                algorithm = NextReaction(), parallel = false)
        par = simulate_ensemble(spec; nsims = 12, seed = 404,
                                algorithm = NextReaction(), parallel = true)
        _, μ_seq = mean_curve(seq, :I; tgrid = tgrid)
        _, μ_par = mean_curve(par, :I; tgrid = tgrid)
        @test μ_seq == μ_par
        @test [tr.seed for tr in seq] == [tr.seed for tr in par]
        @test [tr.times for tr in seq] == [tr.times for tr in par]
    end

    @testset "Reinfection counts and ensemble averaging (SIS)" begin
        comps = [:S, :I]
        inf = [false, true]
        trs = [
            OutbreakTransition(:S, :I, 0.6, :infection),
            OutbreakTransition(:I, :S, 1.0, :spontaneous),
        ]
        model = OutbreakModel(comps, inf, trs; name = :SIS)
        g = random_regular_graph(400, 4; rng = StableRNG(3))
        spec = OutbreakSpec(model = model, network = g,
                            initial = SeedFraction(:I => 0.05),
                            tspan = (0.0, 30.0))
        ens = simulate_ensemble(spec; nsims = 30, seed = 1)
        @test length(ens) == 30
        t, μI = mean_curve(ens, :I)
        @test length(t) == 200
        # Endemic prevalence positive for these supercritical parameters.
        @test μI[end] > 5.0
        # Reinfection histogram sums to N for each trajectory.
        for traj in ens
            h = reinfection_histogram(traj; L = 5)
            @test sum(h) == 400
        end
        # Final size on a single trajectory.
        @test 0.0 <= final_size(ens[1]) <= 1.0
    end

    @testset "Spec validation" begin
        comps = [:S, :I]
        inf = [false, true]
        trs = [
            OutbreakTransition(:S, :I, 0.5, :infection),
            OutbreakTransition(:I, :S, 1.0, :spontaneous),
        ]
        model = OutbreakModel(comps, inf, trs; name = :SIS)
        g = random_regular_graph(50, 3; rng = StableRNG(2))
        @test_throws ArgumentError SeedFraction(:Z => 1.0) |>
            seed -> NetworkOutbreaks.initial_state(
                OutbreakSpec(model = model, network = g, initial = seed,
                             tspan = (0.0, 1.0)),
                StableRNG(0))
    end

    @testset "EBM adapter (loads via package extension)" begin
        using EdgeBasedModels: sir_model, sis_model
        prog = sis_model()
        # Adapter is provided by the extension; numeric parameters
        # substitute the symbolic rates :β, :γ.
        model = OutbreakModel(prog, Dict(:β => 0.5, :γ => 1.0))
        @test :S in model.compartments
        @test :I in model.compartments
        # The S→I infection transition exists.
        @test any(t -> t.from == :S && t.to == :I && t.type == :infection,
                  model.transitions)
        # Spontaneous I→S exists.
        @test any(t -> t.from == :I && t.to == :S && t.type == :spontaneous,
                  model.transitions)
    end

    @testset "NBM adapter (loads via package extension)" begin
        import NodeBasedModels
        nbm = NodeBasedModels.sir_model()
        model = OutbreakModel(nbm, Dict(:τ => 0.4, :γ => 1.0))
        @test model.compartments == [:S, :I, :R]
        @test model.infectious == [false, true, false]
        rates = Dict((t.from, t.to, t.type) => t.rate for t in model.transitions)
        @test rates[(:S, :I, :infection)] ≈ 0.4
        @test rates[(:I, :R, :spontaneous)] ≈ 1.0
    end

    @testset "Vignette import smoke" begin
        import EdgeBasedModels
        import NodeBasedModels
        using EdgeBasedModels: build_sis_reinfection, polynomial_pgf, compartment
        using NodeBasedModels: with_reinfection_counting, regular_network,
                               generate_pairwise, BernoulliClosure, solve_pairwise,
                               reinfection_totals
        L = 1
        β = 0.4
        γ = 1.0
        nbm_sys = generate_pairwise(with_reinfection_counting(NodeBasedModels.sis_model(τ = :β), L),
                                    regular_network(3), BernoulliClosure();
                                    tspan = (0.0, 1.0), seed_fraction = 0.05)
        nbm_sol = solve_pairwise(nbm_sys, Dict(:β => β, :γ => γ); saveat = 1.0)
        @test haskey(reinfection_totals(nbm_sys, nbm_sol), :I)
        ebm_sys = build_sis_reinfection(polynomial_pgf([0.0, 0.0, 0.0, 1.0]), β, γ, L)
        ebm_sol = EdgeBasedModels.solve_epidemic(ebm_sys; tspan = (0.0, 1.0), saveat = 1.0)
        @test length(compartment(ebm_sol, ebm_sys, :I)) == 2
    end

    # -----------------------------------------------------------------------
    # Phase 3 tests
    # -----------------------------------------------------------------------

    @testset "CompositionRejection determinism" begin
        comps = [:S, :I]
        inf = [false, true]
        trs = [
            OutbreakTransition(:S, :I, 0.6, :infection),
            OutbreakTransition(:I, :S, 1.0, :spontaneous),
        ]
        model = OutbreakModel(comps, inf, trs; name = :SIS)
        g = random_regular_graph(200, 3; rng = StableRNG(17))
        spec = OutbreakSpec(model = model, network = g,
                            initial = SeedFraction(:I => 0.05),
                            tspan = (0.0, 20.0))
        a = simulate(spec; algorithm = CompositionRejection(), seed = 555)
        b = simulate(spec; algorithm = CompositionRejection(), seed = 555)
        @test a.times  == b.times
        @test a.counts == b.counts
        @test a.algorithm == :CompositionRejection
    end

    @testset "CompositionRejection agrees with DirectSSA in ensemble" begin
        comps = [:S, :I]
        inf = [false, true]
        trs = [
            OutbreakTransition(:S, :I, 0.6, :infection),
            OutbreakTransition(:I, :S, 1.0, :spontaneous),
        ]
        model = OutbreakModel(comps, inf, trs; name = :SIS)
        g = random_regular_graph(400, 4; rng = StableRNG(33))
        spec = OutbreakSpec(model = model, network = g,
                            initial = SeedFraction(:I => 0.05),
                            tspan = (0.0, 30.0))
        tgrid = collect(range(0.0, 30.0; length = 61))
        direct = simulate_ensemble(spec; nsims = 40, seed = 101, algorithm = DirectSSA())
        cr     = simulate_ensemble(spec; nsims = 40, seed = 202,
                                   algorithm = CompositionRejection())
        _, μ_direct = mean_curve(direct, :I; tgrid = tgrid)
        _, μ_cr     = mean_curve(cr,     :I; tgrid = tgrid)
        @test maximum(abs.(μ_direct .- μ_cr) ./ 400) <= 0.05
    end

    @testset "Three-way algorithm smoke test" begin
        comps = [:S, :I]
        inf = [false, true]
        trs = [
            OutbreakTransition(:S, :I, 0.6, :infection),
            OutbreakTransition(:I, :S, 1.0, :spontaneous),
        ]
        model = OutbreakModel(comps, inf, trs; name = :SIS)
        g = random_regular_graph(400, 4; rng = StableRNG(44))
        spec = OutbreakSpec(model = model, network = g,
                            initial = SeedFraction(:I => 0.05),
                            tspan = (0.0, 30.0))
        tgrid = collect(range(0.0, 30.0; length = 31))
        direct = simulate_ensemble(spec; nsims = 40, seed = 301, algorithm = DirectSSA())
        next   = simulate_ensemble(spec; nsims = 40, seed = 302, algorithm = NextReaction())
        cr     = simulate_ensemble(spec; nsims = 40, seed = 303, algorithm = CompositionRejection())
        _, μD = mean_curve(direct, :I; tgrid = tgrid)
        _, μN = mean_curve(next,   :I; tgrid = tgrid)
        _, μC = mean_curve(cr,     :I; tgrid = tgrid)
        @test maximum(abs.(μD .- μN) ./ 400) <= 0.05
        @test maximum(abs.(μD .- μC) ./ 400) <= 0.05
        @test maximum(abs.(μN .- μC) ./ 400) <= 0.05
    end

    @testset "TimeVaryingNetwork determinism (DirectSSA)" begin
        comps = [:S, :I]
        inf   = [false, true]
        trs   = [OutbreakTransition(:S, :I, 1.5, :infection),
                 OutbreakTransition(:I, :S, 0.5, :spontaneous)]
        model = OutbreakModel(comps, inf, trs; name = :SIS)
        g = SimpleGraph(4); add_edge!(g, 1, 2)
        updates = [(t = 5.0, src = 1, dst = 3, action = :add),
                   (t = 5.0, src = 2, dst = 4, action = :add)]
        tvn = TimeVaryingNetwork(g, updates)
        spec = OutbreakSpec(model = model, network = tvn,
                            initial = SeedNodes(:I => [1]),
                            tspan = (0.0, 20.0))
        a = simulate(spec; algorithm = DirectSSA(), seed = 77)
        b = simulate(spec; algorithm = DirectSSA(), seed = 77)
        @test a.times  == b.times
        @test a.counts == b.counts
    end

    @testset "TimeVaryingNetwork determinism (NextReaction)" begin
        comps = [:S, :I]
        inf   = [false, true]
        trs   = [OutbreakTransition(:S, :I, 1.5, :infection),
                 OutbreakTransition(:I, :S, 0.5, :spontaneous)]
        model = OutbreakModel(comps, inf, trs; name = :SIS)
        g = SimpleGraph(4); add_edge!(g, 1, 2)
        updates = [(t = 5.0, src = 1, dst = 3, action = :add),
                   (t = 5.0, src = 2, dst = 4, action = :add)]
        tvn = TimeVaryingNetwork(g, updates)
        spec = OutbreakSpec(model = model, network = tvn,
                            initial = SeedNodes(:I => [1]),
                            tspan = (0.0, 20.0))
        a = simulate(spec; algorithm = NextReaction(), seed = 88)
        b = simulate(spec; algorithm = NextReaction(), seed = 88)
        @test a.times  == b.times
        @test a.counts == b.counts
    end

    @testset "TimeVaryingNetwork sanity: edge addition increases prevalence" begin
        # 4 nodes: node 1 (I) — node 2 (S) connected always.
        # Nodes 3, 4 isolated.  At t = 5 we add edges 1–3 and 2–4.
        # The TVN ensemble should have higher mean I at t = 15 than the static one.
        comps = [:S, :I]
        inf   = [false, true]
        trs   = [OutbreakTransition(:S, :I, 2.0, :infection),
                 OutbreakTransition(:I, :S, 0.5, :spontaneous)]
        model = OutbreakModel(comps, inf, trs; name = :SIS)

        g_tvn = SimpleGraph(4); add_edge!(g_tvn, 1, 2)
        updates = [(t = 5.0, src = 1, dst = 3, action = :add),
                   (t = 5.0, src = 2, dst = 4, action = :add)]
        tvn = TimeVaryingNetwork(g_tvn, updates)

        g_static = SimpleGraph(4); add_edge!(g_static, 1, 2)

        spec_tvn    = OutbreakSpec(model = model, network = tvn,
                                   initial = SeedNodes(:I => [1]),
                                   tspan = (0.0, 20.0))
        spec_static = OutbreakSpec(model = model, network = g_static,
                                   initial = SeedNodes(:I => [1]),
                                   tspan = (0.0, 20.0))
        tgrid = [15.0]
        ens_tvn    = simulate_ensemble(spec_tvn;    nsims = 300, seed = 9001,
                                       algorithm = DirectSSA())
        ens_static = simulate_ensemble(spec_static; nsims = 300, seed = 9002,
                                       algorithm = DirectSSA())
        _, μ_tvn    = mean_curve(ens_tvn,    :I; tgrid = tgrid)
        _, μ_static = mean_curve(ens_static, :I; tgrid = tgrid)
        # Nodes 3 and 4 can only be infected in the TVN scenario.
        @test μ_tvn[1] > μ_static[1]
    end

    @testset "TimeVaryingNetwork backward compat: AbstractGraph still works" begin
        # Plain AbstractGraph passed to OutbreakSpec → auto-wrapped, old tests unaffected.
        comps = [:S, :I]
        inf   = [false, true]
        trs   = [OutbreakTransition(:S, :I, 0.5, :infection),
                 OutbreakTransition(:I, :S, 1.0, :spontaneous)]
        model = OutbreakModel(comps, inf, trs; name = :SIS)
        g = random_regular_graph(100, 3; rng = StableRNG(5))
        spec = OutbreakSpec(model = model, network = g,
                            initial = SeedFraction(:I => 0.05),
                            tspan = (0.0, 10.0))
        @test spec.network isa StaticNetwork
        traj = simulate(spec; seed = 1)
        @test length(traj.times) >= 2
    end

    @testset "CompositionRejection rejects TimeVaryingNetwork" begin
        comps = [:S, :I]
        inf   = [false, true]
        trs   = [OutbreakTransition(:S, :I, 0.5, :infection),
                 OutbreakTransition(:I, :S, 1.0, :spontaneous)]
        model = OutbreakModel(comps, inf, trs; name = :SIS)
        g = SimpleGraph(4); add_edge!(g, 1, 2)
        tvn = TimeVaryingNetwork(g, [(t=5.0, src=1, dst=3, action=:add)])
        spec = OutbreakSpec(model = model, network = tvn,
                            initial = SeedNodes(:I => [1]),
                            tspan = (0.0, 10.0))
        @test_throws ArgumentError simulate(spec; algorithm = CompositionRejection(), seed = 1)
    end

    @testset "HAS: dispatch returns expected error" begin
        comps = [:S, :I]
        inf   = [false, true]
        trs   = [OutbreakTransition(:S, :I, 0.5, :infection),
                 OutbreakTransition(:I, :S, 1.0, :spontaneous)]
        model = OutbreakModel(comps, inf, trs; name = :SIS)
        g = random_regular_graph(50, 3; rng = StableRNG(3))
        spec = OutbreakSpec(model = model, network = g,
                            initial = SeedFraction(:I => 0.05),
                            tspan = (0.0, 5.0))
        # HAS is now fully implemented — it must not throw an ErrorException.
        @test_nowarn simulate(spec; algorithm = HAS(), seed = 1)
        traj = simulate(spec; algorithm = HAS(), seed = 1)
        @test traj.algorithm == :HAS
    end

    @testset "via-specific infection event indices" begin
        comps = [:S, :A, :B, :I]
        inf   = [false, true, true, true]
        trs   = [OutbreakTransition(:S, :I, 1.0, :infection; via = [:A]),
                 OutbreakTransition(:S, :I, 1.0, :infection; via = [:B])]
        model = OutbreakModel(comps, inf, trs; name = :via_index)
        g = SimpleGraph(2); add_edge!(g, 1, 2)
        spec = OutbreakSpec(model = model, network = g,
                            initial = SeedNodes(:B => [2]),
                            tspan = (0.0, 10.0))
        traj = simulate(spec; algorithm = DirectSSA(), seed = 1, keep = :events)
        @test length(traj.events) == 1
        @test traj.events[1].transition_index == 2
    end

    @testset "CompositionRejection keep=:events" begin
        comps = [:S, :I]
        inf   = [false, true]
        trs   = [OutbreakTransition(:S, :I, 0.6, :infection),
                 OutbreakTransition(:I, :S, 1.0, :spontaneous)]
        model = OutbreakModel(comps, inf, trs; name = :SIS)
        g = random_regular_graph(100, 3; rng = StableRNG(22))
        spec = OutbreakSpec(model = model, network = g,
                            initial = SeedFraction(:I => 0.05),
                            tspan = (0.0, 10.0))
        traj = simulate(spec; algorithm = CompositionRejection(), seed = 42,
                        keep = :events)
        @test length(traj.events) == length(traj.times) - 2  # exclude initial+final
        @test all(e -> 1 <= e.transition_index <= length(model.transitions),
                  traj.events)
    end

    @testset "TimeVaryingNetwork: NextReaction sanity (edge addition)" begin
        comps = [:S, :I]
        inf   = [false, true]
        trs   = [OutbreakTransition(:S, :I, 2.0, :infection),
                 OutbreakTransition(:I, :S, 0.5, :spontaneous)]
        model = OutbreakModel(comps, inf, trs; name = :SIS)

        g_tvn = SimpleGraph(4); add_edge!(g_tvn, 1, 2)
        updates = [(t = 5.0, src = 1, dst = 3, action = :add),
                   (t = 5.0, src = 2, dst = 4, action = :add)]
        tvn = TimeVaryingNetwork(g_tvn, updates)

        g_static = SimpleGraph(4); add_edge!(g_static, 1, 2)

        spec_tvn    = OutbreakSpec(model = model, network = tvn,
                                   initial = SeedNodes(:I => [1]),
                                   tspan = (0.0, 20.0))
        spec_static = OutbreakSpec(model = model, network = g_static,
                                   initial = SeedNodes(:I => [1]),
                                   tspan = (0.0, 20.0))
        tgrid = [15.0]
        ens_tvn    = simulate_ensemble(spec_tvn;    nsims = 300, seed = 9011,
                                       algorithm = NextReaction())
        ens_static = simulate_ensemble(spec_static; nsims = 300, seed = 9012,
                                       algorithm = NextReaction())
        _, μ_tvn    = mean_curve(ens_tvn,    :I; tgrid = tgrid)
        _, μ_static = mean_curve(ens_static, :I; tgrid = tgrid)
        @test μ_tvn[1] > μ_static[1]
    end

    # -----------------------------------------------------------------------
    @testset "MultiplexNetwork (DirectSSA)" begin
        comps = [:S, :I, :R]
        inf   = [false, true, false]
        trs   = [OutbreakTransition(:S, :I, 0.3, :infection; via=[:I]),
                 OutbreakTransition(:I, :R, 0.2, :spontaneous)]
        model = OutbreakModel(comps, inf, trs; name = :SIR)
        N = 400
        g1 = erdos_renyi(N, 4/N, rng=StableRNG(101))
        g2 = erdos_renyi(N, 4/N, rng=StableRNG(102))
        seed = SeedFraction(:I => 0.02)
        tspan = (0.0, 25.0)

        # Single-layer multiplex with weight 1 ≡ static graph
        spec_single = OutbreakSpec(; model, network=MultiplexNetwork([g1], [1.0]),
                                   initial=seed, tspan)
        spec_static = OutbreakSpec(; model, network=g1, initial=seed, tspan)
        ens_m = simulate_ensemble(spec_single; nsims=30, seed=11, algorithm=DirectSSA())
        ens_s = simulate_ensemble(spec_static; nsims=30, seed=11, algorithm=DirectSSA())
        finalR_m = mean(compartment_series(s, :R)[end] for s in ens_m)
        finalR_s = mean(compartment_series(s, :R)[end] for s in ens_s)
        @test isapprox(finalR_m, finalR_s; rtol=0.10)

        # Two distinct layers should produce more infections than a single layer.
        spec_two = OutbreakSpec(; model,
                                network=MultiplexNetwork([g1, g2], [1.0, 1.0]),
                                initial=seed, tspan)
        ens_two = simulate_ensemble(spec_two; nsims=30, seed=11, algorithm=DirectSSA())
        finalR_two = mean(compartment_series(s, :R)[end] for s in ens_two)
        @test finalR_two > finalR_s

        # Validation: non-positive layer rates rejected.
        @test_throws ArgumentError MultiplexNetwork([g1, g2], [1.0, -0.1])
        # Validation: layer node counts must agree.
        g3 = erdos_renyi(N+1, 4/N, rng=StableRNG(103))
        @test_throws ArgumentError MultiplexNetwork([g1, g3], [1.0, 1.0])

        # Other algorithms reject MultiplexNetwork.
        spec_nr = OutbreakSpec(; model,
                               network=MultiplexNetwork([g1, g2], [1.0, 1.0]),
                               initial=seed, tspan)
        @test_throws ArgumentError simulate(spec_nr; algorithm=NextReaction(), seed=1)
        @test_throws ArgumentError simulate(spec_nr; algorithm=CompositionRejection(), seed=1)
        @test_throws ArgumentError simulate(spec_nr; algorithm=HAS(), seed=1)
    end

    @testset "HAS algorithm" begin

        # Shared SIS model for most tests.
        comps = [:S, :I]
        inf   = [false, true]
        trs_sis = [OutbreakTransition(:S, :I, 0.6, :infection),
                   OutbreakTransition(:I, :S, 1.0, :spontaneous)]
        model_sis = OutbreakModel(comps, inf, trs_sis; name = :SIS)

        # ---- 1. No more ErrorException ----------------------------------------
        @testset "HAS no longer throws" begin
            g = random_regular_graph(50, 3; rng = StableRNG(1))
            spec = OutbreakSpec(model = model_sis, network = g,
                                initial = SeedFraction(:I => 0.05),
                                tspan = (0.0, 10.0))
            @test_nowarn simulate(spec; algorithm = HAS(), seed = 7)
        end

        # ---- 2. Determinism ---------------------------------------------------
        @testset "HAS determinism" begin
            g = random_regular_graph(200, 3; rng = StableRNG(17))
            spec = OutbreakSpec(model = model_sis, network = g,
                                initial = SeedFraction(:I => 0.05),
                                tspan = (0.0, 20.0))
            a = simulate(spec; algorithm = HAS(), seed = 42)
            b = simulate(spec; algorithm = HAS(), seed = 42)
            @test a.times  == b.times
            @test a.counts == b.counts
            @test a.final_infection_counts == b.final_infection_counts
        end

        # ---- 3. Statistical agreement with DirectSSA -------------------------
        @testset "HAS agrees with DirectSSA (SIS ensemble)" begin
            g    = random_regular_graph(400, 4; rng = StableRNG(11))
            spec = OutbreakSpec(model = model_sis, network = g,
                                initial = SeedFraction(:I => 0.05),
                                tspan = (0.0, 30.0))
            tgrid  = collect(range(0.0, 30.0; length = 61))
            e_dir  = simulate_ensemble(spec; nsims = 40, seed = 101,
                                       algorithm = DirectSSA(), parallel = true)
            e_has  = simulate_ensemble(spec; nsims = 40, seed = 20240502,
                                       algorithm = HAS(), parallel = true)
            _, μ_dir = mean_curve(e_dir, :I; tgrid = tgrid)
            _, μ_has = mean_curve(e_has, :I; tgrid = tgrid)
            @test maximum(abs.(μ_dir .- μ_has) ./ 400) <= 0.05
        end

        # ---- 4. Three-way agreement (DirectSSA, NextReaction, CR, HAS) -------
        @testset "HAS agrees with three other algorithms" begin
            g    = random_regular_graph(400, 4; rng = StableRNG(33))
            spec = OutbreakSpec(model = model_sis, network = g,
                                initial = SeedFraction(:I => 0.05),
                                tspan = (0.0, 20.0))
            tgrid = collect(range(0.0, 20.0; length = 41))
            e_dir  = simulate_ensemble(spec; nsims = 40, seed = 301,
                                       algorithm = DirectSSA())
            e_nr   = simulate_ensemble(spec; nsims = 40, seed = 302,
                                       algorithm = NextReaction())
            e_cr   = simulate_ensemble(spec; nsims = 40, seed = 303,
                                       algorithm = CompositionRejection())
            e_has  = simulate_ensemble(spec; nsims = 40, seed = 304,
                                       algorithm = HAS())
            _, μ_dir = mean_curve(e_dir, :I; tgrid = tgrid)
            _, μ_nr  = mean_curve(e_nr,  :I; tgrid = tgrid)
            _, μ_cr  = mean_curve(e_cr,  :I; tgrid = tgrid)
            _, μ_has = mean_curve(e_has, :I; tgrid = tgrid)
            N = 400
            @test maximum(abs.(μ_has .- μ_dir) ./ N) < 0.06
            @test maximum(abs.(μ_has .- μ_nr)  ./ N) < 0.06
            @test maximum(abs.(μ_has .- μ_cr)  ./ N) < 0.06
        end

        # ---- 5. Final-state conservation (S + I = N) -------------------------
        @testset "HAS conservation" begin
            g = random_regular_graph(100, 3; rng = StableRNG(22))
            spec = OutbreakSpec(model = model_sis, network = g,
                                initial = SeedFraction(:I => 0.05),
                                tspan = (0.0, 10.0))
            traj = simulate(spec; algorithm = HAS(), seed = 55)
            S = compartment_series(traj, :S)
            I = compartment_series(traj, :I)
            @test all(S .+ I .== 100)
        end

        # ---- 6. Event log consistency ----------------------------------------
        @testset "HAS event log" begin
            g = random_regular_graph(100, 3; rng = StableRNG(22))
            spec = OutbreakSpec(model = model_sis, network = g,
                                initial = SeedFraction(:I => 0.05),
                                tspan = (0.0, 10.0))
            traj = simulate(spec; algorithm = HAS(), seed = 42, keep = :events)
            @test length(traj.events) == length(traj.times) - 2
            @test all(e -> 1 <= e.transition_index <= length(model_sis.transitions),
                      traj.events)
            @test traj.algorithm == :HAS
        end

        # ---- 7. TimeVaryingNetwork sanity ------------------------------------
        @testset "HAS TimeVaryingNetwork" begin
            # Use high-R0 SIS (τ=2.0, γ=0.5) so the epidemic persists in the
            # 4-node graph.  Same parameters and graph structure as the
            # NextReaction TVN test.
            trs_tvn   = [OutbreakTransition(:S, :I, 2.0, :infection),
                         OutbreakTransition(:I, :S, 0.5, :spontaneous)]
            model_tvn = OutbreakModel([:S, :I], [false, true], trs_tvn; name = :SIS_TVN)

            g_tvn = SimpleGraph(4); add_edge!(g_tvn, 1, 2)
            updates = [(t = 5.0, src = 1, dst = 3, action = :add),
                       (t = 5.0, src = 2, dst = 4, action = :add)]
            tvn = TimeVaryingNetwork(g_tvn, updates)

            g_static = SimpleGraph(4); add_edge!(g_static, 1, 2)

            spec_tvn    = OutbreakSpec(model = model_tvn, network = tvn,
                                       initial = SeedNodes(:I => [1]),
                                       tspan = (0.0, 20.0))
            spec_static = OutbreakSpec(model = model_tvn, network = g_static,
                                       initial = SeedNodes(:I => [1]),
                                       tspan = (0.0, 20.0))
            tgrid = [15.0]
            ens_tvn    = simulate_ensemble(spec_tvn;    nsims = 300, seed = 9011,
                                           algorithm = HAS())
            ens_static = simulate_ensemble(spec_static; nsims = 300, seed = 9012,
                                           algorithm = HAS())
            _, μ_tvn    = mean_curve(ens_tvn,    :I; tgrid = tgrid)
            _, μ_static = mean_curve(ens_static, :I; tgrid = tgrid)
            # Adding edges at t=5 connects isolated nodes → higher prevalence.
            @test μ_tvn[1] > μ_static[1]
        end

    end  # @testset "HAS algorithm"

end  # @testset "NetworkOutbreaks"

include("test_eon_patterns.jl")
