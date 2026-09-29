# Legacy NetworkOutbreaks tests (algorithms), moved verbatim from the 0.1 test/runtests.jl
# (Phase 0 split; see DESIGN_NetworkEpiCore.md §G.1). Owned by the WP that later replaces this area.

using NetworkOutbreaks
using Graphs
using Test
using StableRNGs
using Statistics: mean

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
