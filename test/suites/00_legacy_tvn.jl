# Legacy NetworkOutbreaks tests (tvn), moved verbatim from the 0.1 test/runtests.jl
# (Phase 0 split; see DESIGN_NetworkEpiCore.md §G.1). Owned by the WP that later replaces this area.

using NetworkOutbreaks
using Graphs
using Test
using StableRNGs
using Statistics: mean

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
