# Legacy NetworkOutbreaks tests (core), moved verbatim from the 0.1 test/runtests.jl
# (Phase 0 split; see DESIGN_NetworkEpiCore.md §G.1). Owned by the WP that later replaces this area.

using NetworkOutbreaks
using Graphs
using Test
using StableRNGs
using Statistics: mean

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

@testset "Model and seed validation" begin
    comps = [:S, :I, :R]
    inf = [false, true, false]
    trs = [
        OutbreakTransition(:S, :I, 0.5, :infection; via = [:I]),
        OutbreakTransition(:I, :R, 1.0, :spontaneous),
    ]
    model = OutbreakModel(comps, inf, trs; name = :SIR)

    @test_throws ArgumentError OutbreakModel([:S, :S], [false, true], trs)
    @test_throws ArgumentError OutbreakModel([:S, :I], [false], trs)
    @test_throws ArgumentError OutbreakModel(comps, inf,
        [OutbreakTransition(:X, :I, 0.5, :infection)])
    @test_throws ArgumentError OutbreakModel(comps, inf,
        [OutbreakTransition(:S, :I, 0.5, :unknown)])
    @test_throws ArgumentError OutbreakModel(comps, inf,
        [OutbreakTransition(:S, :I, -0.1, :infection)])
    @test_throws ArgumentError OutbreakModel(comps, inf,
        [OutbreakTransition(:S, :I, 0.5, :infection; via = [:X])])

    g = SimpleGraph(10)
    spec = OutbreakSpec(model = model, network = g,
                        initial = SeedFraction(:I => 0.2),
                        tspan = (0.0, 1.0))
    state = NetworkOutbreaks.initial_state(spec, StableRNG(1))
    @test count(==(model.index_of[:I]), state) == 2
    @test count(==(model.index_of[:S]), state) == 8
    @test sum(count(==(i), state) for i in 1:length(comps)) == 10

    @test_throws ArgumentError NetworkOutbreaks.initial_state(
        OutbreakSpec(model = model, network = g,
                     initial = SeedFraction(:Z => 0.1),
                     tspan = (0.0, 1.0)),
        StableRNG(1))
    @test_throws ArgumentError NetworkOutbreaks.initial_state(
        OutbreakSpec(model = model, network = g,
                     initial = SeedFraction(:I => 0.95, :R => 0.1),
                     tspan = (0.0, 1.0)),
        StableRNG(1))

    node_spec = OutbreakSpec(model = model, network = g,
                             initial = SeedNodes(:I => [2, 4]),
                             tspan = (0.0, 1.0))
    node_state = NetworkOutbreaks.initial_state(node_spec, StableRNG(2))
    @test node_state[2] == model.index_of[:I]
    @test node_state[4] == model.index_of[:I]
    @test count(==(model.index_of[:S]), node_state) == 8
    @test_throws ArgumentError NetworkOutbreaks.initial_state(
        OutbreakSpec(model = model, network = g,
                     initial = SeedNodes(:I => [2], :R => [2]),
                     tspan = (0.0, 1.0)),
        StableRNG(2))
    @test_throws ArgumentError NetworkOutbreaks.initial_state(
        OutbreakSpec(model = model, network = g,
                     initial = SeedNodes(:I => [2]; default = :Z),
                     tspan = (0.0, 1.0)),
        StableRNG(2))
end

@testset "Trajectory and network semantics" begin
    model = OutbreakModel([:S, :I, :R], [false, true, false],
        [OutbreakTransition(:S, :I, 0.5, :infection),
         OutbreakTransition(:I, :R, 1.0, :spontaneous)]; name = :SIR)
    counts = [10 9 8;
               0 1 1;
               0 0 1]
    traj = OutbreakTrajectory(model, [0.0, 1.0, 3.0], counts,
                              zeros(Int, 10), OutbreakEvent[], UInt64(1), :unit)
    @test state_at(traj, -1.0) == [10, 0, 0]
    @test state_at(traj, 0.0) == [10, 0, 0]
    @test state_at(traj, 2.0) == [9, 1, 0]
    @test state_at(traj, 3.0) == [8, 1, 1]
    @test state_at(traj, 10.0) == [8, 1, 1]
    @test_throws ArgumentError compartment_series(traj, :X)

    g = SimpleGraph(4)
    updates = [(t = 5, src = 1, dst = 2, action = "add"),
               (t = 1.0, src = 2, dst = 3, action = :remove)]
    tvn = TimeVaryingNetwork(g, updates)
    @test [u.t for u in tvn.updates] == [1.0, 5.0]
    @test tvn.updates[2].action == :add
    @test tvn.updates[2].src isa Int
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
