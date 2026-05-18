"""
EoN-equivalent test suite for NetworkOutbreaks.jl

Tests algorithm agreement, conservation, and ODE-vs-SSA patterns
from EoN/tests/test_from_joel.py.
"""

using Test
using NetworkOutbreaks
using Graphs
using StableRNGs
using Statistics

@testset "EoN-equivalent tests" begin

    @testset "DirectSSA vs NextReaction agreement (test_SIR_dynamics)" begin
        G = random_regular_graph(500, 6; rng = StableRNG(42))
        model = OutbreakModel([:S, :I, :R], [false, true, false],
            [OutbreakTransition(:S, :I, 0.3, :infection; via = [:I]),
             OutbreakTransition(:I, :R, 1.0, :spontaneous)]; name = :SIR)
        spec = OutbreakSpec(model = model, network = G,
            initial = SeedFraction(:I => 0.05), tspan = (0.0, 20.0))

        # Ensemble means should agree (individual trajectories may differ)
        ens_d = simulate_ensemble(spec; nsims = 50, seed = 1, algorithm = DirectSSA())
        ens_n = simulate_ensemble(spec; nsims = 50, seed = 1, algorithm = NextReaction())

        fs_d = mean(final_size(t; recovered = :R) for t in ens_d.trajectories) / 500
        fs_n = mean(final_size(t; recovered = :R) for t in ens_n.trajectories) / 500

        # Ensemble final sizes should agree within 10%
        @test isapprox(fs_d, fs_n; rtol = 0.10)
    end

    @testset "Conservation: S+I+R=N (test_SIR_dynamics)" begin
        N = 500
        G = random_regular_graph(N, 6; rng = StableRNG(42))
        model = OutbreakModel([:S, :I, :R], [false, true, false],
            [OutbreakTransition(:S, :I, 1/6, :infection; via = [:I]),
             OutbreakTransition(:I, :R, 0.25, :spontaneous)]; name = :SIR)
        spec = OutbreakSpec(model = model, network = G,
            initial = SeedFraction(:I => 0.01), tspan = (0.0, 40.0))
        traj = simulate(spec; seed = UInt64(1))

        # Check conservation at multiple time points
        for t in 0.0:5.0:40.0
            st = state_at(traj, t)
            total = sum(st[i] for i in 1:3)
            @test total == N
        end
    end

    @testset "SIS endemic equilibrium (test_SIS_dynamics)" begin
        N = 1000
        G = erdos_renyi(N, 5 / (N - 1); rng = StableRNG(42))
        model = OutbreakModel([:S, :I], [false, true],
            [OutbreakTransition(:S, :I, 1/6, :infection; via = [:I]),
             OutbreakTransition(:I, :S, 0.25, :spontaneous)]; name = :SIS)
        spec = OutbreakSpec(model = model, network = G,
            initial = SeedFraction(:I => 0.05), tspan = (0.0, 80.0))

        # Run ensemble
        ens = simulate_ensemble(spec; nsims = 20, seed = 1, parallel = true)
        # Mean I at end should be near endemic equilibrium (> 0)
        I_end = mean(state_at(t, 80.0)[model.index_of[:I]] / N for t in ens.trajectories)
        @test I_end > 0.3  # well above 0 — endemic
        @test I_end < 0.9  # not everyone infected
    end

    @testset "Final size monotonic in tau (test_SIR_final_sizes)" begin
        N = 500
        G = random_regular_graph(N, 6; rng = StableRNG(42))

        final_sizes = Float64[]
        for τ in [0.05, 0.1, 0.2, 0.5]
            model = OutbreakModel([:S, :I, :R], [false, true, false],
                [OutbreakTransition(:S, :I, τ, :infection; via = [:I]),
                 OutbreakTransition(:I, :R, 0.25, :spontaneous)]; name = :SIR)
            spec = OutbreakSpec(model = model, network = G,
                initial = SeedFraction(:I => 0.05), tspan = (0.0, 60.0))
            ens = simulate_ensemble(spec; nsims = 30, seed = 1)
            fs = mean(final_size(t; recovered = :R) for t in ens.trajectories)
            push!(final_sizes, fs / N)
        end
        # Final size should be monotonically increasing with tau
        @test issorted(final_sizes)
    end

    @testset "MultiplexNetwork (test_SIR_dynamics)" begin
        N = 200
        g1 = erdos_renyi(N, 3 / (N - 1); rng = StableRNG(1))
        g2 = erdos_renyi(N, 5 / (N - 1); rng = StableRNG(2))
        net = MultiplexNetwork([g1, g2], [0.2, 0.1])
        model = OutbreakModel([:S, :I, :R], [false, true, false],
            [OutbreakTransition(:S, :I, 1.0, :infection; via = [:I]),
             OutbreakTransition(:I, :R, 0.25, :spontaneous)]; name = :SIR)
        spec = OutbreakSpec(model = model, network = net,
            initial = SeedFraction(:I => 0.05), tspan = (0.0, 40.0))
        traj = simulate(spec; seed = UInt64(1))
        fs = final_size(traj; recovered = :R)
        @test fs > 0  # epidemic should occur on multiplex
    end

    @testset "Contact tracing reduces epidemic" begin
        N = 500
        G = erdos_renyi(N, 5 / (N - 1); rng = StableRNG(42))

        # Without tracing
        sir = OutbreakModel([:S, :I, :R], [false, true, false],
            [OutbreakTransition(:S, :I, 1/6, :infection; via = [:I]),
             OutbreakTransition(:I, :R, 0.25, :spontaneous)]; name = :SIR)
        spec_sir = OutbreakSpec(model = sir, network = G,
            initial = SeedFraction(:I => 0.05), tspan = (0.0, 40.0))
        ens_sir = simulate_ensemble(spec_sir; nsims = 30, seed = 1)
        fs_sir = mean(final_size(t; recovered = :R) for t in ens_sir.trajectories)

        # With tracing
        sirq = OutbreakModel([:S, :I, :R, :Q], [false, true, false, false],
            [OutbreakTransition(:S, :I, 1/6, :infection; via = [:I]),
             OutbreakTransition(:I, :R, 0.25, :spontaneous),
             OutbreakTransition(:S, :Q, 0.3, :contact_trace; via = [:I])]; name = :SIRQ)
        spec_ct = OutbreakSpec(model = sirq, network = G,
            initial = SeedFraction(:I => 0.05), tspan = (0.0, 40.0))
        ens_ct = simulate_ensemble(spec_ct; nsims = 30, seed = 1)
        fs_ct = mean(final_size(t; recovered = :R) for t in ens_ct.trajectories)

        # Tracing should reduce final size
        @test fs_ct < fs_sir
    end
end
