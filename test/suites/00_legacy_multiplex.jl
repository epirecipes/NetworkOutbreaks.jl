# Legacy NetworkOutbreaks tests (multiplex), moved from the 0.1 test/runtests.jl (Phase 0 split; see
# DESIGN_NetworkEpiCore.md §G.1). Updated for 0.2: the graph container MultiplexNetwork of 0.1 is MultiplexGraph
# (MultiplexNetwork is NetworkEpiCore's descriptor; MIGRATION.md), and NextReaction and HAS now simulate multiplex
# graphs (WP26), so the 0.1 checks that they refused one are replaced by agreement with DirectSSA.

using NetworkOutbreaks
using Graphs
using Test
using StableRNGs
using Statistics: mean, std

# Two-sample Kolmogorov–Smirnov test: (D, asymptotic p-value with Stephens' small-sample correction), as in
# test/suites/processes.jl.
function kolmogorov_sf(λ::Real)
    λ < 0.2 && return 1.0
    s = 0.0
    for j in 1:200
        s += (isodd(j) ? 2.0 : -2.0) * exp(-2 * j^2 * λ^2)
    end
    return clamp(s, 0.0, 1.0)
end

function ks_two_sample(x::AbstractVector, y::AbstractVector)
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

@testset "MultiplexGraph (DirectSSA, NextReaction, HAS)" begin
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
    spec_single = OutbreakSpec(; model, network=MultiplexGraph([g1], [1.0]),
                               initial=seed, tspan)
    spec_static = OutbreakSpec(; model, network=g1, initial=seed, tspan)
    ens_m = simulate_ensemble(spec_single; nsims=30, seed=11, algorithm=DirectSSA())
    ens_s = simulate_ensemble(spec_static; nsims=30, seed=11, algorithm=DirectSSA())
    finalR_m = mean(compartment_series(s, :R)[end] for s in ens_m)
    finalR_s = mean(compartment_series(s, :R)[end] for s in ens_s)
    @test isapprox(finalR_m, finalR_s; rtol=0.10)

    # Two distinct layers should produce more infections than a single layer.
    spec_two = OutbreakSpec(; model,
                            network=MultiplexGraph([g1, g2], [1.0, 1.0]),
                            initial=seed, tspan)
    ens_two = simulate_ensemble(spec_two; nsims=30, seed=11, algorithm=DirectSSA())
    finalR_two = mean(compartment_series(s, :R)[end] for s in ens_two)
    @test finalR_two > finalR_s

    # Validation: negative layer rates rejected.
    @test_throws ArgumentError MultiplexGraph([g1, g2], [1.0, -0.1])
    # Validation: layer node counts must agree.
    g3 = erdos_renyi(N+1, 4/N, rng=StableRNG(103))
    @test_throws ArgumentError MultiplexGraph([g1, g3], [1.0, 1.0])

    # NextReaction and HAS simulate the same process as DirectSSA on a multiplex graph (unequal layer rates):
    # the final sizes of 300 runs agree in distribution (KS), and the means within 4 standard errors.
    spec_w = OutbreakSpec(; model, network=MultiplexGraph([g1, g2], [1.5, 0.5]), initial=seed, tspan=(0.0, 60.0))
    fs(alg, s) = [compartment_series(t, :R)[end] for t in simulate_ensemble(spec_w; nsims=300, seed=s, algorithm=alg)]
    ref = fs(DirectSSA(), 21)
    for (alg, s) in ((NextReaction(), 22), (HAS(), 23))
        x = fs(alg, s)
        D, p = ks_two_sample(x, ref)
        @test p > 0.001
        @test abs(mean(x) - mean(ref)) < 4 * sqrt(std(x)^2 / length(x) + std(ref)^2 / length(ref))
    end
    # CompositionRejection does not support multiplex graphs.
    @test_throws ArgumentError simulate(spec_two; algorithm=CompositionRejection(), seed=1)
end
