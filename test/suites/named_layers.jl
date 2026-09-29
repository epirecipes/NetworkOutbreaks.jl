# Contacts on a named layer of a multiplex network (the open WP16/WP3/WP25 request in IMPLEMENTATION_NOTES_phase23.md,
# needed by :sir_mpx):
#
#   - `OutbreakTransition(…; layer)` and `MultiplexGraph(…; names)`, with their validation;
#   - `OutbreakModel(cm)` keeps `Contact.layer`, and `sample_graph(::MultiplexNetwork)` names the layers;
#   - the hazard is hazard_j(v) = rate_j · Σ_ℓ [layer_j ∈ (:all, names[ℓ])] · layer_rates[ℓ] · #{u ∈ N_ℓ(v) : state(u) ∈
#     via_j}, checked node by node against that formula written out here;
#   - a two-node chain whose absorption probabilities differ between the layered and the unlayered reading, for
#     DirectSSA, NextReaction and HAS;
#   - the multiplex SIR final size on Poisson layers with per-layer τ against the Miller–Volz fixed point
#     r = 1 − (1 − ρ) exp(−Σ_ℓ κ_ℓ T_ℓ r), T_ℓ = τ_ℓ/(τ_ℓ + γ), through simulate(cm, ::MultiplexNetwork; N);
#   - the :sir_mpx scenario runs;
#   - the errors: a named layer on a single graph, an unknown layer, MassActionSSA.

using NetworkOutbreaks
import NetworkOutbreaks as NO
using Graphs
using Test
using StableRNGs
using Statistics: mean, std

const NEC = NO.NetworkEpiCore

@testset "OutbreakTransition layer and MultiplexGraph names" begin
    tr = OutbreakTransition(:S, :I, 0.5, :infection; via = [:I], layer = :home)
    @test tr.layer === :home
    @test OutbreakTransition(:S, :I, 0.5, :infection).layer === :all
    @test_throws ArgumentError OutbreakModel([:S, :I], [false, true],
        [OutbreakTransition(:I, :S, 1.0, :spontaneous; layer = :home)])
    g = path_graph(3)
    m = MultiplexGraph([g, g], [1.0, 2.0])
    @test m.names == [:layer1, :layer2] && layer_names(m) == [:layer1, :layer2]
    @test MultiplexGraph([g, g], [1.0, 2.0]; names = [:home, :work]).names == [:home, :work]
    @test_throws ArgumentError MultiplexGraph([g, g], [1.0, 2.0]; names = [:home, :home])
    @test_throws ArgumentError MultiplexGraph([g, g], [1.0, 2.0]; names = [:home, :all])
    @test_throws ArgumentError MultiplexGraph([g, g], [1.0, 2.0]; names = [:home])
    # sample_graph names the layers after the descriptor
    mg, info = sample_graph(NEC.MultiplexNetwork(:home => RegularDegree(3), :comm => PoissonDegree(5)), 200;
                            rng = NO.stable_rng(1))
    @test mg.names == [:home, :comm] && first.(info.layers) == [:home, :comm]
    # OutbreakModel(cm) keeps the layers, and show prints them
    cm = ContactModel(:mpx; contacts = [Contact(:S, :I, :I, 0.3; layer = :home), Contact(:S, :I, :I, 0.1)],
                      transitions = [NodeTransition(:I, :R, 0.25)])
    om = OutbreakModel(cm)
    @test [t.layer for t in om.transitions] == [:home, :all, :all]
    @test occursin("on layer home", sprint(show, MIME("text/plain"), om))
end

@testset "the layered hazard is Σ over the layers of the transition" begin
    # compartments S, E, I, Q; contacts S+I→E on :all, S+I→Q on :home (tracing), S+E→E on :work, E+I→Q on :home
    C = [:S, :E, :I, :Q]
    trs = [OutbreakTransition(:S, :E, 0.7, :infection; via = [:I]),
           OutbreakTransition(:S, :Q, 0.3, :contact_trace; via = [:I], layer = :home),
           OutbreakTransition(:S, :E, 0.2, :infection; via = [:E], layer = :work),
           OutbreakTransition(:E, :Q, 1.1, :contact_trace; via = [:I, :E], layer = :home),
           OutbreakTransition(:E, :I, 0.5, :spontaneous)]
    model = OutbreakModel(C, [false, true, true, false], trs)
    n = 60
    rng = StableRNG(7)
    layers = [erdos_renyi(n, 0.1; rng), erdos_renyi(n, 0.08; rng), erdos_renyi(n, 0.12; rng)]
    w = [1.5, 0.0, 0.4]                             # a zero-weight layer contributes nothing
    for names in ([:home, :work, :school], [:school, :home, :work])
        mg = MultiplexGraph(layers, w; names)
        rm = NO._RunModel(model, mg)
        @test !rm.all_layers && length(rm.class_on) == 3
        tally = NO._tally_buffer(rm)
        @test length(tally) == 3 * length(C)
        for trial in 1:5
            node_state = rand(rng, 1:4, n)
            for v in 1:n
                expect = 0.0
                for (j, tr) in pairs(trs)
                    tr.type === :spontaneous && continue
                    model.index_of[tr.from] == node_state[v] || continue
                    via = [model.index_of[x] for x in tr.via]
                    for ℓ in eachindex(layers)
                        (tr.layer === :all || names[ℓ] === tr.layer) || continue
                        expect += tr.rate * w[ℓ] * count(u -> node_state[u] in via, neighbors(layers[ℓ], v))
                    end
                end
                @test NO._contact_hazard!(tally, v, mg.layers, mg.layer_rates, node_state, rm) ≈ expect atol = 1e-12
            end
        end
    end
    # a model without named layers keeps the single all-layer class (the 0.1 fast path) on any network
    sir = OutbreakModel([:S, :I, :R], [false, true, false],
                        [OutbreakTransition(:S, :I, 1.0, :infection), OutbreakTransition(:I, :R, 1.0, :spontaneous)])
    for net in (MultiplexGraph(layers, w), layers[1])
        rm = NO._RunModel(sir, net)
        @test rm.all_layers && length(NO._tally_buffer(rm)) == 3
    end
end

@testset "a contact on one layer ignores the neighbours of the other layers" begin
    # node 1 (S) is linked to node 2 (I) only in :comm; the only contact acts on :home, so node 1 is never infected
    home = SimpleGraph(3); add_edge!(home, 2, 3)
    comm = SimpleGraph(3); add_edge!(comm, 1, 2)
    mg = MultiplexGraph([home, comm], [1.0, 1.0]; names = [:home, :comm])
    model = OutbreakModel([:S, :I, :R], [false, true, false],
        [OutbreakTransition(:S, :I, 50.0, :infection; via = [:I], layer = :home),
         OutbreakTransition(:I, :R, 1.0, :spontaneous)])
    spec = OutbreakSpec(model = model, network = mg, initial = SeedNodes(:I => [2]), tspan = (0.0, 100.0))
    for alg in (DirectSSA(), NextReaction(), HAS())
        runs = [simulate(spec; algorithm = alg, seed = s) for s in 1:60]
        @test all(tr -> tr.final_infection_counts[1] == 0, runs)
        # node 3 is reached through :home with probability 50/51 (expected 58.8 of 60)
        @test count(tr -> tr.final_infection_counts[3] == 1, runs) >= 54
    end
end

@testset "two-node chain: the absorption probability of the layered model" begin
    # node 1 (S) and node 2 (I) are linked in both layers, with layer rates 2 (:home) and 0.5 (:comm). S+I→2I acts on
    # :home at rate a, S+I→V on :comm at rate b, I→R at rate γ. Node 1 is infected with probability
    # 2a/(2a + 0.5b + γ) = 4/7 for a = b = γ = 1 (reading both contacts on every layer would give 5/12).
    g = SimpleGraph(2); add_edge!(g, 1, 2)
    mg = MultiplexGraph([g, copy(g)], [2.0, 0.5]; names = [:home, :comm])
    model = OutbreakModel([:S, :I, :R, :V], [false, true, false, false],
        [OutbreakTransition(:S, :I, 1.0, :infection; via = [:I], layer = :home),
         OutbreakTransition(:S, :V, 1.0, :contact_trace; via = [:I], layer = :comm),
         OutbreakTransition(:I, :R, 1.0, :spontaneous)]; infected = [:I])
    spec = OutbreakSpec(model = model, network = mg, initial = SeedNodes(:I => [2]), tspan = (0.0, Inf))
    p = 4 / 7
    M = 20_000
    for (k, alg) in enumerate((DirectSSA(), NextReaction(), HAS()))
        hits = count(r -> simulate(spec; algorithm = alg, seed = 10^6 * k + r).final_infection_counts[1] == 1, 1:M)
        se = sqrt(p * (1 - p) / M)
        @test abs(hits / M - p) < 4 * se
        @test abs(hits / M - 5 / 12) > 20 * se
    end
end

@testset "multiplex SIR with per-layer τ: final size against the Miller–Volz fixed point (N = 10⁴)" begin
    τh, τc, γ, ρ = 0.6, 0.15, 1.0, 0.01
    cm = ContactModel(:mpx; contacts = [Contact(:S, :I, :I, τh; layer = :home), Contact(:S, :I, :I, τc; layer = :comm)],
                      transitions = [NodeTransition(:I, :R, γ)])
    net = NEC.MultiplexNetwork(:home => PoissonDegree(3), :comm => PoissonDegree(5))
    s = 3 * τh / (τh + γ) + 5 * τc / (τc + γ)
    r = 0.5
    for _ in 1:10_000
        r = 1 - (1 - ρ) * exp(-s * r)
    end
    @test r ≈ 1 - (1 - ρ) * exp(-s * r) atol = 1e-14
    @test final_size(cm, net, Dict{Symbol, Float64}(); initial = SeedFraction(:I => ρ)) ≈ r atol = 1e-8
    for alg in (NextReaction(), HAS())
        ens = simulate(cm, net; N = 10_000, initial = SeedFraction(:I => ρ), tspan = (0.0, 400.0), nsims = 30,
                       algorithm = alg, seed = 11)
        fs = final_size(ens)
        @test all(>(0.2), fs)
        se = std(fs) / sqrt(length(fs))
        @test abs(mean(fs) - r) < 4 * se + 2e-3      # 2e-3: the O(1/N) finite-size and erasure bias
    end
end

@testset "the :sir_mpx scenario runs" begin
    sc = derive(scenario(:sir_mpx); nsims = 4, N = 2000)
    ens = simulate(sc)
    @test ens isa OutbreakEnsemble && length(ens) == 4
    @test [t.layer for t in ens.trajectories[1].model.transitions[1:2]] == [:home, :comm]
    @test all(f -> 0 < f <= 1, final_size(ens))
end

@testset "named-layer errors" begin
    cm = ContactModel(:lay; contacts = [Contact(:S, :I, :I, 0.5; layer = :home)], transitions = [NodeTransition(:I, :R, 1.0)])
    om = OutbreakModel(cm)
    g = random_regular_graph(50, 3; rng = StableRNG(1))
    for alg in (DirectSSA(), NextReaction(), HAS())
        @test_throws ArgumentError simulate(OutbreakSpec(model = om, network = g, initial = SeedCount(:I => 2),
                                                         tspan = (0.0, 5.0)); algorithm = alg, seed = 1)
        @test_throws ArgumentError simulate(OutbreakSpec(model = om, network = MultiplexGraph([g], [1.0]),
                                                         initial = SeedCount(:I => 2), tspan = (0.0, 5.0));
                                            algorithm = alg, seed = 1)
    end
    # the high-level route refuses it up front (NetworkEpiCore's admissibility check names the layer)
    @test_throws AdmissibilityError simulate(cm, ConfigurationNetwork(PoissonDegree(3)); N = 50, initial = SeedCount(:I => 2),
                                        tspan = (0.0, 5.0), nsims = 1, seed = 1)
    @test_throws AdmissibilityError simulate(cm, WellMixed(3.0); N = 50, initial = SeedCount(:I => 2),
                                        tspan = (0.0, 5.0), nsims = 1, seed = 1)
end
