# NetworkOutbreaks adopts NetworkEpiCore (WP16; DESIGN_NetworkEpiCore.md §A.5, §A.7, §G.2, §J.2, §J.6–§J.8).
#
# Owner: WP16. Replaces the 0.1 adapter suite (00_legacy_adapters.jl), which needed the EdgeBasedModels and
# NodeBasedModels package extensions that 0.2 deletes. Acceptance (§G.2 WP16):
#   - OutbreakModel(cm) for SIR, SEIR, SEAIR, two-strain, vaccination and tracing (structure, hand-built
#     equivalence, and the edge-based large-N limit as the reference);
#   - NO's test target no longer loads ModelingToolkit (nor EdgeBasedModels / NodeBasedModels);
#   - the MultiplexGraph rename;
#   - SeedFraction === NetworkEpiCore.SeedFraction;
#   - a smoke test of simulate(model, ConfigurationNetwork(PoissonDegree(5)); N = 1000, …).
# Plus the verified issues N01 (seeding; the skeptic's corrected cases), N05 (the MultiplexGraph docstring), E28
# (seeding correspondence) and the WP3 requests (structural infection status §J.8, traj.seed, state_at at t0, strict
# TimeVaryingNetwork updates).

using NetworkOutbreaks
using NetworkEpiCore
using Graphs
using Test
using StableRNGs
using Statistics: mean, std
import TOML

const NO = NetworkOutbreaks
const NEC = NetworkEpiCore

fields(m::OutbreakModel) = (compartments = m.compartments, infectious = m.infectious, susceptible = m.susceptible,
    transitions = [(t.from, t.to, t.rate, t.type, t.via, t.layer) for t in m.transitions])

init(m, s, n) = NO.initial_state(OutbreakSpec(model = m, network = SimpleGraph(n), initial = s, tspan = (0.0, 1.0)),
                                 StableRNG(1))
cnt(m, s, n, c) = count(==(m.index_of[c]), init(m, s, n))

# ---------------------------------------------------------------------------------------------------------------
# Package: NetworkEpiCore bindings, no EdgeBasedModels / NodeBasedModels / ModelingToolkit
# ---------------------------------------------------------------------------------------------------------------

@testset "NetworkEpiCore bindings and dependencies" begin
    for n in (:SeedSpec, :SeedFraction, :SeedCount, :SeedNodes, :final_size, :reinfection_histogram,
              :ContactModel, :sir_model, :ConfigurationNetwork, :PoissonDegree, :compartment)
        @test Base.isexported(NO, n)
        @test getfield(NO, n) === getfield(NEC, n)
    end
    @test SeedFraction === NEC.SeedFraction
    # every NetworkEpiCore export that NetworkOutbreaks exports is the same binding (design §A.8)
    for n in names(NO)
        (Base.isexported(NO, n) && isdefined(NEC, n) && n !== :NetworkOutbreaks) || continue
        @test getfield(NO, n) === getfield(NEC, n)
    end
    @test isempty(Test.detect_ambiguities(NO))
    # every export is documented
    undocumented = [n for n in names(NO) if Base.isexported(NO, n) && n !== :NetworkOutbreaks &&
                    !Base.Docs.hasdoc(NO, n) && !(isdefined(NEC, n) && getfield(NO, n) === getfield(NEC, n))]
    @test isempty(undocumented)

    # The test target and the package no longer depend on EdgeBasedModels, NodeBasedModels or ModelingToolkit
    root = pkgdir(NO)
    proj = TOML.parsefile(joinpath(root, "Project.toml"))
    heavy = ("EdgeBasedModels", "NodeBasedModels", "ModelingToolkit", "ModelingToolkitBase", "Symbolics", "Catalyst")
    for section in ("deps", "weakdeps", "extras")
        @test isempty(intersect(keys(get(proj, section, Dict())), heavy))
    end
    @test isempty(intersect(get(get(proj, "targets", Dict()), "test", String[]), heavy))
    @test !haskey(proj, "extensions")
    @test !isdir(joinpath(root, "ext"))
    @test !isfile(joinpath(root, "test", "test_eon_patterns.jl"))
    @test haskey(proj["deps"], "NetworkEpiCore")
    @test VersionNumber(proj["version"]) >= v"0.2.0"
    @test !any(m -> nameof(m) in (:ModelingToolkit, :ModelingToolkitBase, :EdgeBasedModels, :NodeBasedModels),
               values(Base.loaded_modules))
end

# ---------------------------------------------------------------------------------------------------------------
# MultiplexGraph (was MultiplexNetwork, now NetworkEpiCore's descriptor)
# ---------------------------------------------------------------------------------------------------------------

@testset "MultiplexGraph rename" begin
    g1 = random_regular_graph(50, 3; rng = StableRNG(1))
    g2 = erdos_renyi(50, 0.1; rng = StableRNG(2))
    mg = MultiplexGraph([g1, g2], [2.0, 1.0])
    @test mg isa AbstractContactNetwork && nv(mg) == 50 && ne(mg) == ne(g1) + ne(g2)
    @test Base.isexported(NO, :MultiplexGraph)
    # MultiplexNetwork is NetworkEpiCore's descriptor, re-exported (the 0.1 name of the container is not an alias any
    # more), so `using NetworkOutbreaks` alone resolves it and it is unambiguous next to `using NetworkEpiCore`
    @test Base.isexported(NO, :MultiplexNetwork) && NO.MultiplexNetwork === NEC.MultiplexNetwork
    @test MultiplexNetwork === NEC.MultiplexNetwork
    @test MultiplexNetwork(:home => RegularDegree(3), :work => PoissonDegree(2.0)) isa NetworkDescriptor
    # the 0.1 container call MultiplexNetwork(graphs, rates) fails with a message that migrates it to MultiplexGraph
    for call in (() -> MultiplexNetwork([g1, g2], [2.0, 1.0]), () -> NO.MultiplexNetwork([g1], [1.0]))
        err = try
            call()
        catch e
            e
        end
        @test err isa MethodError
        msg = sprint(showerror, err)
        @test occursin("NetworkOutbreaks 0.1", msg) && occursin("MultiplexGraph(layers, layer_rates)", msg)
    end
    # … and only that call: other MethodErrors carry no hint
    other = try
        MultiplexNetwork(1, 2)
    catch e
        e
    end
    @test other isa MethodError && !occursin("MultiplexGraph", sprint(showerror, other))
    @test_throws ArgumentError MultiplexGraph([g1, g2], [1.0, -0.1])
    @test_throws ArgumentError MultiplexGraph([g1, SimpleGraph(10)], [1.0, 1.0])
    @test_throws ArgumentError MultiplexGraph(SimpleGraph{Int}[], Float64[])
    # N05 #3: the docstring example is valid code (it used SeedFraction(0.01))
    doc = string(@doc MultiplexGraph)
    @test !occursin("SeedFraction(0.01)", doc) && occursin("SeedFraction(:I => 0.01)", doc)
    # the multiplex container still simulates (DirectSSA)
    sir = OutbreakModel(sir_model(), Dict(:τ => 0.3, :γ => 0.5))
    tr = simulate(OutbreakSpec(model = sir, network = mg, initial = SeedFraction(:I => 0.1), tspan = (0.0, 5.0));
                  seed = 3)
    @test sum(tr.counts[:, end]) == 50
    # m3 under the new name: directed layers are rejected
    dlayers = [SimpleDiGraph(path_graph(50)), SimpleDiGraph(path_graph(50))]
    @test_throws ArgumentError simulate(OutbreakSpec(model = sir, network = MultiplexGraph(dlayers, [1.0, 1.0]),
                                                     initial = SeedNodes(:I => [1]), tspan = (0.0, 5.0)); seed = 1)
end

# ---------------------------------------------------------------------------------------------------------------
# OutbreakModel(cm::ContactModel, p)
# ---------------------------------------------------------------------------------------------------------------

const P_SEAIR = Dict(:τI => 1 / 6, :τA => 1 / 12, :p => 0.6, :σ => 1 / 5, :γ => 1 / 4)

tracing_model() = ContactModel(:trace; contacts = [Contact(:S, :I, :I, :τ), Contact(:S, :D, :Q, :α)],
                               transitions = [NodeTransition(:I, :D, :γ), NodeTransition(:D, :R, :δ)])
quarantine_model() = ContactModel(:quar; contacts = [Contact(:S, :I, :E, :τ), Contact(:E, :I, :Eq, :α),
                                                      Contact(:S, :Iq, :E, :τq)],
    transitions = [NodeTransition(:E, :I, :σ), NodeTransition(:I, :R, :γ), NodeTransition(:Eq, :Iq, :σ),
                   NodeTransition(:Iq, :R, :γ)])
awareness_model() = ContactModel(:aware; contacts = [Contact(:S, :Sa, :Sa, :κ), Contact(:S, :I, :E, :τ),
                                                    Contact(:Sa, :I, :E, :τa)],
                                 transitions = [NodeTransition(:E, :I, :σ), NodeTransition(:I, :R, :γ)],
                                 susceptible = [:S, :Sa])

@testset "OutbreakModel(cm): structure (§A.5)" begin
    T = OutbreakTransition
    cases = [
        ("SIR", sir_model(), Dict(:τ => 1.5, :γ => 1.0),
         OutbreakModel([:S, :I, :R], [false, true, false],
                       [T(:S, :I, 1.5, :infection; via = [:I]), T(:I, :R, 1.0, :spontaneous)])),
        ("SEIR", seir_model(), Dict(:τ => 0.3, :σ => 0.5, :γ => 0.2),
         OutbreakModel([:S, :E, :I, :R], [false, false, true, false],
                       [T(:S, :E, 0.3, :infection; via = [:I]), T(:E, :I, 0.5, :spontaneous),
                        T(:I, :R, 0.2, :spontaneous)])),
        ("SEAIR (branching, two infectors)", seair_model(), P_SEAIR,
         OutbreakModel([:S, :E, :I, :A, :R], [false, false, true, true, false],
                       [T(:S, :E, 1 / 6, :infection; via = [:I]), T(:S, :E, 1 / 12, :infection; via = [:A]),
                        T(:E, :I, 0.6 * 0.2, :spontaneous), T(:E, :A, (1 - 0.6) * 0.2, :spontaneous),
                        T(:I, :R, 0.25, :spontaneous), T(:A, :R, 0.25, :spontaneous)])),
        ("two strains", twostrain_model(), Dict(:τ1 => 1 / 6, :τ2 => 1 / 5, :γ => 1 / 4),
         OutbreakModel([:S, :I1, :I2, :R], [false, true, true, false],
                       [T(:S, :I1, 1 / 6, :infection; via = [:I1]), T(:S, :I2, 1 / 5, :infection; via = [:I2]),
                        T(:I1, :R, 0.25, :spontaneous), T(:I2, :R, 0.25, :spontaneous)])),
        ("vaccination (exit)", sirv_model(), Dict(:τ => 1 / 6, :γ => 1 / 4, :ν => 0.02),
         OutbreakModel([:S, :I, :R, :V], [false, true, false, false],
                       [T(:S, :I, 1 / 6, :infection; via = [:I]), T(:I, :R, 0.25, :spontaneous),
                        T(:S, :V, 0.02, :spontaneous)])),
        # §B.8 tracing: S + D → Q + D is a contact whose product is not infectious. The tracer D is an infector.
        ("tracing", tracing_model(), Dict(:τ => 0.3, :α => 0.2, :γ => 0.25, :δ => 0.5),
         OutbreakModel([:S, :I, :Q, :D, :R], [false, true, false, true, false],
                       [T(:S, :I, 0.3, :infection; via = [:I]), T(:S, :Q, 0.2, :infection; via = [:D]),
                        T(:I, :D, 0.25, :spontaneous), T(:D, :R, 0.5, :spontaneous)])),
        # quarantine of exposed contacts: E + I → Eq + I is a node contact, hence status-preserving (§J.8); the
        # quarantined still transmit a little (S + Iq → E)
        ("quarantine of exposed contacts", quarantine_model(),
         Dict(:τ => 0.3, :α => 0.2, :τq => 0.05, :γ => 0.25, :σ => 0.5),
         OutbreakModel([:S, :E, :I, :Eq, :Iq, :R], [false, false, true, false, true, false],
                       [T(:S, :E, 0.3, :infection; via = [:I]), T(:E, :Eq, 0.2, :contact_trace; via = [:I]),
                        T(:S, :E, 0.05, :infection; via = [:Iq]), T(:E, :I, 0.5, :spontaneous),
                        T(:I, :R, 0.25, :spontaneous), T(:Eq, :Iq, 0.5, :spontaneous),
                        T(:Iq, :R, 0.25, :spontaneous)])),
        # awareness S + Sa → 2Sa between susceptible classes: a status-preserving contact; Sa is not infectious
        ("awareness", awareness_model(), Dict(:τ => 0.3, :κ => 0.2, :τa => 0.05, :γ => 0.25, :σ => 0.5),
         OutbreakModel([:S, :Sa, :E, :I, :R], [false, false, false, true, false],
                       [T(:S, :Sa, 0.2, :contact_trace; via = [:Sa]), T(:S, :E, 0.3, :infection; via = [:I]),
                        T(:Sa, :E, 0.05, :infection; via = [:I]), T(:E, :I, 0.5, :spontaneous),
                        T(:I, :R, 0.25, :spontaneous)]; susceptible = [:S, :Sa])),
    ]
    for (label, cm, p, hand) in cases
        @testset "$label" begin
            m = OutbreakModel(cm, p)
            @test fields(m) == fields(hand)
            @test m.name === nameof(cm)
            # transitions are in NetworkEpiCore reaction order (contacts, then node transitions)
            @test length(m.transitions) == length(contacts(cm)) + length(node_transitions(cm))
            # identical sample paths to the hand-built model
            g = random_regular_graph(300, 4; rng = StableRNG(5))
            initial = SeedCount(hand.compartments[findfirst(hand.infectious)] => 5)
            a = simulate(OutbreakSpec(; model = m, network = g, initial, tspan = (0.0, 30.0)); seed = 11,
                         algorithm = NextReaction())
            b = simulate(OutbreakSpec(; model = hand, network = g, initial, tspan = (0.0, 30.0)); seed = 11,
                         algorithm = NextReaction())
            @test a.times == b.times && a.counts == b.counts
            @test final_size(a) == final_size(b)
        end
    end

    # removals X → ∅ go to the absorbing sink :removed (§J.2)
    sird = ContactModel(:sird; contacts = [Contact(:S, :I, :I, :τ)],
                        transitions = [NodeTransition(:I, :R, :γ), NodeTransition(:I, nothing, :μ)])
    m = OutbreakModel(sird, Dict(:τ => 0.3, :γ => 0.25, :μ => 0.1))
    @test m.compartments == [:S, :I, :R, NO.REMOVED_SINK] && NO.REMOVED_SINK === :removed
    @test (m.transitions[3].from, m.transitions[3].to, m.transitions[3].type) == (:I, :removed, :spontaneous)
    @test m.infectious == [false, true, false, false]
    tr = simulate(OutbreakSpec(model = m, network = random_regular_graph(400, 5; rng = StableRNG(2)),
                               initial = SeedFraction(:I => 0.05), tspan = (0.0, 200.0)); seed = 1)
    @test all(sum(tr.counts; dims = 1) .== 400)                         # conservation with the sink
    @test tr.counts[4, end] > 0 && tr.counts[2, end] == 0
    clash = ContactModel(:clash; contacts = [Contact(:S, :I, :I, :τ)],
                         transitions = [NodeTransition(:I, :removed, :γ), NodeTransition(:I, nothing, :μ)])
    @test_throws ArgumentError OutbreakModel(clash, Dict(:τ => 0.3, :γ => 0.25, :μ => 0.1))

    # reinfection counting (a T_net model): infections are the Sus → I contacts, every susceptible class is Σ
    r2 = OutbreakModel(with_reinfection_counting(sis_model(), 2), Dict(:τ => 0.8, :γ => 0.25))
    @test r2.susceptible == [:S_0, :S_1, :S_2]
    @test all(t -> t.type === (t.from in r2.susceptible ? :infection : :spontaneous), r2.transitions)
    tr = simulate(OutbreakSpec(model = r2, network = random_regular_graph(300, 4; rng = StableRNG(3)),
                               initial = SeedFraction(:I_1 => 0.05), tspan = (0.0, 40.0)); seed = 2)
    h = reinfection_histogram(tr; L = 2)
    @test sum(h) == 300
    # nodes in X_p have been infected min(p, L) times, or more (the counting saturates at L = 2)
    @test h[1] == tr.counts[r2.index_of[:S_0], end]
    @test h[2] == tr.counts[r2.index_of[:S_1], end] + tr.counts[r2.index_of[:I_1], end]

    # parameters: model defaults fill in, missing ones are an error, numeric models pass through
    @test OutbreakModel(sir_model(τ = 0.4, γ = 0.2)).transitions[1].rate == 0.4
    @test_throws ArgumentError OutbreakModel(sir_model(), Dict(:β => 1.5, :γ => 1.0))   # τ is missing
    hand = last(first(cases))
    @test OutbreakModel(hand) === hand
    @test OutbreakModel(hand, Dict{Symbol,Float64}()) === hand
    @test_throws ArgumentError OutbreakModel(hand, Dict(:τ => 1.0))
    # not admissible for the stochastic back end: time-dependent rates
    @test_throws AdmissibilityError OutbreakModel(sir_model(τ = :(τ0 * exp(-t)), γ = 0.2), Dict(:τ0 => 1.0))
    # a contact on a named layer keeps its layer (checked against `network` when one is given; the samplers check it
    # against the MultiplexGraph layer names, test/suites/named_layers.jl)
    layered = ContactModel(:layered; contacts = [Contact(:S, :I, :I, :τ; layer = :home)],
                           transitions = [NodeTransition(:I, :R, :γ)])
    @test OutbreakModel(layered, Dict(:τ => 0.3, :γ => 0.2)).transitions[1].layer === :home
    mpx = NEC.MultiplexNetwork(:home => RegularDegree(3), :work => PoissonDegree(2.0))
    @test OutbreakModel(layered, Dict(:τ => 0.3, :γ => 0.2); network = mpx).transitions[1].layer === :home
    nohome = NEC.MultiplexNetwork(:school => RegularDegree(3), :work => PoissonDegree(2.0))
    @test_throws AdmissibilityError OutbreakModel(layered, Dict(:τ => 0.3, :γ => 0.2); network = nohome)
    @test_throws AdmissibilityError OutbreakModel(layered, Dict(:τ => 0.3, :γ => 0.2);
                                                  network = ConfigurationNetwork(PoissonDegree(3)))
    # the same checks against the layer names of a MultiplexGraph
    g3 = path_graph(4)
    @test OutbreakModel(layered, Dict(:τ => 0.3, :γ => 0.2);
                        network = MultiplexGraph([g3, g3], [1.0, 1.0]; names = [:home, :work])).transitions[1].layer === :home
    @test_throws ArgumentError OutbreakModel(layered, Dict(:τ => 0.3, :γ => 0.2);
                                             network = MultiplexGraph([g3, g3], [1.0, 1.0]))
    @test_throws AdmissibilityError OutbreakModel(layered, Dict(:τ => 0.3, :γ => 0.2); network = g3)
end

# Any model that `contact_model` accepts goes through the fallback (this replaces the 0.1 extensions).
struct _ToyLegacyModel
    τ::Float64
end
NetworkEpiCore.contact_model(m::_ToyLegacyModel) = sir_model(τ = m.τ, γ = 0.5)

@testset "OutbreakModel(x, p) through contact_model" begin
    m = OutbreakModel(_ToyLegacyModel(0.7))
    @test fields(m) == fields(OutbreakModel(sir_model(), Dict(:τ => 0.7, :γ => 0.5)))
    @test_throws ArgumentError OutbreakModel(42)
    @test_throws ArgumentError OutbreakModel("sir", Dict(:τ => 1.0))
end

@testset "rate conventions use the nominal mean degree (§A.5, §B.6)" begin
    fd = ContactModel(:sir_fd; contacts = [Contact(:S, :I, :I, :β)], transitions = [NodeTransition(:I, :R, :γ)],
                      convention = FrequencyDependent())
    p = Dict(:β => 0.5, :γ => 0.25)
    net = ConfigurationNetwork(PoissonDegree(5.0))
    @test OutbreakModel(fd, p; network = net).transitions[1].rate ≈ 0.1               # τ = β/⟨k⟩
    @test_throws ArgumentError OutbreakModel(fd, p)                                   # needs a network
    g = random_regular_graph(100, 4; rng = StableRNG(1))
    @test OutbreakModel(fd, p; network = g).transitions[1].rate ≈ 0.125               # the realised ⟨k⟩ of g
    ens = simulate(fd, net; N = 200, p, initial = SeedFraction(:I => 0.05), tspan = (0.0, 5.0), seed = 1)
    @test ens.spec.model.transitions[1].rate ≈ 0.1
end

# ---------------------------------------------------------------------------------------------------------------
# Structural infection status (§J.8): final_size counts infections decided by the typing
# ---------------------------------------------------------------------------------------------------------------

@testset "structural infection status (§J.8)" begin
    # The infected compartments of OutbreakModel(cm) are NetworkEpiCore's infected_species(cm), the one implementation
    # of §J.8 (§L.6) behind with_reinfection_counting and the edge-based :cumulative accumulator; the contacts NO
    # counts as infections (an :infection contact with an infectious catalyst from a non-infected into an infected
    # compartment) are exactly NetworkEpiCore's infections (its internal `_is_infection` rule, cross-checked here).
    # The status is structural: it is the same with every rate positive, with each rate set to 0 in turn, and with
    # all of them 0 (a contact with rate 0 still has its infector, §A.5).
    for cm in (sir_model(), seir_model(), seair_model(), twostrain_model(), sirv_model(), tracing_model(),
               quarantine_model(), awareness_model(), with_reinfection_counting(sis_model(), 2))
        Σ = Set(susceptible_species(cm))
        chain = NEC._infection_chain(cm)
        nec = [c.name for c in contacts(cm) if NEC._is_infection(c, Σ, chain)]
        ks = rate_parameters(cm)
        positive = OutbreakModel(cm, Dict(k => 0.3 for k in ks))
        infectors = Set(c.infector for c in contacts(cm))                 # §A.5: the flags mark the infectors
        @test positive.infectious == Bool[X in infectors && !(X in Σ) for X in positive.compartments]
        @test positive.compartments[NO._infected_mask(positive)] == infected_species(cm)
        # on this corpus the structural rule of hand-built models agrees with the typing
        @test NO._structural_infected(positive.compartments, positive.infectious, positive.transitions,
                                      positive.index_of) == positive.infected
        variants = [Dict(k => 0.3 for k in ks); [Dict(k => (k === z ? 0.0 : 0.3) for k in ks) for z in ks];
                    Dict(k => 0.0 for k in ks)]
        for p in variants
            m = OutbreakModel(cm, p)
            mask = NO._infected_mask(m)
            no = [c.name for (c, t) in zip(contacts(cm), m.transitions)
                  if t.type === :infection && any(v -> m.infectious[m.index_of[v]], t.via) &&
                     !mask[m.index_of[t.from]] && mask[m.index_of[t.to]]]
            @test no == nec
            @test m.infectious == positive.infectious
            @test m.compartments[mask] == infected_species(cm)
        end
    end
    # Where the structural rule cannot see the typing: recovery I → Sa into a susceptible class whose only exit is
    # importation Sa → E. The ContactModel declares Sa susceptible, so NetworkEpiCore, and hence OutbreakModel(cm),
    # does not count Sa as infected and a re-importation is a new infection; the structural rule of a hand-built
    # model reads Sa as a latent phase (the limit in its docstring), unless `infected` is given.
    imp = ContactModel(:recover_import; contacts = [Contact(:S, :I, :E, :τ)],
                       transitions = [NodeTransition(:E, :I, :σ), NodeTransition(:I, :Sa, :γ),
                                      NodeTransition(:Sa, :E, :η)], susceptible = [:S, :Sa])
    mi = OutbreakModel(imp, Dict(:τ => 0.3, :σ => 0.5, :γ => 0.25, :η => 0.2))
    @test mi.compartments == [:S, :Sa, :E, :I] && mi.susceptible == [:S, :Sa]
    @test mi.compartments[NO._infected_mask(mi)] == infected_species(imp) == [:E, :I]
    handi = OutbreakModel(mi.compartments, mi.infectious, mi.transitions)
    @test handi.compartments[NO._infected_mask(handi)] == [:Sa, :E, :I]
    @test OutbreakModel(mi.compartments, mi.infectious, mi.transitions; infected = [:E, :I]).infected == mi.infected
    gi = random_regular_graph(1000, 5; rng = StableRNG(31))
    run(m) = simulate(OutbreakSpec(model = m, network = gi, initial = SeedFraction(:I => 0.05), tspan = (0.0, 60.0));
                      seed = 3, algorithm = NextReaction())
    a, b = run(mi), run(handi)
    @test a.counts == b.counts                                          # the same dynamics …
    @test maximum(a.final_infection_counts) >= 2                        # … re-importations are reinfections
    @test maximum(b.final_infection_counts) == 1                        # (the structural rule never counts them)
    @test final_size(a) == final_size(b)
    # Peer vaccination S + V → 2V: a non-susceptible catalyst is an infector, so the §J.8 rule counts V as infected;
    # `infected` counts it otherwise
    pv = ContactModel(:peer_vacc; contacts = [Contact(:S, :I, :I, :τ), Contact(:S, :V, :V, :κ)],
                      transitions = [NodeTransition(:I, :R, :γ)])
    ppv = Dict(:τ => 0.2, :κ => 0.3, :γ => 0.25)
    @test OutbreakModel(pv, ppv).compartments[OutbreakModel(pv, ppv).infected] == infected_species(pv) == [:I, :V]
    mv = OutbreakModel(pv, ppv; infected = [:I])
    @test mv.compartments[mv.infected] == [:I] && mv.infectious == [false, true, true, false]
    tr = simulate(OutbreakSpec(model = mv, network = gi, initial = SeedNodes(:I => 1:10, :V => 11:20),
                               tspan = (0.0, 100.0)); seed = 5)
    c = tr.counts[:, end]
    @test c[mv.index_of[:V]] > 20                                       # vaccination spread …
    @test final_size(tr) == (c[mv.index_of[:I]] + c[mv.index_of[:R]]) / 1000   # … and is not an infection
    # the `infected` keyword is validated
    T = OutbreakTransition
    trs = [T(:S, :I, 0.5, :infection), T(:I, :R, 0.25, :spontaneous)]
    @test OutbreakModel([:S, :I, :R], [false, true, false], trs).infected == [false, true, false]
    @test_throws ArgumentError OutbreakModel([:S, :I, :R], [false, true, false], trs; infected = [:S])   # susceptible
    @test_throws ArgumentError OutbreakModel([:S, :I, :R], [false, true, false], trs; infected = [:Z])
    @test_throws ArgumentError OutbreakModel([:S, :I, :R], [false, true, false], trs; infected = [:I, :I])
    @test_throws ArgumentError OutbreakModel([:S, :I, :R], [false, true, false], trs; infected = ["I"])
    @test_throws ArgumentError OutbreakModel(sir_model(), Dict(:τ => 0.3, :γ => 0.25); infected = [:S])
    @test occursin("infected:     I", sprint(show, MIME"text/plain"(), OutbreakModel(sir_model(), Dict(:τ => 0.3, :γ => 0.25))))
    g = random_regular_graph(2000, 5; rng = StableRNG(8))
    # Tracing: Q (traced susceptibles) is never infected; everyone ever infected passed through I.
    m = OutbreakModel(tracing_model(), Dict(:τ => 0.3, :α => 0.4, :γ => 0.25, :δ => 0.5))
    @test NO._infected_mask(m) == [false, true, false, true, false]
    for alg in (DirectSSA(), NextReaction(), HAS())
        tr = simulate(OutbreakSpec(model = m, network = g, initial = SeedFraction(:I => 0.01), tspan = (0.0, 80.0));
                      seed = 4, algorithm = alg)
        c = tr.counts[:, end]
        @test c[m.index_of[:Q]] > 0
        @test final_size(tr) == (c[m.index_of[:I]] + c[m.index_of[:D]] + c[m.index_of[:R]]) / 2000
    end
    # Quarantine of exposed contacts: E stays infected, so E seeds count (with :infection it would be a barrier).
    # (τ, α, τq and σ are 0, so the E seeds neither progress nor infect; a rate of 0 leaves the infection status
    # unchanged: I and Iq are the infectors of their contacts, §A.5, §J.8)
    mq = OutbreakModel(quarantine_model(), Dict(:τ => 0.0, :α => 0.0, :τq => 0.0, :γ => 0.25, :σ => 0.0))
    @test NO._infected_mask(mq) == [false, true, true, true, true, false]
    @test NO._infected_mask(mq)[mq.index_of[:E]]
    tr = simulate(OutbreakSpec(model = mq, network = g, initial = SeedFraction(:E => 0.05), tspan = (0.0, 1.0));
                  seed = 1)
    @test final_size(tr) == 0.05
    relabelled = OutbreakModel(mq.compartments, mq.infectious,
                               [OutbreakTransition(t.from, t.to, t.rate, t.type === :contact_trace ? :infection : t.type;
                                                   via = t.via) for t in mq.transitions])
    @test !NO._infected_mask(relabelled)[mq.index_of[:E]]              # the misclassification §J.8 avoids
    # Awareness: aware susceptibles are neither infectious nor infected (τ = τa = 0: no infections at all).
    ma = OutbreakModel(awareness_model(), Dict(:τ => 0.0, :κ => 1.0, :τa => 0.0, :γ => 0.25, :σ => 0.5))
    @test ma.infectious == [false, false, false, true, false]
    @test !any(NO._infected_mask(ma)[[ma.index_of[:S], ma.index_of[:Sa]]])
    tr = simulate(OutbreakSpec(model = ma, network = g, initial = SeedNodes(:I => 1:10, :Sa => 11:100),
                               tspan = (0.0, 20.0)); seed = 2)
    @test tr.counts[ma.index_of[:Sa], end] > 1000                       # awareness spread …
    @test final_size(tr) == 10 / 2000                                  # … and is not an infection
end

# A contact rate of 0 changes the dynamics, never the infection status (§A.5: the infectious flags mark the
# infectors; §J.8: status is structural; §E.2: final_size includes the seeds).
@testset "a contact rate of 0 keeps the infection status (§A.5, §J.8)" begin
    g = random_regular_graph(1000, 4; rng = StableRNG(21))
    # SIR with τ = 0: the seeds are infected (and recover); nobody else is
    m0 = OutbreakModel(sir_model(), Dict(:τ => 0.0, :γ => 0.25))
    @test m0.infectious == [false, true, false]
    @test NO._infected_mask(m0) == [false, true, false]
    @test m0.infectious == OutbreakModel(sir_model(), Dict(:τ => 1e-12, :γ => 0.25)).infectious   # no threshold
    for alg in (DirectSSA(), NextReaction(), CompositionRejection(), HAS())
        tr = simulate(OutbreakSpec(model = m0, network = g, initial = SeedFraction(:I => 0.05), tspan = (0.0, 200.0));
                      seed = 1, algorithm = alg)
        @test final_size(tr) == 0.05
        @test tr.counts[:, end] == [950, 0, 50]
    end
    # SEAIR with τA = 0, seeded in A: the asymptomatic seeds infect nobody, and they count
    mA = OutbreakModel(seair_model(), merge(P_SEAIR, Dict(:τA => 0.0)))
    @test mA.infectious == [false, false, true, true, false]
    tr = simulate(OutbreakSpec(model = mA, network = g, initial = SeedFraction(:A => 0.05), tspan = (0.0, 200.0));
                  seed = 2)
    @test final_size(tr) == 0.05
    @test NO._initially_infected(tr) == 50
    # SEIR with τ = 0: the unseeded remainder goes to R (the OutbreakSpec rule), never to the latent class E
    me = OutbreakModel(seir_model(), Dict(:τ => 0.0, :σ => 0.2, :γ => 0.25))
    @test me.infectious == [false, false, true, false]
    s0 = init(me, SeedFraction(:S => 0.5, :I => 0.1), 10)
    @test [count(==(i), s0) for i in 1:4] == [5, 0, 1, 4]
    @test s0 == init(OutbreakModel(seir_model(), Dict(:τ => 0.3, :σ => 0.2, :γ => 0.25)),
                     SeedFraction(:S => 0.5, :I => 0.1), 10)
end

# ---------------------------------------------------------------------------------------------------------------
# Seeding (N01, corrected fix; NetworkEpiCore seed_counts, RoundNearestTiesAway, §E.2)
# ---------------------------------------------------------------------------------------------------------------

const SIR = OutbreakModel([:S, :I, :R], [false, true, false],
                          [OutbreakTransition(:S, :I, 0.5, :infection), OutbreakTransition(:I, :R, 0.25, :spontaneous)])
const SEIR = OutbreakModel([:S, :E, :I, :R], [false, false, true, false],
                           [OutbreakTransition(:S, :E, 0.5, :infection), OutbreakTransition(:E, :I, 1.0, :spontaneous),
                            OutbreakTransition(:I, :R, 0.25, :spontaneous)])
const SIS = OutbreakModel([:S, :I], [false, true],
                          [OutbreakTransition(:S, :I, 0.5, :infection), OutbreakTransition(:I, :S, 0.25, :spontaneous)])

@testset "N01: seeding counts" begin
    @test SIR.susceptible == [:S] && SEIR.susceptible == [:S] && SIS.susceptible == [:S]
    @test cnt(SIR, SeedFraction(:I => 0.001), 500, :I) == 1               # was 0 (vignettes 01, 02-TVN)
    @test_throws ArgumentError init(SIR, SeedFraction(:I => 0.01), 40)    # was a silent 0 seeds
    @test cnt(SIR, SeedFraction(:I => 0.001), 2500, :I) == 3              # was 2 (ties to even)
    @test cnt(SIR, SeedFraction(:S => 0.7, :I => 0.3), 5, :R) == 0        # threw "exceed 1.0"
    @test cnt(SIR, SeedFraction(:I => 0.1, :S => 0.9), 15, :R) == 0       # threw "exceed 1.0"
    @test cnt(SIR, SeedFraction(:S => 0.9, :I => 0.1), 25, :R) == 0       # was R = 1
    @test cnt(SEIR, SeedFraction(:S => 0.9, :I => 0.1), 25, :E) == 0      # was E = 1 (an extra latent infection)
    @test cnt(SIR, SeedFraction(:I => 0.1, :S => 0.9), 5, :I) == 1        # was I = 0, R = 1
    @test all(n -> cnt(SIR, SeedFraction(:I => 0.1, :S => 0.9), n, :R) == 0, 5:200)
    @test all(n -> abs(cnt(SIR, SeedFraction(:I => 0.1, :S => 0.9), n, :I) - 0.1n) <= 0.5, 5:200)
    @test cnt(SIR, SeedFraction(:I => 0.1, :S => 0.5), 10, :R) == 4       # a genuine remainder → R
    @test cnt(SIR, SeedFraction(:S => 0.95, :I => 0.049), 100, :I) == 5   # skeptic: no false error
    @test cnt(SEIR, SeedFraction(:S => 0.98, :I => 0.019), 100, :E) == 0  # skeptic: no false error
    @test cnt(SEIR, SeedFraction(:S => 0.5, :I => 0.1), 10, :E) == 0      # a remainder never becomes latent
    @test cnt(SEIR, SeedFraction(:S => 0.5, :I => 0.1), 10, :R) == 4
    @test length(init(SIR, SeedFraction(:I => 0.1), 0)) == 0              # N = 0 is allowed
    @test_throws ArgumentError SeedFraction(:S => 0.9, :I => 0.3)         # sums to 1.2
    @test_throws ArgumentError SeedFraction(:I => NaN)
    @test_throws ArgumentError SeedFraction(:I => 0.5, :I => 0.5)         # left nodes unwritten
    @test_throws ArgumentError SeedFraction(:I => -0.1)                   # out-of-bounds write under @inbounds
    @test_throws ArgumentError init(SIS, SeedFraction(:S => 0.9, :I => 0.05), 100)   # nowhere to put the rest
    @test cnt(SIS, SeedFraction(:I => 0.05; default = :S), 100, :I) == 5
    @test_throws ArgumentError init(SIS, SeedFraction(:S => 0.9, :I => 0.05; default = :I), 100)   # NEC: must sum to 1
    @test_throws ArgumentError init(SIR, SeedFraction(:Z => 0.1), 10)
    @test_throws ArgumentError init(SIR, SeedFraction(:I => 0.1; default = :Z), 10)
    # SeedCount and SeedNodes
    @test cnt(SIR, SeedCount(:I => 7), 50, :I) == 7 && cnt(SIR, SeedCount(:I => 7), 50, :S) == 43
    @test_throws ArgumentError init(SIR, SeedCount(:I => 11), 10)
    s = init(SIR, SeedNodes(:I => [2, 4], :R => [5]), 10)
    @test s[2] == s[4] == SIR.index_of[:I] && s[5] == SIR.index_of[:R] && count(==(SIR.index_of[:S]), s) == 7
    @test_throws ArgumentError init(SIR, SeedNodes(:I => [11]), 10)
    s = init(SIR, SeedNodes(:S => 1:5, :I => [6]), 10)                   # S named, the rest to R (as in 0.1)
    @test count(==(SIR.index_of[:R]), s) == 4
    # seeded nodes are uniform without replacement: every node is seeded with probability n_I/N
    freq = zeros(Int, 10)
    always3 = true
    spec10 = OutbreakSpec(model = SIR, network = SimpleGraph(10), initial = SeedFraction(:I => 0.3), tspan = (0.0, 1.0))
    for r in 1:20_000
        st = NO.initial_state(spec10, NO.stable_rng(r))
        always3 &= count(==(SIR.index_of[:I]), st) == 3
        freq .+= st .== SIR.index_of[:I]
    end
    @test always3
    p̂ = freq ./ 20_000
    @test all(abs.(p̂ .- 0.3) .< 4 * sqrt(0.3 * 0.7 / 20_000))
end

@testset "background rule with several susceptible classes (WP16 review)" begin
    ma = OutbreakModel(awareness_model(), Dict(:τ => 0.3, :κ => 0.2, :τa => 0.05, :γ => 0.25, :σ => 0.5))
    @test ma.susceptible == [:S, :Sa]
    c(s, n) = Dict(X => count(==(ma.index_of[X]), init(ma, s, n)) for X in ma.compartments)
    # a seeding that covers the population puts the rounding into its named susceptible class, never into the
    # unnamed one (was S = 7, Sa = 1; and an error for the second)
    @test c(SeedFraction(:S => 0.74, :E => 0.13, :I => 0.13), 10) == Dict(:S => 8, :Sa => 0, :E => 1, :I => 1, :R => 0)
    @test c(SeedFraction(:S => 0.9, :I => 0.1), 25) == Dict(:S => 22, :Sa => 0, :E => 0, :I => 3, :R => 0)
    @test c(SeedFraction(:Sa => 0.9, :I => 0.1), 25)[:Sa] == 22
    # a seeding that does not cover it still fills the first unnamed susceptible class
    @test c(SeedFraction(:S => 0.5, :I => 0.1), 10) == Dict(:S => 5, :Sa => 4, :E => 0, :I => 1, :R => 0)
    @test c(SeedFraction(:I => 0.1), 10)[:S] == 9
    # the remainder goes to the first compartment that is neither susceptible, infected nor named, in compartment
    # order: the traced class Q of the tracing model (documented in the OutbreakSpec docstring)
    mt = OutbreakModel(tracing_model(), Dict(:τ => 0.3, :α => 0.2, :γ => 0.25, :δ => 0.5))
    @test mt.compartments == [:S, :I, :Q, :D, :R]
    @test count(==(mt.index_of[:Q]), init(mt, SeedFraction(:S => 0.5, :I => 0.1), 10)) == 4
end

@testset "stratified models on a network without node types (§J.6)" begin
    cm = stratify(sir_model(), [:a, :b])
    m = OutbreakModel(cm, Dict(k => 0.3 for k in rate_parameters(cm)))
    @test m.strata == Dict(:S_a => :a, :I_a => :a, :R_a => :a, :S_b => :b, :I_b => :b, :R_b => :b)
    @test m.susceptible == [:S_a, :S_b]
    # the background rule would put all 95 unseeded nodes into S_a and none into S_b: refused
    err = try
        init(m, SeedFraction(:I_a => 0.05), 100)
    catch e
        e
    end
    @test err isa ArgumentError && occursin("S_b", err.msg) && occursin("node types", err.msg)
    @test_throws ArgumentError init(m, SeedCount(:I_a => 5), 100)
    @test_throws ArgumentError init(m, SeedNodes(:I_a => [1, 2]), 100)
    @test_throws ArgumentError simulate(OutbreakSpec(model = m, network = SimpleGraph(100),
                                                     initial = SeedFraction(:I_a => 0.05), tspan = (0.0, 1.0)); seed = 1)
    # a remainder that would go to R_a alone: refused as well
    @test_throws ArgumentError init(m, SeedFraction(:S_a => 0.3, :S_b => 0.3, :I_a => 0.05), 100)
    # seedings that assign every stratum are placed as usual
    c(s) = Dict(X => count(==(m.index_of[X]), init(m, s, 100)) for X in m.compartments)
    @test c(SeedFraction(:S_a => 0.5, :S_b => 0.45, :I_a => 0.05)) ==
          Dict(:S_a => 50, :S_b => 45, :I_a => 5, :I_b => 0, :R_a => 0, :R_b => 0)
    @test c(SeedFraction(:S_b => 0.45, :I_a => 0.05))[:S_a] == 50           # one unnamed susceptible class
    @test c(SeedFraction(:I_a => 0.05; default = :S_b))[:S_b] == 95          # an explicit background
    @test c(SeedNodes(:I_a => 1:5, :S_b => 6:50))[:S_a] == 50
    # the hand-built constructor takes `strata`; models without strata are unaffected
    hand = OutbreakModel(m.compartments, m.infectious, m.transitions; strata = m.strata)
    @test hand.strata == m.strata
    @test_throws ArgumentError init(hand, SeedFraction(:I_a => 0.05), 100)
    @test isempty(OutbreakModel(m.compartments, m.infectious, m.transitions).strata)
    @test cnt(OutbreakModel(m.compartments, m.infectious, m.transitions), SeedFraction(:I_a => 0.05), 100, :S_a) == 95
    @test_throws ArgumentError OutbreakModel([:S, :I], [false, true], OutbreakTransition[]; strata = Dict(:Z => :a))
    @test_throws ArgumentError OutbreakModel([:S, :I], [false, true], OutbreakTransition[]; strata = [:S => "a"])
    @test OutbreakModel([:S, :I], [false, true], OutbreakTransition[]; strata = [:S => :a]).strata == Dict(:S => :a)
    @test isempty(OutbreakModel(awareness_model(), Dict(:τ => 0.3, :κ => 0.2, :τa => 0.05, :γ => 0.25, :σ => 0.5)).strata)
    r2 = OutbreakModel(with_reinfection_counting(sis_model(), 2), Dict(:τ => 0.8, :γ => 0.25))
    @test isempty(r2.strata) && cnt(r2, SeedFraction(:I_1 => 0.05), 100, :S_0) == 95
    @test occursin("strata:", sprint(show, MIME"text/plain"(), m))
end

@testset "E28: seeding correspondence of the converted SEIR model" begin
    cm = seir_model()
    m = OutbreakModel(cm, Dict(:τ => 0.3, :σ => 0.5, :γ => 0.5))
    @test m.compartments == [:S, :E, :I, :R] && m.infectious == [false, false, true, false]
    N = 20_000
    counts0(s) = [count(==(i), init(m, s, N)) for i in 1:4]
    @test counts0(SeedFraction(:E => 0.01)) == [19800, 200, 0, 0]
    @test counts0(SeedFraction(:I => 0.01)) == [19800, 0, 200, 0]
    @test counts0(SeedFraction(:E => 0.005, :I => 0.005)) == [19800, 100, 100, 0]
    # the default seeding (§E.2) is the unique entry state: E for SEIR
    @test default_seed_state(cm) === :E
    @test counts0(default_seed(cm, 0.01)) == [19800, 200, 0, 0]
end

# ---------------------------------------------------------------------------------------------------------------
# Trajectories: traj.seed and right-continuous state_at (WP3 requests)
# ---------------------------------------------------------------------------------------------------------------

@testset "traj.seed and state_at at t0" begin
    @test fieldtype(OutbreakTrajectory, :seed) == Union{Nothing, UInt64}
    tr = OutbreakTrajectory(SIR, [0.0, 1.0], [10 9; 0 1; 0 0], zeros(Int, 10), OutbreakEvent[], nothing, :unit)
    @test tr.seed === nothing
    # an intervention at t0: state_at(t0) is the state after it, as mean_curve (was the state before it)
    sirv = OutbreakModel([:S, :I, :R, :V], [false, true, false, false],
                         [OutbreakTransition(:S, :I, 0.2, :infection), OutbreakTransition(:I, :R, 0.25, :spontaneous)])
    spec = OutbreakSpec(model = sirv, network = random_regular_graph(200, 4; rng = StableRNG(1)),
                        initial = SeedFraction(:I => 0.05), tspan = (0.0, 10.0))
    plan = InterventionPlan([ScheduledStateChange(0.0, :V, 0.95; from = [:S])])
    tr = simulate(spec; seed = 1, interventions = plan)
    @test tr.times[1] == tr.times[2] == 0.0                              # snapshots before and after the pulse
    @test state_at(tr, 0.0) == [0, 10, 0, 190]
    @test tr(0.0) == state_at(tr, 0.0)
    @test state_at(tr, -1.0) == tr.counts[:, 1] == [190, 10, 0, 0]
    ens = simulate_ensemble(spec; nsims = 3, seed = 1, interventions = plan)
    _, V = mean_curve(ens, :V; tgrid = [0.0])
    @test V == [190.0]
    @test all(state_at(t, 0.0)[4] == 190 for t in ens)
    @test_throws ArgumentError state_at(tr, NaN)
end

# ---------------------------------------------------------------------------------------------------------------
# TimeVaryingNetwork: strict add/remove semantics (WP3 request, m4)
# ---------------------------------------------------------------------------------------------------------------

@testset "TimeVaryingNetwork strict updates" begin
    si = OutbreakModel([:S, :I], [false, true], [OutbreakTransition(:S, :I, 1.0, :infection)])
    spec(tvn, t0 = 0.0) = OutbreakSpec(model = si, network = tvn, initial = SeedNodes(:I => [1]), tspan = (t0, 10.0))
    g = path_graph(3)                                                     # edges 1–2, 2–3
    ok = TimeVaryingNetwork(g, [(t = 1.0, src = 1, dst = 3, action = :add), (t = 2.0, src = 1, dst = 3, action = :remove),
                                (t = 3.0, src = 3, dst = 1, action = :add), (t = 4.0, src = 1, dst = 2, action = :remove)])
    @test spec(ok) isa OutbreakSpec
    @test_throws ArgumentError spec(TimeVaryingNetwork(g, [(t = 1.0, src = 1, dst = 2, action = :add)]))
    @test_throws ArgumentError spec(TimeVaryingNetwork(g, [(t = 1.0, src = 1, dst = 3, action = :remove)]))
    @test_throws ArgumentError spec(TimeVaryingNetwork(g, [(t = 1.0, src = 1, dst = 3, action = :add),
                                                          (t = 2.0, src = 3, dst = 1, action = :add)]))
    @test_throws ArgumentError spec(TimeVaryingNetwork(g, [(t = 1.0, src = 1, dst = 9, action = :add)]))
    @test_throws ArgumentError spec(TimeVaryingNetwork(g, [(t = 1.0, src = 2, dst = 2, action = :add)]))
    @test_throws ArgumentError spec(TimeVaryingNetwork(g, [(t = 1.0, src = 1, dst = 3, action = :toggle)]))
    # updates before tspan[1] are the network's past: skipped, so not checked against the base graph
    past = TimeVaryingNetwork(g, [(t = -1.0, src = 1, dst = 2, action = :add), (t = 1.0, src = 1, dst = 3, action = :add)])
    @test spec(past) isa OutbreakSpec
    @test_throws ArgumentError spec(past, -2.0)
    # the spec's graph is not mutated by validation or simulation
    tr = simulate(spec(ok); seed = 1, algorithm = NextReaction())
    @test ne(g) == 2 && sum(tr.counts[:, end]) == 3
    @test_throws ArgumentError OutbreakSpec(model = si, network = g, initial = SeedNodes(:I => [1]), tspan = (1.0, 0.0))
end

# ---------------------------------------------------------------------------------------------------------------
# sample_graph fallback (§C.3)
# ---------------------------------------------------------------------------------------------------------------

struct _NoSamplerNetwork <: NetworkDescriptor end

@testset "sample_graph fallback" begin
    g, info = sample_graph(ConfigurationNetwork(RegularDegree(6)), 1000; rng = NO.stable_rng(1))
    @test nv(g) == 1000 && all(==(6), degree(g)) && info.method === :random_regular
    @test_throws ArgumentError sample_graph(ConfigurationNetwork(RegularDegree(3)), 11; rng = NO.stable_rng(1))
    g, info = sample_graph(ConfigurationNetwork(PoissonDegree(5.0)), 20_000; rng = NO.stable_rng(2))
    @test info.method === :erdos_renyi && abs(info.mean_degree - 5) < 4 * sqrt(5 / 20_000) * sqrt(2)
    @test abs(info.excess_degree - 5) < 0.1                               # 4 sampling sd (0.024)
    bim = EmpiricalDegree(Dict(2 => 5 / 6, 10 => 1 / 6))
    g, info = sample_graph(ConfigurationNetwork(bim), 20_000; rng = NO.stable_rng(3))
    @test info.method === :erased_configuration && !has_self_loops(g) && !is_directed(g)
    @test info.erased_fraction < 1e-3
    # 4 sampling sd (sd 0.020 and 0.036 over 200 draws at N = 2·10⁴)
    @test abs(info.mean_degree - mean_degree(bim)) < 0.08 && abs(info.excess_degree - excess_degree(bim)) < 0.15
    # reproducible under the same stream, different under another
    a, _ = sample_graph(ConfigurationNetwork(bim), 500; rng = NO.stable_rng(9))
    b, _ = sample_graph(ConfigurationNetwork(bim), 500; rng = NO.stable_rng(9))
    c, _ = sample_graph(ConfigurationNetwork(bim), 500; rng = NO.stable_rng(10))
    @test collect(edges(a)) == collect(edges(b)) && collect(edges(a)) != collect(edges(c))
    h = path_graph(7)
    @test first(sample_graph(ExplicitGraph(h), 7)) === h
    @test_throws ArgumentError sample_graph(ExplicitGraph(h), 8)
    # WellMixed(κ): K_N as a one-layer MultiplexGraph with layer rate κ/(N − 1) (the lumping test; WP25)
    wm, info = sample_graph(WellMixed(5), 100; rng = NO.stable_rng(1))
    @test wm isa MultiplexGraph && length(wm.layers) == 1 && wm.layer_rates == [5 / 99]
    @test ne(wm) == 100 * 99 ÷ 2 && info.method === :complete
    # a DynamicNetwork gives the run's initial contact network: a sample of the base with the process attached
    dn, dinfo = sample_graph(DynamicNetwork(RegularDegree(4), NeighbourExchange(1.0)), 200; rng = NO.stable_rng(4))
    @test dn isa DynamicGraph && dn.process isa NeighbourExchangeProcess && all(==(4), degree(dn.graph))
    @test collect(edges(dn.graph)) == collect(edges(first(sample_graph(ConfigurationNetwork(RegularDegree(4)), 200;
                                                                       rng = NO.stable_rng(4)))))
    @test dinfo.mean_degree == 4.0
    # the fallback: a descriptor without a sampler is an ArgumentError naming it
    err = try
        sample_graph(_NoSamplerNetwork(), 10)
    catch e
        e
    end
    @test err isa ArgumentError && occursin("_NoSamplerNetwork", err.msg)
end

# ---------------------------------------------------------------------------------------------------------------
# simulate(model, net; N, …) and simulate(sc::Scenario)
# ---------------------------------------------------------------------------------------------------------------

@testset "simulate(model, ConfigurationNetwork(PoissonDegree(5)); N = 1000, …) smoke test" begin
    net = ConfigurationNetwork(PoissonDegree(5))
    p = Dict(:τ => 1 / 6, :γ => 1 / 4)
    kw = (N = 1000, p, initial = SeedFraction(:I => 0.01), tspan = (0.0, 60.0))
    ens = simulate(sir_model(), net; kw..., nsims = 8, seed = 20260926)
    @test ens isa OutbreakEnsemble && length(ens) == 8
    @test ens.spec.network isa SampledNetwork && ens.spec.network.N == 1000 && nv(ens.spec.network) == 1000
    @test ens.spec.model.compartments == [:S, :I, :R]
    @test_throws ArgumentError simulate(ens.spec)                         # a SampledNetwork is not one graph
    @test all(t -> t.times == collect(range(0.0, 60.0; length = 201)), ens)   # keep = :grid
    @test all(t -> all(sum(t.counts; dims = 1) .== 1000), ens)
    @test all(t -> t.counts[:, 1] == [990, 10, 0], ens)
    fs = final_size(ens)
    @test all(0.01 .<= fs .<= 1)
    @test count(>(0.1), fs) >= 6                                         # 10 seeds: nearly all runs are major
    # §J.7 streams: run r is simulate(spec_r; seed = b + 2^32 + r) on graph r from stable_rng(b + r)
    b = UInt64(20260926)
    @test [t.seed for t in ens] == [b + UInt64(2)^32 + UInt64(r) for r in 1:8]
    for r in (1, 5)
        g = first(sample_graph(net, 1000; rng = NO.stable_rng(b + r)))
        tr = simulate(OutbreakSpec(ens.spec.model, g, kw.initial, kw.tspan); seed = ens[r].seed,
                      algorithm = NextReaction())
        @test NO._on_grid(tr, ens[r].times).counts == ens[r].counts
        @test final_size(tr) == fs[r]
    end
    # reproducible; parallel = sequential; fresh graphs per run
    again = simulate(sir_model(), net; kw..., nsims = 8, seed = 20260926, parallel = true)
    @test all(ens[r].counts == again[r].counts for r in 1:8)
    fixed = simulate(sir_model(), net; kw..., nsims = 3, seed = 7, graphs = :fixed)
    @test fixed.spec.network isa StaticNetwork
    pool = simulate(sir_model(), net; kw..., nsims = 4, seed = 7, graphs = (:pool, 2))
    @test pool.spec.network.graphs == (:pool, 2)
    # keep = :counts / :events, a custom grid, seed / rng exclusivity
    evs = simulate(sir_model(), net; kw..., nsims = 2, seed = 3, keep = :events)
    @test !isempty(evs[1].events) && length(evs[1].times) > 201
    grid = simulate(sir_model(), net; kw..., nsims = 2, seed = 3, tgrid = 0:1:60)
    @test grid[1].times == collect(0.0:1.0:60.0)
    @test final_size(grid) == final_size(evs)                            # same streams, any keep mode
    viarng = simulate(sir_model(), net; kw..., nsims = 2, rng = StableRNG(4))
    @test length(viarng) == 2
    @test_throws ArgumentError simulate(sir_model(), net; kw..., seed = 1, rng = StableRNG(4))
    @test_throws ArgumentError simulate(sir_model(), net; kw..., tgrid = 0:1:70)
    @test_throws ArgumentError simulate(sir_model(), net; kw..., keep = :all)
    @test_throws ArgumentError simulate(sir_model(), net; kw..., graphs = :each)
    err = try
        simulate(sir_model(), net; kw..., graphs = :each)
    catch e
        e
    end
    @test occursin(":per_run, :fixed or (:pool, G)", err.msg)
    @test_throws ArgumentError SampledNetwork(net, 1000, :fixed)       # an ensemble with one graph holds the graph
    # a well-mixed population runs with the count-level MassActionSSA by default (WP26)
    wm = simulate(sir_model(), WellMixed(5); kw..., nsims = 2, seed = 1)
    @test wm isa OutbreakEnsemble && length(wm) == 2 && all(t -> t.algorithm === :MassActionSSA, wm)
    @test wm.spec.network isa SampledNetwork && wm.spec.network.descriptor == WellMixed(5)
    @test all(t -> all(sum(t.counts; dims = 1) .== 1000), wm)
    # an OutbreakModel, a fixed graph, an ExplicitGraph
    om = OutbreakModel(sir_model(), p)
    g = random_regular_graph(500, 5; rng = StableRNG(12))
    e1 = simulate(om, g; initial = SeedFraction(:I => 0.02), tspan = (0.0, 30.0), nsims = 3, seed = 5)
    e2 = simulate(sir_model(), ExplicitGraph(g); p, initial = SeedFraction(:I => 0.02), tspan = (0.0, 30.0),
                  nsims = 3, seed = 5)
    @test e1.spec.network isa StaticNetwork && all(e1[r].counts == e2[r].counts for r in 1:3)
    @test_throws ArgumentError simulate(om, g; p, initial = SeedFraction(:I => 0.02), tspan = (0.0, 30.0))
    # N is optional on a fixed graph and must match it
    e3 = simulate(om, g; N = 500, initial = SeedFraction(:I => 0.02), tspan = (0.0, 30.0), nsims = 3, seed = 5)
    @test all(e1[r].counts == e3[r].counts for r in 1:3)
    @test_throws ArgumentError simulate(om, g; N = 499, initial = SeedFraction(:I => 0.02), tspan = (0.0, 30.0))
    @test_throws ArgumentError simulate(sir_model(), ExplicitGraph(g); N = 499, p, initial = SeedFraction(:I => 0.02),
                                        tspan = (0.0, 30.0))
    # an error inside a run is raised as itself, also with parallel = true (not as a TaskFailedException)
    tiny = merge(kw, (initial = SeedFraction(:I => 1e-4),))                # rounds to 0 seeds on 1000 nodes
    for parallel in (false, true)
        @test_throws ArgumentError simulate(sir_model(), net; tiny..., nsims = 3, seed = 1, parallel)
    end
    # NetworkEpiCore observables on trajectories
    @test compartment(ens[1], :I) == compartment_series(ens[1], :I) ./ 1000
    @test sort(collect(keys(compartments(ens[1], [:S, :R])))) == [:R, :S]
end

@testset "simulate(sc::Scenario)" begin
    sc = derive(scenario(:sir_pois5); N = 500, nsims = 4)
    ens = simulate(sc)
    @test length(ens) == 4 && ens[1].times == collect(sc.tgrid)
    @test ens.spec.network.descriptor == sc.network && ens.spec.network.N == 500
    @test [t.seed for t in ens] == [sc.sim.base_seed + UInt64(2)^32 + UInt64(r) for r in 1:4]
    g = first(sample_graph(sc.network, 500; rng = NO.stable_rng(sc.sim.base_seed + 2)))
    tr = simulate(OutbreakSpec(ens.spec.model, g, sc.initial, sc.tspan); seed = ens[2].seed, algorithm = NextReaction())
    @test NO._on_grid(tr, collect(sc.tgrid)).counts == ens[2].counts
    @test length(simulate(sc; nsims = 2)) == 2
    @test NO._algorithm(:direct) isa DirectSSA && NO._algorithm(:has) isa HAS
    @test_throws ArgumentError NO._algorithm(:gillespie)
end

# ---------------------------------------------------------------------------------------------------------------
# Stochastic validation: the converted models against the edge-based large-N limit
# ---------------------------------------------------------------------------------------------------------------
# Poisson(5) configuration networks (Erdős–Rényi G(N, 5/(N − 1)) samples), N = 10⁴, a fresh graph per run, the §J.7
# streams, NextReaction, conditioned on major outbreaks (cumulative incidence minus seeds ≥ 0.05, §E.2). The
# edge-based references are NetworkEpiCore's final size, or the Miller–Volz edge-based ODE written out here and
# integrated with RK4 (ψ(x) = exp(5(x − 1)), q = 1 − Σρ).

ψ(x) = exp(5 * (x - 1)); ψ1(x) = 5ψ(x); ψ2(x) = 25ψ(x)

function rk4(f, u0, tgrid; dt = 0.005)
    u = copy(u0); t = first(tgrid); out = [copy(u)]
    for tk in tgrid[2:end]
        while t < tk - 1e-12
            h = min(dt, tk - t)
            k1 = f(u, t); k2 = f(u .+ (h / 2) .* k1, t + h / 2); k3 = f(u .+ (h / 2) .* k2, t + h / 2)
            k4 = f(u .+ h .* k3, t + h)
            u = u .+ (h / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4); t += h
        end
        push!(out, copy(u))
    end
    return reduce(hcat, out)
end

const VAL_N = 10_000
const VAL_RUNS = 50

function major_runs(model, p, initial, tspan, tgrid; seed)
    ens = simulate(model, ConfigurationNetwork(PoissonDegree(5)); N = VAL_N, p, initial, tspan, tgrid,
                   nsims = VAL_RUNS, seed)
    ρ = sum(last, initial.fractions)
    major = [t for t in ens if final_size(t) - ρ >= 0.05]
    return major, length(ens)
end
curve(runs, X) = mean(compartment(t, X) for t in runs)
se_final(runs) = std(final_size.(runs)) / sqrt(length(runs))

@testset "validation against the edge-based limit (N = $(VAL_N), $(VAL_RUNS) runs per model)" begin
    tg = 0.0:1.0:60.0
    # SIR: final size against NetworkEpiCore's R∞ (τ = 1/6, γ = 1/4, R₀ = 2)
    p = Dict(:τ => 1 / 6, :γ => 1 / 4)
    Rinf = final_size(sir_model(), ConfigurationNetwork(PoissonDegree(5.0)), p; initial = SeedFraction(:I => 0.01))
    runs, n = major_runs(sir_model(), p, SeedFraction(:I => 0.01), (0.0, 60.0), tg; seed = 101)
    fs = mean(final_size.(runs)); se = se_final(runs)
    @info "SIR Poisson(5): NO final size vs NEC R∞" N = VAL_N runs = n major = length(runs) fs se Rinf
    @test length(runs) >= 0.95n
    @test abs(fs - Rinf) < 4se + 0.005

    # SEAIR (branching into two infectious classes, two infectors): final size against NEC (R₀ = 1.70)
    Rinf = final_size(seair_model(), ConfigurationNetwork(PoissonDegree(5.0)), P_SEAIR; initial = SeedFraction(:E => 0.01))
    runs, n = major_runs(seair_model(), P_SEAIR, SeedFraction(:E => 0.01), (0.0, 200.0), 0.0:2.0:200.0; seed = 102)
    fs = mean(final_size.(runs)); se = se_final(runs)
    @info "SEAIR Poisson(5): NO final size vs NEC R∞" N = VAL_N runs = n major = length(runs) fs se Rinf
    @test length(runs) >= 0.9n
    @test abs(fs - Rinf) < 4se + 0.005

    # SEIR seeded in E and in I (E28): the time course against the EB ODE from the same seeding
    τ, σ, γ = 1 / 6, 1 / 5, 1 / 4
    tgs = 0.0:2.0:150.0
    for (X, ρE, ρI) in ((:E, 0.01, 0.0), (:I, 0.0, 0.01))
        q = 1 - ρE - ρI
        f(u, t) = begin
            θ, φE, φI, E, I = u
            dθ = -τ * φI
            dφS = q * ψ2(θ) * dθ / ψ1(1.0)
            dS = q * ψ1(θ) * dθ
            [dθ, -dφS - σ * φE, σ * φE - (τ + γ) * φI, -dS - σ * E, σ * E - γ * I]
        end
        U = rk4(f, [1.0, ρE, ρI, ρE, ρI], tgs)
        runs, n = major_runs(seir_model(), Dict(:τ => τ, :σ => σ, :γ => γ), SeedFraction(X => 0.01), (0.0, 150.0),
                             tgs; seed = X === :E ? 103 : 104)
        DI = maximum(abs.(curve(runs, :I) .- U[5, :]))
        DE = maximum(abs.(curve(runs, :E) .- U[4, :]))
        Sinf = q * ψ(U[1, end])
        fs = mean(final_size.(runs)); se = se_final(runs)
        @info "SEIR Poisson(5), seeds in $X: NO vs EB ODE" N = VAL_N runs = n major = length(runs) DI DE fs se EB_final = 1 - Sinf
        @test length(runs) >= 0.95n
        @test DI < 0.01 && DE < 0.01
        @test abs(fs - (1 - Sinf)) < 4se + 0.005
    end

    # SIR + vaccination S → V (the exit type, ξ = e^{−νt}): S(t) and I(t) against the EB ODE
    ν = 0.02
    q = 0.99
    fv(u, t) = begin
        θ, φI, I = u
        ξ = exp(-ν * t)
        [-τ * φI, q * ξ * ψ2(θ) * τ * φI / ψ1(1.0) - (τ + γ) * φI, q * ξ * ψ1(θ) * τ * φI - γ * I]
    end
    U = rk4(fv, [1.0, 0.01, 0.01], tg)
    S_eb = [q * exp(-ν * t) * ψ(U[1, k]) for (k, t) in enumerate(tg)]
    runs, n = major_runs(sirv_model(), Dict(:τ => τ, :γ => γ, :ν => ν), SeedFraction(:I => 0.01), (0.0, 60.0), tg;
                         seed = 105)
    DS = maximum(abs.(curve(runs, :S) .- S_eb)); DI = maximum(abs.(curve(runs, :I) .- U[3, :]))
    @info "SIR + vaccination Poisson(5): NO vs EB ODE with exits" N = VAL_N runs = n major = length(runs) DS DI
    @test length(runs) >= 0.95n
    @test DS < 0.01 && DI < 0.01

    # two strains with full cross-immunity (two entries): I1(t), I2(t) against the per-reaction EB field
    τ1, τ2 = 1 / 6, 1 / 5
    f2(u, t) = begin
        θ, φ1, φ2, I1, I2 = u
        h1, h2 = τ1 * φ1, τ2 * φ2
        w1 = h1 / (h1 + h2)
        dθ = -(h1 + h2)
        Fφ = -q * ψ2(θ) * dθ / ψ1(1.0)
        F = -q * ψ1(θ) * dθ
        [dθ, w1 * Fφ - (τ1 + γ) * φ1, (1 - w1) * Fφ - (τ2 + γ) * φ2, w1 * F - γ * I1, (1 - w1) * F - γ * I2]
    end
    U = rk4(f2, [1.0, 0.005, 0.005, 0.005, 0.005], tg)
    runs, n = major_runs(twostrain_model(), Dict(:τ1 => τ1, :τ2 => τ2, :γ => γ),
                         SeedFraction(:I1 => 0.005, :I2 => 0.005), (0.0, 60.0), tg; seed = 106)
    D1 = maximum(abs.(curve(runs, :I1) .- U[4, :])); D2 = maximum(abs.(curve(runs, :I2) .- U[5, :]))
    @info "two strains Poisson(5): NO vs EB ODE" N = VAL_N runs = n major = length(runs) D1 D2
    @test length(runs) >= 0.95n
    @test D1 < 0.01 && D2 < 0.01
end
