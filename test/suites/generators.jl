# Graph generators (DESIGN_NetworkEpiCore.md §C.3, §G.2 WP25, §J.6, §J.7; verified issue N06).
#
# Acceptance (WP25): realised degree histograms match the pmf (χ²); typed stub matching has exact reciprocal totals;
# Newman–Miller transitivity is 2/15 for (2, 2) (within 3 SE, in the N → ∞ limit: see below); graphs are
# reproducible under StableRNG; GraphInfo is filled, including the erased fraction. The simulation checks compare
# NetworkOutbreaks with NetworkEpiCore's edge-based final size (conditioned on a major outbreak, §E.2) and with the
# exact final-size law of the lumped SIR chain on K_N.

using NetworkOutbreaks
import NetworkOutbreaks as NO
using Graphs
using Random
using Statistics
using Test

const MPX = NO.NetworkEpiCore.MultiplexNetwork          # inside NetworkOutbreaks `MultiplexNetwork` is the graph alias

# Pinned edge-list checksums (`edge_checksum` below) of `sample_graph(net, 500; rng = NO.stable_rng(20260926))` for
# the four networks of the reproducibility testset, recorded when WP25 landed.
const PINNED_CONFIG = 285516626270061640
const PINNED_NM = 1054089319756011368
const PINNED_SBM = 1043313158577746687
const PINNED_TYPED = 2109024740644919795

# ---------------------------------------------------------------------------------------------------------------
# Statistics helpers (Base only)
# ---------------------------------------------------------------------------------------------------------------

# log Γ(a) for a ∈ ℕ/2 (all that χ² needs), exactly as sums of logarithms.
function loggamma_half(a::Real)
    if isinteger(a)
        return sum(log, 1:(Int(a) - 1); init = 0.0)
    end
    n = Int(a - 1 / 2)
    return log(pi) / 2 + sum(k -> log(k - 1 / 2), 1:n; init = 0.0)
end

# Upper tail of the χ² distribution, Q(df/2, x/2) (series or continued fraction, Numerical Recipes gammq).
function chisq_sf(x::Real, df::Integer)
    a, y = df / 2, x / 2
    y <= 0 && return 1.0
    lpre = -y + a * log(y) - loggamma_half(a)
    if y < a + 1
        ap, del = a, 1 / a
        s = del
        for _ in 1:100_000
            ap += 1
            del *= y / ap
            s += del
            abs(del) < abs(s) * 1e-16 && break
        end
        return 1 - s * exp(lpre)
    end
    tiny = 1e-300
    b = y + 1 - a
    c, d = 1 / tiny, 1 / b
    h = d
    for i in 1:100_000
        an = -i * (i - a)
        b += 2
        d = an * d + b
        abs(d) < tiny && (d = tiny)
        c = b + an / c
        abs(c) < tiny && (c = tiny)
        d = 1 / d
        del = d * c
        h *= del
        abs(del - 1) < 1e-16 && break
    end
    return exp(lpre) * h
end

# Pearson χ² of the counts of `values` (non-negative integers) against the pmf `p` (p[k+1] = P(K = k)), pooling
# consecutive values into bins with expected count ≥ 5 (the last bin takes the upper tail, including any mass the
# pmf leaves out). Returns (statistic, degrees of freedom, p-value).
function pooled_chisq(values::AbstractVector{<:Integer}, p::AbstractVector{<:Real}; minexp = 5.0)
    n = length(values)
    kmax = max(maximum(values), length(p) - 1)
    obs = zeros(Int, kmax + 1)
    for v in values
        obs[v + 1] += 1
    end
    pk = zeros(kmax + 1)
    pk[1:length(p)] .= p
    pk[end] += max(0.0, 1 - sum(pk))
    O, E = Int[], Float64[]
    o, e = 0, 0.0
    for k in 1:(kmax + 1)
        o += obs[k]
        e += n * pk[k]
        if e >= minexp && n * sum(pk[(k + 1):end]) >= minexp
            push!(O, o)
            push!(E, e)
            o, e = 0, 0.0
        end
    end
    if e > 0 || o > 0                  # the upper tail: its own bin if large enough, else merged into the last bin
        if isempty(E) || e >= minexp
            push!(O, o)
            push!(E, e)
        else
            O[end] += o
            E[end] += e
        end
    end
    stat = sum((O .- E) .^ 2 ./ E)
    df = length(E) - 1
    return stat, df, df >= 1 ? chisq_sf(stat, df) : 1.0
end

# A checksum of the edge list that is stable across Julia versions (unlike `hash`).
function edge_checksum(g::AbstractGraph)
    h = Int128(0)
    m = Int128(2)^61 - 1
    for e in edges(g)
        h = (h * 1_000_003 + src(e) * 7919 + dst(e)) % m
    end
    return Int64(h)
end

# The pmf of k = s + 2t for independent s and t with pmfs ps and pt.
function degree_pmf_st(ps::Vector{Float64}, pt::Vector{Float64})
    p = zeros(length(ps) + 2 * (length(pt) - 1))
    for (i, a) in enumerate(ps), (j, b) in enumerate(pt)
        p[(i - 1) + 2 * (j - 1) + 1] += a * b
    end
    return p
end

# Neighbours of each type-a node that have type b.
function typed_degrees(g::TypedGraph, a::Symbol, b::Symbol)
    types = node_types(g)
    return [count(u -> types[u] === b, neighbors(g, v)) for v in nodes_of_type(g, a)]
end

# Exact final-size law of the SIR chain on K_N lumped to (S, I): infection rate rate·S·I, recovery γI. Returns the
# probabilities of each number of nodes ever infected (index n + 1).
function kn_final_size_law(N::Int, n0::Int, rate::Float64, γ::Float64)
    P = zeros(N + 1, N + 1)                # P[s + 1, i + 1]
    P[N - n0 + 1, n0 + 1] = 1.0
    law = zeros(N + 1)
    for s in (N - n0):-1:0, i in (N - s):-1:0
        m = P[s + 1, i + 1]
        m == 0 && continue
        if i == 0
            law[N - s + 1] += m
            continue
        end
        a, b = rate * s * i, γ * i
        s > 0 && (P[s, i + 2] += m * a / (a + b))
        P[s + 1, i] += m * b / (a + b)
    end
    return law
end

major(fs; ρ = 0.01) = filter(x -> x - ρ >= 0.05, fs)                     # MajorOutbreak(0.05), design §E.2

# ---------------------------------------------------------------------------------------------------------------

@testset "χ² helper against scipy.stats.chi2.sf" begin
    # reference values: scipy.stats.chi2.sf(x, df)
    for (x, df, ref) in ((3.841458820694124, 1, 0.04999999999999994), (10.0, 7, 0.18857346751345005),
                         (25.0, 12, 0.014822874597441575), (0.5, 3, 0.9188914116546758),
                         (100.0, 80, 0.064570368921133), (1e-3, 2, 0.9995001249791693))
        @test chisq_sf(x, df) ≈ ref rtol = 1e-10
    end
end

@testset "GraphInfo replaces the fallback's NamedTuple" begin
    # before WP25, sample_graph(::ConfigurationNetwork) was the src/convenience.jl fallback returning a NamedTuple
    g, info = sample_graph(ConfigurationNetwork(PoissonDegree(5.0)), 1000; rng = NO.stable_rng(1))
    @test info isa GraphInfo
    @test info.method === :erdos_renyi && info.N == 1000 && info.edges == ne(g)
    @test info.edge_probability ≈ 5 / 999
    @test info.descriptor == ConfigurationNetwork(PoissonDegree(5.0))
    @test :edge_probability in propertynames(info) && :erased_fraction in propertynames(info)
    @test_throws ArgumentError info.no_such_property
    @test occursin("erdos_renyi", sprint(show, info)) && occursin("mean degree", sprint(show, MIME"text/plain"(), info))
end

@testset "configuration networks: degree histograms (χ²) and GraphInfo" begin
    N = 10_000
    pvals = Dict{String, Float64}()
    laws = ("Poisson(5)" => PoissonDegree(5.0), "NegBin(4, 8)" => NegBinDegree(mean = 4, var = 8),
            "bimodal {2, 10}" => EmpiricalDegree(Dict(2 => 5 / 6, 10 => 1 / 6)),
            "power law 2.5 on 2..60" => PowerLawDegree(2.5, 2, 60), "Binomial(10, 0.4)" => BinomialDegree(10, 0.4),
            "mixture" => MixtureDegree([0.3, 0.7], [RegularDegree(2), PoissonDegree(6.0)]))
    for (seed, (name, d)) in enumerate(laws)
        net = ConfigurationNetwork(d)
        g, info = sample_graph(net, N; rng = NO.stable_rng(100 + seed))
        @test nv(g) == N && !is_directed(g) && !has_self_loops(g)
        @test info.method === (d isa PoissonDegree ? :erdos_renyi : :erased_configuration)
        stat, df, pv = pooled_chisq(degree(g), degree_probabilities(d))
        pvals[name] = pv
        @test df >= 1
        @test pv > 1e-3
        # GraphInfo: realised statistics, erased fraction consistent with its counts
        @test info.N == N && info.edges == ne(g)
        @test info.mean_degree ≈ 2ne(g) / N
        @test info.excess_degree ≈ sum(k -> k * (k - 1), degree(g)) / sum(degree(g))
        @test abs(info.mean_degree - mean_degree(d)) < 5 * sqrt(var(degree(g)) / N) + 0.02
        if info.method === :erased_configuration
            @test info.candidate_edges == ne(g) + info.erased
            @test info.erased_fraction ≈ info.erased / info.candidate_edges
            @test info.erased_fraction < 5e-3
            @test info.structural_cutoff ≈ sqrt(N * mean_degree(d)) && !info.cutoff_exceeded
        else
            @test info.erased == 0 && info.erased_fraction == 0
        end
    end
    @info "configuration χ² p-values (N = $N)" pvals

    # RegularDegree: random_regular_graph, exact degrees
    g, info = sample_graph(ConfigurationNetwork(RegularDegree(6)), N; rng = NO.stable_rng(1))
    @test all(==(6), degree(g)) && info.method === :random_regular && info.erased_fraction == 0
    @test info.excess_degree == 5.0
    @test_throws ArgumentError sample_graph(ConfigurationNetwork(RegularDegree(3)), 11; rng = NO.stable_rng(1))
    @test_throws ArgumentError sample_graph(ConfigurationNetwork(RegularDegree(12)), 12; rng = NO.stable_rng(1))
    # a law on odd degrees with N odd cannot have an even degree sum
    @test_throws ArgumentError sample_graph(ConfigurationNetwork(EmpiricalDegree(Dict(3 => 1.0))), 11;
                                            rng = NO.stable_rng(1))
    @test_throws ArgumentError sample_graph(ConfigurationNetwork(PoissonDegree(5.0)), 0)
end

@testset "erased configuration model: erased stub pairs → ν/2 + ν²/4 (Janson 2009)" begin
    # Self-loops ~ Poisson(ν/2) and double edges ~ Poisson(ν²/4) in the configuration model, ν = E[k(k−1)]/E[k];
    # every erased pair is one of them (triple edges are O(1/N)).
    d = EmpiricalDegree(Dict(2 => 5 / 6, 10 => 1 / 6))
    ν = excess_degree(d)                                                    # 5
    R = 60
    erased = [sample_graph(ConfigurationNetwork(d), 10_000; rng = NO.stable_rng(500 + r))[2].erased for r in 1:R]
    expected = ν / 2 + ν^2 / 4                                              # 8.75
    @test abs(mean(erased) - expected) < 4 * std(erased) / sqrt(R)
end

@testset "structural cutoff flag" begin
    d = PowerLawDegree(2.0, 1, 2000)
    g, info = @test_logs (:warn, r"structural cutoff") match_mode = :any sample_graph(ConfigurationNetwork(d), 2000;
                                                                                      rng = NO.stable_rng(3))
    @test info.cutoff_exceeded && info.structural_cutoff < 2000
    @test info.erased > 0
end

@testset "explicit graphs" begin
    h = cycle_graph(7)
    g, info = sample_graph(ExplicitGraph(h), 7)
    @test g === h && info.method === :explicit && info.mean_degree == 2 && info.erased == 0
    @test_throws ArgumentError sample_graph(ExplicitGraph(h), 8)
end

# ---------------------------------------------------------------------------------------------------------------
# Multitype networks
# ---------------------------------------------------------------------------------------------------------------

# A non-Poisson typed network (verified issue N06, part A): A–A 3-regular, A→B Poisson(2); B→A Binomial(4, 1/3),
# B–B ∈ {1, 5}. Reciprocity: 0.4·2 = 0.6·4/3.
const TYPED = MultitypeNetwork([:A, :B], [0.4, 0.6],
    [IndependentDegrees(:A => RegularDegree(3), :B => PoissonDegree(2.0)),
     IndependentDegrees(:A => BinomialDegree(4, 1 / 3), :B => EmpiricalDegree(Dict(1 => 0.5, 5 => 0.5)))])
const SBM2 = sbm_network([:a, :b], [0.5, 0.5]; mean_contacts = [6 2; 2 4])      # the :sir_sbm2 network
const UNSTR = unstructured(RegularDegree(6), [:a, :b], [0.5, 0.5])             # the :sir_unstr2 network (M10)

@testset "TypedGraph" begin
    g = TypedGraph(path_graph(4), [:x, :y, :x, :y])
    @test g.types == [:x, :y] && g.type_of == [1, 2, 1, 2] && node_types(g) == [:x, :y, :x, :y]
    @test nodes_of_type(g, :y) == [2, 4]
    @test_throws ArgumentError nodes_of_type(g, :z)
    @test nv(g) == 4 && ne(g) == 3 && collect(neighbors(g, 2)) == [1, 3] && has_edge(g, 3, 4) && !has_edge(g, 1, 3)
    @test !is_directed(g) && !is_directed(typeof(g)) && degree(g) == [1, 2, 2, 1] && collect(vertices(g)) == 1:4
    @test collect(edges(g)) == collect(edges(path_graph(4))) && eltype(g) == Int
    @test copy(g) == g && copy(g) !== g
    @test global_clustering_coefficient(TypedGraph(complete_graph(4), fill(:x, 4))) == 1.0
    @test occursin("x (2)", sprint(show, g))
    @test_throws ArgumentError TypedGraph(path_digraph(3), [:x, :x, :x])
    @test_throws ArgumentError TypedGraph(path_graph(3), [:x, :y], [1, 2])
    @test_throws ArgumentError TypedGraph(path_graph(2), [:x, :y], [1, 3])
    @test_throws ArgumentError TypedGraph(path_graph(2), [:x, :x], [1, 2])
    # usable as a contact graph: NetworkEpiCore's ExplicitGraph and simulate(model, g)
    @test mean_degree(ExplicitGraph(g)) == 1.5
end

@testset "type sizes by largest remainders" begin
    net3 = sbm_network([:a, :b, :c], [1 / 3, 1 / 3, 1 / 3]; mean_contacts = [1.0 1 1; 1 1 1; 1 1 1])
    @test sample_graph(net3, 10; rng = NO.stable_rng(1))[2].type_sizes == [4, 3, 3]
    @test sample_graph(TYPED, 10; rng = NO.stable_rng(1))[2].type_sizes == [4, 6]
    tiny = sbm_network([:a, :b], [0.999, 0.001]; mean_contacts = [1.0 1.0; 999.0 0.0])
    @test_throws ArgumentError sample_graph(tiny, 10; rng = NO.stable_rng(1))
end

@testset "typed stub matching: exact reciprocal totals" begin
    for (seed, N) in ((1, 10_000), (2, 9_999), (3, 101))
        g, info = sample_graph(TYPED, N; rng = NO.stable_rng(seed))
        @test info.method === :typed_configuration && g isa TypedGraph
        S, M = info.stub_totals, info.matched_stubs
        @test M == M'                                                        # reciprocal
        @test S[1, 2] == S[2, 1] && iseven(S[1, 1])                          # the totals were conditioned exactly
        # B–B degrees are 1 or 5, both odd: with N_B odd the B–B total is odd whatever the draws, so one stub is
        # trimmed (the only trimming here)
        oddBB = isodd(info.type_sizes[2])
        @test isodd(S[2, 2]) == oddBB
        @test info.trimmed_stubs == (oddBB ? 1 : 0)
        @test info.trimmed_blocks == (oddBB ? [(:B, :B)] : Tuple{Symbol, Symbol}[])
        @test M == S .- (oddBB ? [0 0; 0 1] : [0 0; 0 0])
        @test info.candidate_edges == M[1, 1] ÷ 2 + M[1, 2] + M[2, 2] ÷ 2
        @test info.candidate_edges == ne(g) + info.erased
        E = info.block_edges
        @test E == E' && E[1, 1] + E[1, 2] + E[2, 2] == ne(g)
        @test info.mean_contacts ≈ [2E[1, 1] E[1, 2]; E[2, 1] 2E[2, 2]] ./ info.type_sizes
        @test info.target_mean_contacts ≈ mean_contacts(TYPED)
        @test count(==(1), g.type_of) == info.type_sizes[1] && nodes_of_type(g, :A) == 1:info.type_sizes[1]
    end
    # Degenerate laws cannot be conditioned: 3-regular A→B against 2-regular B→A with 3N_A ≠ 2N_B. The excess of the
    # larger side is trimmed, and the matched totals are still exactly reciprocal.
    deg = MultitypeNetwork([:A, :B], [0.4, 0.6], [IndependentDegrees(:B => RegularDegree(3)),
                                                 IndependentDegrees(:A => RegularDegree(2))])
    g, info = sample_graph(deg, 11; rng = NO.stable_rng(1))                  # N_A = 4, N_B = 7: 12 vs 14 stubs
    @test info.type_sizes == [4, 7] && info.stub_totals == [0 12; 14 0]
    @test info.matched_stubs == [0 12; 12 0] && info.trimmed_stubs == 2 && info.trimmed_blocks == [(:A, :B)]
    @test info.trimmed_fraction ≈ 2 / 26
    @test info.candidate_edges == 12 && ne(g) + info.erased == 12
    # SplitDegrees laws that are not `unstructured` couple the blocks of a node: trimmed, still reciprocal
    split = MultitypeNetwork([:A, :B], [0.5, 0.5], [SplitDegrees(PoissonDegree(4.0), :A => 0.5, :B => 0.5),
                                                   SplitDegrees(PoissonDegree(2.0), :A => 1.0, :B => 0.0)])
    g, info = sample_graph(split, 2000; rng = NO.stable_rng(4))
    @test info.method === :typed_configuration && info.matched_stubs == info.matched_stubs'
    @test info.trimmed_stubs == sum(info.stub_totals) - sum(info.matched_stubs)
    @test info.trimmed_stubs < 5 * sqrt(2 * 1000 * 2) + 2                  # O(√N): 5 sd of the block difference
end

@testset "typed degree histograms (χ²) and unbiased block means" begin
    N = 10_000
    g, info = sample_graph(TYPED, N; rng = NO.stable_rng(20))
    pvals = Dict{String, Float64}()
    for (name, a, b, d) in (("A→B Poisson(2)", :A, :B, PoissonDegree(2.0)),
                            ("B→A Bin(4,1/3)", :B, :A, BinomialDegree(4, 1 / 3)),
                            ("B–B {1,5}", :B, :B, EmpiricalDegree(Dict(1 => 0.5, 5 => 0.5))))
        pvals[name] = last(pooled_chisq(typed_degrees(g, a, b), degree_probabilities(d)))
        @test pvals[name] > 1e-3
    end
    kAA = typed_degrees(g, :A, :A)
    @test count(==(3), kAA) >= length(kAA) - 2 * info.erased                  # 3-regular up to erasures
    # SBM (Bernoulli edges): typed degrees Binomial(N_b, M/N_b) ≈ Poisson(M)
    gs, is = sample_graph(SBM2, N; rng = NO.stable_rng(21))
    @test is.method === :stochastic_block_model && is.trimmed_stubs == 0 && is.erased == 0
    for (a, b, m) in ((:a, :a, 6.0), (:a, :b, 2.0), (:b, :a, 2.0), (:b, :b, 4.0))
        pvals["SBM $a→$b"] = last(pooled_chisq(typed_degrees(gs, a, b), degree_probabilities(PoissonDegree(m))))
        @test pvals["SBM $a→$b"] > 1e-3
    end
    @info "typed χ² p-values (N = $N)" pvals
    # Conditioning keeps E[k_{a→b}] unbiased (trimming the excess would lower the A→B mean by E|S_AB − S_BA|/(2N_A),
    # about 0.012, i.e. 5 SE below).
    R = 40
    kAB = [sample_graph(TYPED, N; rng = NO.stable_rng(600 + r))[2].mean_contacts[1, 2] for r in 1:R]
    @test abs(mean(kAB) - 2) < 3 * std(kAB) / sqrt(R)
    kBB = [sample_graph(TYPED, N; rng = NO.stable_rng(600 + r))[2].mean_contacts[2, 2] for r in 1:5]
    @test all(x -> abs(x - 3) < 0.1, kBB)
end

@testset "unstructured networks: exact degrees, types independent of the graph (M10)" begin
    N = 10_000
    g, info = sample_graph(UNSTR, N; rng = NO.stable_rng(8))
    @test info.method === :unstructured && all(==(6), degree(g)) && info.trimmed_stubs == 0
    @test info.type_sizes == [5000, 5000]
    # the number of type-a neighbours of a type-a node is Binomial(6, ≈1/2)
    na = typed_degrees(g, :a, :a)
    @test last(pooled_chisq(na, [binomial(6, j) / 64 for j in 0:6])) > 1e-3
    # a Poisson total gives an Erdős–Rényi graph with types
    g2, info2 = sample_graph(unstructured(PoissonDegree(5.0), [:x, :y, :z], [0.2, 0.3, 0.5]), N; rng = NO.stable_rng(9))
    @test info2.method === :unstructured && info2.type_sizes == [2000, 3000, 5000]
    @test abs(info2.mean_degree - 5) < 0.1
end

@testset "seeding on typed graphs (design §J.6)" begin
    m = OutbreakModel(stratify(sir_model(), [:a, :b]), Dict(:τ => 0.1, :γ => 0.25))
    ia(X) = m.index_of[X]
    g, _ = sample_graph(SBM2, 2000; rng = NO.stable_rng(3))
    types = node_types(g)
    rng = NO.stable_rng(4)
    # Before WP25 a stratified model could only be seeded on an untyped graph by naming every stratum's susceptible
    # class: SeedFraction(:I_a => ρ) is refused there.
    @test_throws ArgumentError NO.initial_state(OutbreakSpec(m, g.graph, SeedFraction(:I_a => 0.01), (0.0, 1.0)), rng)
    st = NO.initial_state(OutbreakSpec(m, g, SeedFraction(:I_a => 0.01, :I_b => 0.02), (0.0, 1.0)), rng)
    @test count(==(ia(:I_a)), st) == 20 && count(==(ia(:I_b)), st) == 40        # fractions of all N (§J.6)
    @test all(types[v] === :a for v in findall(==(ia(:I_a)), st))
    @test all(types[v] === :b for v in findall(==(ia(:I_b)), st))
    @test all(st[v] == (types[v] === :a ? ia(:S_a) : ia(:S_b)) for v in 1:2000 if !(st[v] in (ia(:I_a), ia(:I_b))))
    st = NO.initial_state(OutbreakSpec(m, g, SeedCount(:I_b => 7, :R_a => 3), (0.0, 1.0)), rng)
    @test count(==(ia(:I_b)), st) == 7 && count(==(ia(:R_a)), st) == 3 && count(==(ia(:S_a)), st) == 997
    @test all(types[v] === :a for v in findall(==(ia(:R_a)), st))
    # SeedNodes must respect the node types
    b1 = first(nodes_of_type(g, :b))
    st = NO.initial_state(OutbreakSpec(m, g, SeedNodes(:I_b => [b1]), (0.0, 1.0)), rng)
    @test st[b1] == ia(:I_b) && count(==(ia(:S_b)), st) == 999 && count(==(ia(:S_a)), st) == 1000
    @test_throws ArgumentError NO.initial_state(OutbreakSpec(m, g, SeedNodes(:I_a => [b1]), (0.0, 1.0)), rng)
    # more seeds of a stratum than nodes of its type; a stratified `default`; strata that are not the types
    @test_throws ArgumentError NO.initial_state(OutbreakSpec(m, g, SeedFraction(:I_a => 0.6), (0.0, 1.0)), rng)
    @test_throws ArgumentError NO.initial_state(OutbreakSpec(m, g, SeedFraction(:I_a => 0.01; default = :S_a),
                                                             (0.0, 1.0)), rng)
    m3 = OutbreakModel(stratify(sir_model(), [:a, :b, :c]), Dict(:τ => 0.1, :γ => 0.25))
    @test_throws ArgumentError NO.initial_state(OutbreakSpec(m3, g, SeedFraction(:I_a => 0.01), (0.0, 1.0)), rng)
    # an unstratified model ignores the types
    m1 = OutbreakModel(sir_model(), Dict(:τ => 0.1, :γ => 0.25))
    st = NO.initial_state(OutbreakSpec(m1, g, SeedFraction(:I => 0.05), (0.0, 1.0)), rng)
    @test count(==(m1.index_of[:I]), st) == 100 && count(==(m1.index_of[:S]), st) == 1900
    # a TypedGraph is a graph: simulate(model, g) on the fixed typed graph seeds each stratum on its type
    ens = simulate(stratify(sir_model(), [:a, :b]), g; p = Dict(:τ => 0.1, :γ => 0.25),
                   initial = SeedFraction(:I_b => 0.01), tspan = (0.0, 0.0), nsims = 2, seed = 1, keep = :counts)
    for tr in ens.trajectories
        @test compartment_series(tr, :I_b)[1] == 20 && compartment_series(tr, :I_a)[1] == 0
        @test compartment_series(tr, :S_a)[1] == 1000 && compartment_series(tr, :S_b)[1] == 980
    end
end

@testset "stratified SIR on sampled typed graphs vs the multitype edge-based final size" begin
    # Fresh graph per run, N = 10⁴, conditioned on major outbreaks (§E.2); seeds are fractions of all N (§J.6).
    N, R = 10_000, 200
    cases = (("non-Poisson typed stubs", TYPED, [:A, :B], Dict(:τ => 0.6, :γ => 1.0),
              SeedFraction(:I_A => 0.004, :I_B => 0.006)),
             ("SBM (:sir_sbm2)", SBM2, [:a, :b], Dict(:τ => 0.0955, :γ => 0.25),
              SeedFraction(:I_a => 0.005, :I_b => 0.005)),
             ("unstructured Regular(6) (:sir_unstr2)", UNSTR, [:a, :b], Dict(:τ => 1 / 6, :γ => 0.25),
              SeedFraction(:I_a => 0.005, :I_b => 0.005)))
    for (seed, (name, net, strata, p, init)) in enumerate(cases)
        m = stratify(sir_model(), strata)
        R∞ = final_size(m, net, p; initial = init)
        ens = simulate(m, net; N, p, initial = init, tspan = (0.0, 150.0), nsims = R, seed = 20260926 + 1000seed)
        fs = major(final_size(ens))
        se = std(fs) / sqrt(length(fs))
        z = (mean(fs) - R∞) / se
        @info "typed final size: $name" N R major = length(fs) R∞ mean = mean(fs) se z
        @test length(fs) >= 0.95R
        @test abs(z) < 3
    end
    # :sir_unstr2 reproduces :sir_reg6 (the unit law M10): R∞ = 0.9295
    @test final_size(stratify(sir_model(), [:a, :b]), UNSTR, Dict(:τ => 1 / 6, :γ => 0.25);
                     initial = SeedFraction(:I_a => 0.005, :I_b => 0.005)) ≈ 0.9295 atol = 1e-4
end

# ---------------------------------------------------------------------------------------------------------------
# Clustered networks
# ---------------------------------------------------------------------------------------------------------------

# The N → ∞ limit of the realised transitivity. A finite Newman–Miller graph closes O(1) accidental triangles
# (and loses O(1) to degenerate groups), so T_N = C + c/N + O(N⁻²) with c = O(1). For (2, 2) the triples are fixed
# (every degree is 6), so the per-graph SD of T_N is also O(1/N): at N = 10⁴ the offset c/N ≈ 3·10⁻⁴ is ~4 per-graph
# SDs, whatever the number of graphs. The Richardson extrapolation (N₂T₂ − N₁T₁)/(N₂ − N₁) removes c/N; its SE is
# √(N₂²se₂² + N₁²se₁²)/(N₂ − N₁).
function extrapolated_transitivity(net; N1 = 10_000, R1 = 40, N2 = 100_000, R2 = 10, seed = 0)
    T(N, r) = sample_graph(net, N; rng = NO.stable_rng(seed + r))[2].transitivity
    T1 = [T(N1, r) for r in 1:R1]
    T2 = [T(N2, R1 + r) for r in 1:R2]
    se1, se2 = std(T1) / sqrt(R1), std(T2) / sqrt(R2)
    T∞ = (N2 * mean(T2) - N1 * mean(T1)) / (N2 - N1)
    se∞ = sqrt((N2 * se2)^2 + (N1 * se1)^2) / (N2 - N1)
    return (; T∞, se∞, T1 = mean(T1), se1, T2 = mean(T2), se2)
end

@testset "Newman–Miller: transitivity 2/15 for (2, 2) and the NetworkEpiCore clustering coefficient" begin
    net = ClusteredNetwork(RegularDegree(2), RegularDegree(2))
    C = clustering_coefficient(net)
    @test C ≈ 2 / 15
    N = 10_000
    g, info = sample_graph(net, N; rng = NO.stable_rng(1))
    @test info.method === :newman_miller && info.target_transitivity ≈ 2 / 15
    @test info.triangles_placed == (2N) ÷ 3 && info.trimmed_corners == (2N) % 3 && info.trimmed_stubs == 0
    @test info.transitivity ≈ global_clustering_coefficient(g)                 # cross-check the triangle count
    @test info.connected_triples == sum(k -> k * (k - 1) ÷ 2, degree(g))
    @test info.candidate_edges == N + 3info.triangles_placed && info.candidate_edges == ne(g) + info.erased
    @test count(==(6), degree(g)) >= N - 2info.erased - info.trimmed_corners
    @test info.erased_fraction < 1e-3
    r = extrapolated_transitivity(net; seed = 700)
    @info "Newman–Miller (2, 2) transitivity" C r
    @test abs(r.T∞ - C) < 3 * r.se∞
    @test r.T1 - C > 0 && abs(r.T1 - C) < 1e-3                               # the O(1/N) offset at N = 10⁴

    # Poisson (s, t) (the :sir_clust_pois12 network) and a joint law P[s+1, t+1]
    P = [0.1 0.2; 0.3 0.2; 0.1 0.1]
    for (seed, net) in ((800, ClusteredNetwork(PoissonDegree(1.0), PoissonDegree(2.0))), (900, ClusteredNetwork(P)))
        C = clustering_coefficient(net)
        r = extrapolated_transitivity(net; seed, R1 = 20, R2 = 6)
        @info "Newman–Miller transitivity" net C r
        @test abs(r.T∞ - C) < 3 * r.se∞
        g, info = sample_graph(net, 10_000; rng = NO.stable_rng(seed))
        @test abs(info.mean_degree - mean_degree(net)) < 0.1
        @test info.target_transitivity ≈ C
    end
    # degree histogram of k = s + 2t (χ²)
    net = ClusteredNetwork(PoissonDegree(1.0), PoissonDegree(2.0))
    g, _ = sample_graph(net, 10_000; rng = NO.stable_rng(33))
    pk = degree_pmf_st(degree_probabilities(PoissonDegree(1.0)), degree_probabilities(PoissonDegree(2.0)))
    @test last(pooled_chisq(degree(g), pk)) > 1e-3
    g, _ = sample_graph(ClusteredNetwork(P), 10_000; rng = NO.stable_rng(34))
    pk = zeros(5)
    for i in axes(P, 1), j in axes(P, 2)
        pk[(i - 1) + 2(j - 1) + 1] += P[i, j]
    end
    @test last(pooled_chisq(degree(g), pk)) > 1e-3
end

# ---------------------------------------------------------------------------------------------------------------
# Multiplex and well-mixed
# ---------------------------------------------------------------------------------------------------------------

@testset "multiplex: independent layers on one node set" begin
    net = MPX(:home => RegularDegree(3), :comm => PoissonDegree(5.0))
    N = 10_000
    mg, info = sample_graph(net, N; rng = NO.stable_rng(3))
    @test mg isa MultiplexGraph && length(mg.layers) == 2 && mg.layer_rates == [1.0, 1.0] && nv(mg) == N
    @test info.method === :multiplex && first.(info.layers) == [:home, :comm]
    @test last(info.layers[1]).method === :random_regular && last(info.layers[2]).method === :erdos_renyi
    # the layers are sample_graph of each layer in turn, from the same rng
    rng = NO.stable_rng(3)
    @test mg.layers == [first(sample_graph(net[:home], N; rng)), first(sample_graph(net[:comm], N; rng))]
    total = degree(mg.layers[1]) .+ degree(mg.layers[2])
    @test info.edges == ne(mg.layers[1]) + ne(mg.layers[2]) && info.mean_degree ≈ mean(total)
    @test info.excess_degree ≈ sum(k -> k * (k - 1), total) / sum(total)
    # independent layers: the degrees of a node in two random layers are uncorrelated
    mg2, _ = sample_graph(MPX(:x => PoissonDegree(3.0), :y => NegBinDegree(mean = 4, var = 8)), N;
                          rng = NO.stable_rng(5))
    @test abs(cor(Float64.(degree(mg2.layers[1])), Float64.(degree(mg2.layers[2])))) < 4 / sqrt(N)
    # simulate on the descriptor (DirectSSA supports MultiplexGraph)
    ens = simulate(sir_model(), net; N = 300, p = Dict(:τ => 0.1, :γ => 0.25), initial = SeedCount(:I => 5),
                   tspan = (0.0, 50.0), nsims = 3, seed = 1, algorithm = DirectSSA())
    @test length(ens.trajectories) == 3 && ens.spec.network isa SampledNetwork
end

@testset "well mixed: K_N with rate κ/(N − 1) lumps to mass action" begin
    g, info = sample_graph(WellMixed(5), 200)
    @test g isa MultiplexGraph && length(g.layers) == 1 && ne(g.layers[1]) == 200 * 199 ÷ 2
    @test g.layer_rates == [5 / 199] && info.rate_scale == 5 / 199 && info.method === :complete
    @test_throws ArgumentError sample_graph(WellMixed(5), 1)
    @test_throws ArgumentError sample_graph(WellMixed(5), 10_000)
    # The final-size law on K_N is that of the lumped chain with infection rate τκ S I/(N − 1): exact, computed here.
    N, n0, τ, γ, κ = 40, 4, 0.1, 0.25, 5.0
    m = OutbreakModel(sir_model(), Dict(:τ => τ, :γ => γ); network = WellMixed(κ))
    spec = OutbreakSpec(m, first(sample_graph(WellMixed(κ), N)), SeedCount(:I => n0), (0.0, Inf))
    R = 2000
    ens = simulate_ensemble(spec; nsims = R, seed = 20260926, algorithm = DirectSSA())
    sizes = round.(Int, final_size(ens) .* N)
    law = kn_final_size_law(N, n0, τ * κ / (N - 1), γ)
    @test sum(law) ≈ 1
    exact_mean = sum((n - 1) * law[n] for n in eachindex(law))
    @test abs(mean(sizes) - exact_mean) < 4 * std(sizes) / sqrt(R)
    stat, df, pv = pooled_chisq(sizes, law)
    @info "K_N lumping: final-size law χ²" N R exact_mean mean(sizes) stat df pv
    @test pv > 1e-3
end

# ---------------------------------------------------------------------------------------------------------------
# Reproducibility (design §J.7)
# ---------------------------------------------------------------------------------------------------------------

@testset "graphs are reproducible under StableRNG" begin
    N = 2000
    nets = (ConfigurationNetwork(RegularDegree(4)), ConfigurationNetwork(PoissonDegree(3.0)),
            ConfigurationNetwork(NegBinDegree(mean = 4, var = 8)), ConfigurationNetwork(PowerLawDegree(2.5, 2, 60)),
            ClusteredNetwork(PoissonDegree(1.0), PoissonDegree(2.0)), ClusteredNetwork([0.1 0.2; 0.3 0.4]),
            TYPED, SBM2, UNSTR, MPX(:home => RegularDegree(3), :comm => PoissonDegree(5.0)))
    graphs(x::MultiplexGraph) = x.layers
    graphs(x::TypedGraph) = (x.graph, x.type_of)
    graphs(x::AbstractGraph) = x
    for net in nets
        a, ia = sample_graph(net, N; rng = NO.stable_rng(42))
        Random.seed!(1)                                       # the global RNG plays no part
        b, ib = sample_graph(net, N; rng = NO.stable_rng(42))
        c, _ = sample_graph(net, N; rng = NO.stable_rng(43))
        @test graphs(a) == graphs(b)
        @test graphs(a) != graphs(c)
        @test ia.edges == ib.edges && ia.erased == ib.erased && ia.details == ib.details
    end
    # Pinned edge-list checksums for the constructions that use only StableRNG draws and this package's code
    # (`rand(rng)`, `rand(rng, 1:n)`, table sampling, Fisher–Yates). A change here changes every graph of the reference
    # ensembles (design §E), so it must come with their regeneration.
    pinned = ((ConfigurationNetwork(EmpiricalDegree(Dict(2 => 5 / 6, 10 => 1 / 6))), PINNED_CONFIG),
              (ClusteredNetwork(RegularDegree(2), RegularDegree(2)), PINNED_NM),
              (SBM2, PINNED_SBM),
              (MultitypeNetwork([:A, :B], [0.5, 0.5],
                                [IndependentDegrees(:A => EmpiricalDegree(Dict(1 => 0.5, 3 => 0.5)),
                                                    :B => EmpiricalDegree(Dict(0 => 0.5, 2 => 0.5))),
                                 IndependentDegrees(:A => EmpiricalDegree(Dict(0 => 0.5, 2 => 0.5)),
                                                    :B => RegularDegree(2))]), PINNED_TYPED))
    for (net, ref) in pinned
        g, _ = sample_graph(net, 500; rng = NO.stable_rng(20260926))
        @test edge_checksum(g) == ref
    end
end
