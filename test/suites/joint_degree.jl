# The joint-degree-matrix ("2K") sampler for DegreeCorrelatedNetwork (DESIGN_NetworkEpiCore.md §K WP36a, §C.3).
#
# Checks: the degree classes get their largest-remainder sizes and every node its class degree (up to erasure and the
# reported trimming); the integer joint degree matrix is symmetric with an even diagonal, has the class stub totals as
# row sums, stays within O(1) of N⟨k⟩e and never uses a block with e_kl = 0; the realised joint degree matrix and
# Newman assortativity match the descriptor; each node's stubs are split over the partner classes uniformly (χ²
# against the binomial limit); graphs are reproducible under stable_rng (pinned checksum); the TypedGraph types are
# those of MultitypeNetwork(net) and seed stratified models within their class; and the simulated final size on
# :sir_dc_bim_r05 / :sir_dc_bim_rn05 (fresh graph per run, conditioned on a major outbreak, §E.2) matches
# NetworkEpiCore's degree-correlated final size, while the uncorrelated one (:sir_bim) is far off.

using NetworkOutbreaks
import NetworkOutbreaks as NO
using Graphs
using Random
using Statistics
using Test

const BIMODAL = EmpiricalDegree(Dict(2 => 5 / 6, 10 => 1 / 6))

# Pinned edge-list checksum (`edge_checksum`) of sample_graph(degree_correlated(BIMODAL; r = 0.5), 500;
# rng = NO.stable_rng(20260926)), recorded when WP36a landed.
const PINNED_JOINT_DEGREE = 549327741140689623

function edge_checksum(g::AbstractGraph)
    h = Int128(0)
    m = Int128(2)^61 - 1
    for e in edges(g)
        h = (h * 1_000_003 + src(e) * 7919 + dst(e)) % m
    end
    return Int64(h)
end

# Upper 0.1% point of χ²(df) by the Wilson–Hilferty approximation (z = 3.0902).
chisq_crit(df) = df * (1 - 2 / (9df) + 3.0902 * sqrt(2 / (9df)))^3

# Pearson χ² of integer counts against the Binomial(n, π) pmf, pooling bins to expected counts ≥ 5.
function binomial_chisq(x::AbstractVector{<:Integer}, n::Int, π::Float64)
    pmf = [binomial(n, j) * π^j * (1 - π)^(n - j) for j in 0:n]
    obs = [count(==(j), x) for j in 0:n]
    N = length(x)
    O, E = Int[], Float64[]
    o, e = 0, 0.0
    for j in 0:n
        o += obs[j + 1]
        e += N * pmf[j + 1]
        if e >= 5 && N * sum(pmf[(j + 2):end]; init = 0.0) >= 5
            push!(O, o)
            push!(E, e)
            o, e = 0, 0.0
        end
    end
    if o > 0 || e > 0
        O[end] += o
        E[end] += e
    end
    return sum((O .- E) .^ 2 ./ E), length(E) - 1
end

@testset "joint-degree (2K) sampler" begin
    @testset "exact classes, degrees and joint degree matrix" begin
        N = 10_000
        nets = [degree_correlated(BIMODAL; r) for r in (0.0, 0.5, -0.5)]
        push!(nets, degree_correlated(PoissonDegree(5.0); r = 0.3))              # 28 classes, empty tail classes
        # degrees 0, 1, 3, 6 with ⟨k⟩ = 3.2, so the edge-end marginals are q = (1/16, 3/8, 9/16) on degrees 1, 3, 6
        push!(nets, degree_correlated([0.1, 0.2, 0.0, 0.4, 0.0, 0.0, 0.3],
                                      [0 0 0 0 0 0 0; 0 0.0125 0 0.025 0 0 0.025; 0 0 0 0 0 0 0;
                                       0 0.025 0 0.2 0 0 0.15; 0 0 0 0 0 0 0; 0 0 0 0 0 0 0;
                                       0 0.025 0 0.15 0 0 0.3875]))
        for (s, net) in enumerate(nets)
            g, info = sample_graph(net, N; rng = NO.stable_rng(100 + s))
            K = length(net.degrees)
            @test g isa TypedGraph && nv(g) == N && !has_self_loops(g.graph)
            @test info isa GraphInfo && info.method === :joint_degree && info.descriptor == net
            @test info.types == MultitypeNetwork(net).types == g.types
            # class sizes: largest remainders
            @test sum(info.class_sizes) == N
            @test all(abs.(info.class_sizes .- N .* net.probabilities) .< 1)
            @test [length(nodes_of_type(g, a)) for a in g.types] == info.class_sizes
            # every node has its class degree, except the nodes touched by erasure or trimming
            deg = degree(g)
            cls = g.type_of
            off = count(v -> deg[v] != net.degrees[cls[v]], 1:N)
            @test off <= 2info.erased + info.trimmed_stubs
            @test all(v -> deg[v] <= net.degrees[cls[v]], 1:N)
            # the integer joint degree matrix
            m = info.edge_end_counts
            @test m == m' && all(iseven, [m[i, i] for i in 1:K]) && all(>=(0), m)
            @test info.stub_totals == info.class_sizes .* net.degrees
            short = info.stub_totals .- vec(sum(m; dims = 2))        # row sums: the stub totals, less the trimmed
            @test all(>=(0), short) && sum(short) == info.trimmed_stubs
            @test sum(m) == 2info.candidate_edges
            @test info.trimmed_stubs <= 1                                    # an odd stub total at most
            @test maximum(abs.(m .- info.target_edge_ends)) <= K + 2          # within O(1) of N⟨k⟩e
            @test all(m[i, j] == 0 for i in 1:K, j in 1:K if net.edge_ends[i, j] == 0)
            @test info.target_edge_ends ≈ sum(info.stub_totals) .* net.edge_ends
            # the realised edge ends, recounted from the graph
            ends = zeros(Int, K, K)
            for e in edges(g)
                a, b = cls[src(e)], cls[dst(e)]
                ends[a, b] += 1
                ends[b, a] += 1
            end
            @test ends == info.realised_edge_ends
            @test sum(m) - sum(ends) == 2info.erased
            @test maximum(abs.(info.realised_joint_degree .- net.edge_ends)) < 2e-3
            # Newman assortativity of the realised graph: our formula is Graphs.jl's, and it matches the descriptor
            # (the Poisson descriptor includes tail classes that get no node at N = 10⁴, hence the wider tolerance)
            @test info.assortativity ≈ Graphs.assortativity(g.graph) atol = 1e-12
            @test abs(info.assortativity - degree_assortativity(net)) < (s == 4 ? 0.015 : 0.01)
            @test abs(info.mean_degree - mean_degree(net)) < 0.01 * mean_degree(net)
            @test abs(info.excess_degree - excess_degree(net)) < 0.02 * excess_degree(net)
            @test info.erased_fraction < 2e-3
        end
    end

    @testset "each node's stubs are split uniformly over the partner classes (χ²)" begin
        # at N = 10⁵, the number of degree-10 neighbours of a class-k node is hypergeometric, within 10⁻³ of
        # Binomial(k, Q(k10 | k)) in variance: Q(k10 | k2) = 0.25, Q(k10 | k10) = 0.75 at r = 0.5
        net = degree_correlated(BIMODAL; r = 0.5)
        g, info = sample_graph(net, 100_000; rng = NO.stable_rng(7))
        Q = net.edge_ends ./ sum(net.edge_ends; dims = 2)
        types = node_types(g)
        deg = degree(g)
        for (a, i) in ((:k2, 1), (:k10, 2))
            k = net.degrees[i]
            vs = [v for v in nodes_of_type(g, a) if deg[v] == k]
            x = [count(u -> types[u] === :k10, neighbors(g, v)) for v in vs]
            @test mean(x) ≈ k * Q[i, 2] rtol = 0.01
            stat, df = binomial_chisq(x, k, Q[i, 2])
            @test df >= 1 && stat < chisq_crit(df)
        end
    end

    @testset "extreme mixing, trimming and small N" begin
        # perfectly assortative {3, 7} (r = 1): no edge between the classes; 501 nodes of degree 3 have an odd
        # number of stubs, one of which is trimmed
        net = degree_correlated(EmpiricalDegree(Dict(3 => 0.5, 7 => 0.5)); r = 1.0)
        g, info = sample_graph(net, 1001; rng = NO.stable_rng(2))
        @test info.class_sizes == [501, 500] && info.trimmed_stubs == 1
        @test info.edge_end_counts[1, 2] == 0 && info.realised_edge_ends[1, 2] == 0
        @test all(e -> g.type_of[src(e)] == g.type_of[dst(e)], edges(g))
        # perfectly disassortative bimodal (r = −1, e_kk = 0 up to rounding): no edge within a class; the stub totals
        # of the two classes differ by 2 at N = 10001, and those stubs are trimmed
        net = degree_correlated(BIMODAL; r = -1.0)
        g, info = sample_graph(net, 10_001; rng = NO.stable_rng(4))
        @test info.stub_totals == [16668, 16670] && info.trimmed_stubs == 2
        @test all(e -> g.type_of[src(e)] != g.type_of[dst(e)], edges(g))
        @test info.assortativity ≈ -1 atol = 1e-3
        # tiny graphs and empty classes
        net = degree_correlated(BIMODAL; r = 0.5)
        for N in (1, 2, 5, 7)
            g, info = sample_graph(net, N; rng = NO.stable_rng(N))
            @test nv(g) == N && sum(info.class_sizes) == N && !has_self_loops(g.graph)
        end
        @test_throws ArgumentError sample_graph(net, 0)
    end

    @testset "reproducibility" begin
        net = degree_correlated(BIMODAL; r = 0.5)
        g1, i1 = sample_graph(net, 500; rng = NO.stable_rng(20260926))
        g2, i2 = sample_graph(net, 500; rng = NO.stable_rng(20260926))
        g3, _ = sample_graph(net, 500; rng = NO.stable_rng(20260927))
        @test g1 == g2 && i1 == i2
        @test g1 != g3
        @test edge_checksum(g1) == PINNED_JOINT_DEGREE
    end

    @testset "seeding on the degree classes (§J.6)" begin
        net = degree_correlated(BIMODAL; r = 0.5)
        g, _ = sample_graph(net, 2000; rng = NO.stable_rng(9))
        p = Dict(:τ => 1 / 6, :γ => 1 / 4)
        # an unstratified model is seeded uniformly over all nodes (both classes get seeds)
        om = OutbreakModel(sir_model(), p)
        tr = simulate(OutbreakSpec(om, g, SeedFraction(:I => 0.05), (0.0, 1.0)); seed = 1, keep = :events)
        @test NO.state_at(tr, 0.0)[om.index_of[:I]] == 100
        # a model stratified over the degree classes seeds each stratum within its class
        st = stratify(sir_model(), MultitypeNetwork(net).types)
        omst = OutbreakModel(st, p)
        tr = simulate(OutbreakSpec(omst, g, SeedFraction(:I_k10 => 0.02), (0.0, 1.0)); seed = 3, keep = :events)
        counts = NO.state_at(tr, 0.0)
        @test counts[omst.index_of[:I_k10]] == 40 && counts[omst.index_of[:I_k2]] == 0
        @test counts[omst.index_of[:S_k10]] == length(nodes_of_type(g, :k10)) - 40
        @test counts[omst.index_of[:S_k2]] == length(nodes_of_type(g, :k2))
    end

    @testset "final size on :sir_dc_bim_r05 and :sir_dc_bim_rn05 against NetworkEpiCore" begin
        for id in (:sir_dc_bim_r05, :sir_dc_bim_rn05)
            sc = scenario(id)
            ens = simulate(sc; nsims = 60)
            fs = final_size(ens)
            ρ = 0.01
            major = filter(x -> x - ρ >= 0.05, fs)                  # MajorOutbreak(0.05), excluding the seeds
            @test length(major) >= 55
            μ, se = mean(major), std(major) / sqrt(length(major))
            ref = final_size(sc.model, sc.network, sc.params; initial = sc.initial)
            uncorrelated = final_size(sc.model, ConfigurationNetwork(BIMODAL), sc.params; initial = sc.initial)
            @info "2K sampler: simulated vs degree-correlated final size, $(id)" N = sc.sim.N runs = 60 major = length(major) mean = μ se ref uncorrelated
            @test abs(μ - ref) < 3se + 0.003
            @test abs(μ - uncorrelated) > 0.03                          # the correlations matter
        end
    end
end
