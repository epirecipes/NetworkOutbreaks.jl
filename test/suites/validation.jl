# The scenario runner: conditioning, time alignment, summaries, cache and regeneration (DESIGN_NetworkEpiCore.md
# §G.2 WP27, §E.2, §E.4, §J.7; verified issue N02 and its skeptic's corrected fix).
#
# Acceptance (WP27):
#   - conditioning and alignment on synthetic trajectories; the Wilson interval;
#   - a hash mismatch is refused (and another algorithm or summary revision); the strict-cache mode works: with
#     NETEPI_STRICT_CACHE=1 a missing or stale committed summary is an error even when the user cache holds a valid
#     copy;
#   - a tiny scenario (N = 500, 20 runs) regenerates bit-identical files twice;
#   - scenario_graph(sc, r) reproduces run r's graph.
# Plus the N02 regression against independent references: P(major | 1 seed) against the branching-process extinction
# probability, the conditioned final size against the edge-based final-size equation, and the aligned conditioned
# prevalence against an independent RK4 solution of the edge-based ODE (Miller 2011 with the ρ correction of Miller
# 2014); the unconditioned, unaligned mean is shown to be biased, as in the report.

using NetworkOutbreaks
import NetworkOutbreaks as NO
using Graphs
using Statistics
using Test

const NEC = NO.NetworkEpiCore
const ANCHOR = Dict(:τ => 1 / 6, :γ => 1 / 4)

# ---------------------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------------------

edgeset(g) = Set((min(src(e), dst(e)), max(src(e), dst(e))) for e in edges(g))
same_graph(a, b) = nv(a) == nv(b) && edgeset(a) == edgeset(b)

# The fields of a ScenarioRun that a re-simulation must reproduce.
run_fields(r) = (r.seeds, r.new_infections, r.final_infected, r.infectious_end, r.absorbed, r.peak_time,
                 r.peak_count, r.grid, r.realised)

files_bytes(dir) = Dict(f => read(joinpath(dir, f)) for f in readdir(dir))

# Elementwise |a − b| ≤ atol (NaN matches NaN): the summaries round each statistic (NetworkOutbreaks.SUMMARY_DIGITS).
close_to(a, b; atol) = length(a) == length(b) &&
    all(i -> (isnan(a[i]) && isnan(b[i])) || abs(a[i] - b[i]) <= atol, eachindex(a, b))

# A synthetic SIR trajectory on N nodes of the model `om` ([:S, :I, :R]; transition 1 = infection S → I, 2 =
# recovery I → R): `events` are (time, transition) pairs in time order.
function synthetic_trajectory(om, events; N = 10, seeds = 1, tend = 10.0)
    counts = [N - seeds, seeds, 0]
    times = [0.0]
    cols = [copy(counts)]
    evs = NO.OutbreakEvent[]
    for (k, (t, j)) in enumerate(events)
        if j == 1
            counts[1] -= 1; counts[2] += 1
        else
            counts[2] -= 1; counts[3] += 1
        end
        push!(times, t); push!(cols, copy(counts)); push!(evs, NO.OutbreakEvent(t, j, k))
    end
    times[end] < tend && (push!(times, tend); push!(cols, copy(counts)))
    fic = zeros(Int, N)
    fic[1:(seeds + count(e -> e[2] == 1, events))] .= 1
    return OutbreakTrajectory(om, times, reduce(hcat, cols), fic, evs, nothing, :synthetic)
end

# P(extinction) q of the branching process of SIR on a configuration Poisson(κ) network with per-contact rate τ and
# Exp(γ) infectious periods: q = G(q), G(s) = ∫₀¹ exp(κ(1 − u^{τ/γ})(s − 1)) du (u = e^{−γD}); composite Simpson.
function extinction_probability(κ, τ, γ; n = 20_000)
    a = τ / γ
    G(s) = begin
        h = 1 / n
        acc = 0.0
        for i in 0:n
            u = i * h
            w = (i == 0 || i == n) ? 1 : (isodd(i) ? 4 : 2)
            acc += w * exp(κ * (1 - u^a) * (s - 1))
        end
        acc * h / 3
    end
    q = 0.0
    for _ in 1:2000
        q = G(q)
    end
    return q
end

# The edge-based SIR ODE on a configuration Poisson(κ) network (Miller 2011, compact form, with the seed fraction ρ
# as in Miller 2014): θ' = −τθ + τ(1 − ρ)ψ'(θ)/ψ'(1) + γ(1 − θ), R' = γI, S = (1 − ρ)ψ(θ), I = 1 − S − R,
# ψ(x) = e^{κ(x − 1)}; RK4 with step h, sampled on `tgrid`.
function ebcm_poisson(κ, τ, γ, ρ, tgrid; h = 0.005)
    ψ(x) = exp(κ * (x - 1))
    f(u) = begin
        θ, R = u
        S = (1 - ρ) * ψ(θ)
        (-τ * θ + τ * (1 - ρ) * ψ(θ) + γ * (1 - θ), γ * (1 - S - R))
    end
    u = (1.0, 0.0)
    t = 0.0
    out = Dict(X => Float64[] for X in (:S, :I, :R, :infectious, :cumulative))
    for tk in tgrid
        while t < tk - 1e-12
            dt = min(h, tk - t)
            k1 = f(u)
            k2 = f(u .+ dt / 2 .* k1)
            k3 = f(u .+ dt / 2 .* k2)
            k4 = f(u .+ dt .* k3)
            u = u .+ dt / 6 .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
            t += dt
        end
        S = (1 - ρ) * ψ(u[1])
        I = 1 - S - u[2]
        push!(out[:S], S); push!(out[:I], I); push!(out[:R], u[2]); push!(out[:infectious], I)
        push!(out[:cumulative], 1 - S)
    end
    return ModelCurves(collect(Float64, tgrid), out; label = "EBCM (independent RK4)", representation = :edge_based)
end

# ---------------------------------------------------------------------------------------------------------------

@testset "Wilson interval" begin
    # Reference values: scipy 1.17.1, scipy.stats.binomtest(k, n).proportion_ci(0.95, method = "wilson").
    refs = [(0, 10, 0.0, 0.27753279986288926), (10, 10, 0.7224672001371109, 1.0),
            (81, 263, 0.2552885198782742, 0.36620957698280004), (198, 200, 0.9642782382838232, 0.9972533418664556),
            (122, 200, 0.5409374595716976, 0.6749165686253038), (1, 2000, 8.826773070546787e-5, 0.0028268625032662137),
            (1219, 2000, 0.587928435231776, 0.6306517314136297)]
    for (k, n, lo, hi) in refs
        l, h = wilson_interval(k, n)
        @test isapprox(l, lo; atol = 1e-13)
        @test isapprox(h, hi; atol = 1e-13)
    end
    # closed form at k = 0: the upper limit is z²/(n + z²)
    z = NO.WILSON_Z
    @test wilson_interval(0, 37)[2] ≈ z^2 / (37 + z^2) rtol = 1e-14
    @test wilson_interval(0, 5) == (0.0, wilson_interval(0, 5)[2]) && wilson_interval(5, 5)[2] == 1.0
    @test z ≈ 1.959963984540054
    @test_throws ArgumentError wilson_interval(3, 2)
    @test_throws ArgumentError wilson_interval(0, 0)
    @test_throws ArgumentError wilson_interval(1, 2; z = -1)
end

@testset "conditioning and alignment on synthetic trajectories" begin
    syn = Scenario(:wp27_synthetic; model = sir_model(), network = ConfigurationNetwork(PoissonDegree(2)),
                   params = Dict(:τ => 0.5, :γ => 0.5), initial = SeedFraction(:I => 0.1), tspan = (0, 10),
                   tstep = 1, sim = SimConfig(; N = 10, nsims = 4, condition = MajorOutbreak(0.3),
                                              align = CumulativeCrossing(0.2)))
    om = OutbreakModel(syn.model, syn.params; network = syn.network)
    @test om.compartments == [:S, :I, :R]
    @test om.transitions[1].type === :infection && om.transitions[2].type === :spontaneous
    # run 1: 4 infections (major), the 2nd (the crossing of 2/10) at 1.5; absorbed
    t1 = synthetic_trajectory(om, [(0.5, 1), (1.5, 1), (2.5, 1), (3.5, 1), (4.5, 2), (5.5, 2), (6.5, 2), (7.5, 2),
                                   (8.5, 2)])
    # run 2: one infection (minor), never crosses; absorbed
    t2 = synthetic_trajectory(om, [(0.25, 1), (1.25, 2), (2.25, 2)])
    # run 3: 3 infections, exactly the threshold 3/10 ≥ 0.3 (major), crossing at 3.25; 3 infectious at t_end
    t3 = synthetic_trajectory(om, [(2.25, 1), (3.25, 1), (5.25, 1), (6.75, 2)])
    # run 4: 3 infections (major), crossing at 4.75; absorbed at 9.25
    t4 = synthetic_trajectory(om, [(4.25, 1), (4.75, 1), (5.75, 1), (6.25, 2), (7.25, 2), (8.25, 2), (9.25, 2)])
    ens = scenario_ensemble(syn, [t1, t2, t3, t4])
    @test length(ens) == 4
    @test [r.new_infections for r in ens] == [4, 1, 3, 3]
    @test [r.seeds for r in ens] == [1, 1, 1, 1]
    @test [r.absorbed for r in ens] == [true, true, false, true]
    @test isequal([r.crossing for r in ens], [1.5, NaN, 3.25, 4.75])
    @test [(r.peak_time, r.peak_count) for r in ens] == [(3.5, 5), (0.25, 2), (5.25, 4), (5.75, 4)]

    s = summarise(ens)
    @test s.major == [true, false, true, true]
    @test s.n_major == 3 && s.p_major == 3 / 4 && s.p_major_ci == wilson_interval(3, 4)
    @test s.final_size == [0.5, 0.2, 0.4, 0.4]
    @test s.peak == [(3.5, 0.5), (0.25, 0.2), (5.25, 0.4), (5.75, 0.4)]
    # the median crossing time of the major runs is 3.25, the nearest grid time t* = 3 (index 4)
    al = s.extras[:alignment]
    @test al["reference_time"] == 3.0 && al["reference_index"] == 4 && al["n_crossed"] == 3 && al["level"] == 0.2
    @test s.shifts == [-1.5, 0.0, 0.25, 1.75]
    # aligned prevalence at t = 0, 1, …, 10: run i at its own time crossing_i + (t − 3). Run 1 is missing before its
    # start, run 3 after its end (not absorbed); run 4 is held after its end (absorbed); run 2 is not shifted.
    I1 = [missing, missing, 2, 3, 4, 5, 4, 3, 2, 1, 0]
    I3 = [1, 1, 2, 3, 3, 4, 4, 3, 3, 3, missing]
    I4 = [1, 1, 1, 3, 4, 3, 2, 1, 0, 0, 0]
    I2 = [1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    pointwise(f, runs...) = [begin
                                 v = Float64[x / 10 for x in skipmissing(getindex.(runs, k))]
                                 isempty(v) ? NaN : f(v)
                             end for k in 1:11]
    @test al["cond_n"] == [2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 2]
    @test al["uncond_n"] == al["cond_n"] .+ 1
    @test close_to(s.cond[:I].mean, pointwise(mean, I1, I3, I4); atol = 5e-7)
    @test close_to(s.cond[:I].sd, pointwise(std, I1, I3, I4); atol = 5e-6)
    @test close_to(s.cond[:I].se, pointwise(v -> std(v) / sqrt(length(v)), I1, I3, I4); atol = 5e-7)
    for (f, q) in ((:q025, 0.025), (:q25, 0.25), (:q50, 0.5), (:q75, 0.75), (:q975, 0.975))
        @test close_to(getfield(s.cond[:I], f), pointwise(v -> quantile(v, q), I1, I3, I4); atol = 5e-6)
    end
    @test close_to(s.uncond[:I].mean, pointwise(mean, I1, I2, I3, I4); atol = 5e-7)
    @test s.cond[:I].sd[1] == 0.0 && s.cond[:I].mean[1] == 0.1
    @test s.cond[:cumulative].mean[10] ≈ (5 + 4 + 4) / 30 atol = 1e-6   # run 4 held after its end (absorbed)
    @test s.cond[:cumulative].mean[11] ≈ (5 + 4) / 20 atol = 1e-6       # run 3 missing: runs 1 and 4
    @test close_to(s.cond[:S].mean + s.cond[:I].mean + s.cond[:R].mean, ones(11); atol = 2e-6)
    c = s.extras[:conditioning]
    @test c["largest_unselected"] == 0.1 && c["smallest_selected"] == 0.3 && c["n_selected"] == 3
    @test s.extras[:sampling]["se"] == "iid (sd/sqrt(n))"

    # the unaligned companion: same runs on the absolute clock, no shifts, every run sampled everywhere
    u = unaligned_scenario(syn)
    @test u.id === :wp27_synthetic_unaligned && :unaligned in u.tags && scenario_hash(u) != scenario_hash(syn)
    su = summarise(ens; scenario = u)
    @test su.id === u.id && su.scenario_hash == scenario_hash(u) && isempty(su.shifts)
    @test !haskey(su.extras, :alignment)
    A1 = [1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 0]
    A3 = [1, 1, 1, 2, 3, 3, 4, 3, 3, 3, 3]
    A4 = [1, 1, 1, 1, 1, 3, 4, 3, 2, 1, 0]
    @test close_to(su.cond[:I].mean, pointwise(mean, A1, A3, A4); atol = 5e-7)
    @test_throws ArgumentError unaligned_scenario(u)

    # other conditioning rules on the same runs; the reference time follows the kept runs
    s35 = summarise(ens; scenario = derive(syn; condition = MajorOutbreak(0.35)))
    @test s35.major == [true, false, false, false]
    @test s35.extras[:alignment]["reference_time"] == 2.0            # run 1 alone: 1.5 → t* = 2
    ssurv = summarise(ens; scenario = derive(syn; condition = Survival()))
    @test ssurv.major == [false, false, true, false]
    # the conditioning extras describe the quantity the rule selects on: for Survival the prevalence at t_end (run 3
    # has 3 infectious nodes at t = 10, the others none), not the new infections (0.4 for the discarded run 1)
    cs = ssurv.extras[:conditioning]
    @test cs["measure"] == "prevalence (infectious nodes) at t_end / N"
    @test cs["largest_unselected"] == 0.0 && cs["smallest_selected"] == 0.3 && cs["n_selected"] == 1
    @test c["measure"] == "new infections by t_end (excluding the seeds) / N"
    sall = summarise(ens; scenario = derive(syn; condition = Unconditioned()))
    @test sall.n_major == 4 && isequal(sall.cond, sall.uncond)
    # an empty selection: NaN statistics and a warning, not an error (N02 corrected fix (4))
    snone = @test_logs (:warn, r"no run satisfies") summarise(ens; scenario = derive(syn; condition = MajorOutbreak(0.9)))
    @test snone.n_major == 0 && snone.p_major_ci[1] == 0.0 && all(isnan, snone.cond[:I].mean)
    @test all(==(0), snone.extras[:alignment]["cond_n"])
    @test snone.extras[:alignment]["reference_time"] == 3.0          # falls back to every crossing run
    # the reference index on a one-point grid is its only point (no grid step to divide by)
    @test NO._reference_index([(crossing = 0.3,), (crossing = NaN,)], [true, false], [0.0]) == 1
    @test NO._reference_index([(crossing = NaN,)], [true], [0.0]) == 0

    # the summarising scenario must describe the same simulation, with the ensemble's alignment or none
    @test_throws ArgumentError summarise(ens; scenario = derive(syn; params = Dict(:τ => 0.6)))
    @test_throws ArgumentError summarise(ens; scenario = derive(syn; align = CumulativeCrossing(0.3)))
    # synthetic trajectories must carry their event log
    no_log = OutbreakTrajectory(om, t1.times, t1.counts, t1.final_infection_counts, NO.OutbreakEvent[], nothing, :x)
    @test_throws ArgumentError scenario_ensemble(syn, [no_log])

    # the summary survives the file format unchanged (extras hold TOML values only)
    mktempdir() do d
        save_summary(d, s)
        @test load_summary(d, syn; algorithm_revision = ALGORITHM_REVISION) == s
    end
end

@testset "aligned_curves" begin
    syn = Scenario(:wp27_synthetic2; model = sir_model(), network = ConfigurationNetwork(PoissonDegree(2)),
                   params = Dict(:τ => 0.5, :γ => 0.5), initial = SeedFraction(:I => 0.1), tspan = (0, 10),
                   tstep = 1, sim = SimConfig(; N = 10, nsims = 1, align = CumulativeCrossing(0.2)))
    om = OutbreakModel(syn.model, syn.params; network = syn.network)
    s = summarise(scenario_ensemble(syn, [synthetic_trajectory(om, [(0.5, 1), (1.5, 1), (2.5, 1)])]))
    @test s.extras[:alignment]["reference_time"] == 2.0
    t = collect(0.0:0.5:10.0)
    mc = ModelCurves(t, Dict(:cumulative => 0.1 .+ 0.05 .* t, :I => t ./ 10); label = "linear")
    ac = aligned_curves(mc, s)            # the curve crosses 0.2 at t = 4, so it is shifted by 4 − 2 = 2
    @test ac.t == s.t && ac.label == "linear"
    @test ac.metadata[:alignment_shift] ≈ 2.0
    @test ac[:I] ≈ [min(tk + 2, 10.0) / 10 for tk in s.t]           # held at the last value beyond t = 10
    @test ac[:cumulative][1] ≈ 0.2
    @test_throws ArgumentError aligned_curves(ModelCurves(t, Dict(:I => t)), s)
    @test_throws ArgumentError aligned_curves(ModelCurves(t, Dict(:cumulative => fill(0.1, length(t)))), s)
    su = summarise(scenario_ensemble(syn, [synthetic_trajectory(om, [(0.5, 1)])]); scenario = unaligned_scenario(syn))
    @test_throws ArgumentError aligned_curves(mc, su)
end

@testset "runs: streams, observables and scenario_graph" begin
    sc = derive(scenario(:sir_pois5); N = 500, nsims = 12)
    ens = scenario_ensemble(sc)
    @test ens.realised_names == [:clustering, :erased_fraction, :excess_degree, :mean_degree]
    # the runs are those of simulate(sc) (the same graphs and SSA streams, design §J.7)
    sim = simulate(sc)
    om = OutbreakModel(sc.model, sc.params; network = sc.network)
    iS, iI, iR = om.index_of[:S], om.index_of[:I], om.index_of[:R]
    for r in 1:sc.sim.nsims
        run = ens[r]
        tr = sim[r]
        @test tr.times == collect(sc.tgrid)
        @test run.grid[1, :] == tr.counts[iS, :] && run.grid[2, :] == tr.counts[iI, :] && run.grid[3, :] == tr.counts[iR, :]
        @test run.final_infected / sc.sim.N == final_size(tr)
        # SIR: :cumulative is 1 − S, and it ends at final_size
        @test run.grid[5, :] == sc.sim.N .- run.grid[1, :]
        @test run.seeds + run.new_infections == run.final_infected
    end
    # threads give the same ensemble
    ensp = scenario_ensemble(sc; parallel = true)
    @test all(run_fields(a) == run_fields(b) for (a, b) in zip(ens, ensp))

    # scenario_graph(sc, r) is run r's graph: the §J.7 stream, the graph that scenario_run and a direct
    # re-simulation use, and the realised statistics recorded for run r
    for r in (1, 7, 12)
        g = scenario_graph(sc, r)
        @test same_graph(g, first(sample_graph(sc.network, sc.sim.N; rng = NO.stable_rng(sc.sim.base_seed + r))))
        @test ens[r].realised[4] == 2ne(g) / nv(g)
        @test ens[r].realised[1] == global_clustering_coefficient(g)
        @test run_fields(NO._run_record(NO._run_plan(sc), scenario_run(sc, r), r, r, ens[r].realised)) ==
              run_fields(ens[r])
        direct = simulate(OutbreakSpec(om, g, sc.initial, sc.tspan); algorithm = NextReaction(),
                          seed = sc.sim.base_seed + 2^32 + r)
        @test final_size(direct) == ens[r].final_infected / sc.sim.N
    end
    @test !same_graph(scenario_graph(sc, 1), scenario_graph(sc, 2))
    @test_throws ArgumentError scenario_graph(sc)                    # :per_run: every run has its own graph
    @test_throws ArgumentError scenario_graph(sc, 13)

    # quenched: one graph for every run
    fx = derive(scenario(:sir_reg6_fixed); N = 300, nsims = 6)
    efx = scenario_ensemble(fx)
    g = scenario_graph(fx)
    @test all(r -> same_graph(scenario_graph(fx, r), g), 1:6) && all(r -> efx[r].graph == 1, 1:6)
    @test all(run -> run.realised == efx[1].realised, efx)
    @test run_fields(NO._run_record(NO._run_plan(fx), scenario_run(fx, 4), 4, 1, efx[4].realised)) == run_fields(efx[4])
    @test length(unique(r.final_infected for r in efx)) > 1          # different SSA streams on the same graph

    # a pool of 3 graphs: run r on graph mod1(r, 3), between-graph standard errors
    pl = derive(scenario(:sir_pois5); N = 400, nsims = 9, graphs = (:pool, 3))
    epl = scenario_ensemble(pl)
    @test [r.graph for r in epl] == [1, 2, 3, 1, 2, 3, 1, 2, 3]
    @test same_graph(scenario_graph(pl, 5), scenario_graph(pl, 2))
    spl = summarise(epl)
    @test spl.extras[:sampling]["se"] == "cluster (between-graph)" && spl.extras[:sampling]["pool_size"] == 3
    k = 60                                                             # t = 15, mid-epidemic
    v = [r.grid[2, k] / pl.sim.N for r in epl]
    μ = mean(v)
    e = [sum(v[i] - μ for i in j:3:9) for j in 1:3]
    @test spl.uncond[:I].se[k] ≈ sqrt(3 / 2 * sum(abs2, e)) / 9 atol = 1e-6
    @test spl.uncond[:I].mean[k] ≈ μ atol = 1e-6

    # a dynamic network: run r starts from graph r of the base and rewires its own copy
    nex = derive(scenario(:sir_ne_reg6_eta1); N = 300, nsims = 3)
    ene = scenario_ensemble(nex)
    @test ene.realised_names == [:clustering, :excess_degree, :mean_degree]
    g2 = scenario_graph(nex, 2)
    @test g2 isa SimpleGraph && all(==(6), degree(g2))
    @test same_graph(g2, first(sample_graph(nex.network.base, 300; rng = NO.stable_rng(nex.sim.base_seed + 2))))
    @test run_fields(NO._run_record(NO._run_plan(nex), scenario_run(nex, 2), 2, 2, ene[2].realised)) == run_fields(ene[2])

    # a multitype network: typed graphs, each stratum seeded on its own type (§J.6)
    sb = derive(scenario(:sir_sbm2); N = 1000, nsims = 3)
    esb = scenario_ensemble(sb)
    gt = scenario_graph(sb, 3)
    @test gt isa TypedGraph
    @test run_fields(NO._run_record(NO._run_plan(sb), scenario_run(sb, 3), 3, 3, esb[3].realised)) == run_fields(esb[3])
    @test all(r -> r.seeds == 10, esb)

    # well mixed with MassActionSSA: no graph, no realised statistics
    wm = derive(scenario(:sir_wm5); N = 500, nsims = 5)
    ewm = scenario_ensemble(wm)
    @test isempty(ewm.realised_names) && all(r -> r.graph == 1, ewm)
    @test_throws ArgumentError scenario_graph(wm, 1)
    @test run_fields(NO._run_record(NO._run_plan(wm), scenario_run(wm, 3), 3, 1, Float64[])) == run_fields(ewm[3])
    @test summarise(ewm).provenance["algorithm"] == "MassActionSSA"

    # an exit: vaccinated nodes are not infections, so :cumulative is not 1 − S
    vx = derive(scenario(:sir_vax_pois5); N = 500, nsims = 4)
    evx = scenario_ensemble(vx)
    ov = findfirst(==(:cumulative), vx.observables)
    for run in evx
        @test run.grid[ov, end] == run.seeds + run.new_infections == run.final_infected
        @test run.grid[ov, end] < vx.sim.N - run.grid[1, end]
    end
end

@testset "SIS: survival conditioning and the reinfection histogram" begin
    sis = derive(scenario(:sis_reg3); N = 300, nsims = 10)
    @test sis.sim.condition isa Survival
    ens = scenario_ensemble(sis)
    s = summarise(ens)
    @test s.major == BitVector([r.infectious_end > 0 for r in ens])
    oc = findfirst(==(:cumulative), sis.observables)
    for run in ens
        @test run.grid[oc, end] == run.seeds + run.new_infections       # cumulative incidence, reinfections too
        run.infectious_end > 0 && @test run.new_infections > run.final_infected - run.seeds   # reinfections
    end
    @test s.n_major >= 1
    c = s.extras[:conditioning]                                          # Survival selects on the prevalence at t_end
    @test c["smallest_selected"] == minimum(r.infectious_end for r in ens if r.infectious_end > 0) / sis.sim.N
    @test isequal(c["largest_unselected"], s.n_major == length(ens) ? NaN : 0.0)
    h = s.extras[:reinfection_histogram]
    @test sum(h["uncond"]) ≈ 1 atol = 1e-4
    @test sum(h["cond"]) ≈ 1 atol = 1e-4
    @test h["cond"][1] < 0.5                                             # most nodes were infected at least once
    @test length(h["cond"]) > 2                                          # and some several times
end

@testset "masking, seeds and subcritical conditioning (N02 corrected fix)" begin
    base = Scenario(:wp27_mask; model = sir_model(), network = ConfigurationNetwork(PoissonDegree(5)), params = ANCHOR,
                    initial = SeedFraction(:I => 0.05), tspan = (0, 40), tstep = 0.5,
                    sim = SimConfig(; N = 1000, nsims = 20, align = CumulativeCrossing(0.03)))
    short = derive(base; tspan = (0, 8))
    el = scenario_ensemble(base)          # 5% seeds and alignment at 3% of incidence: no error
    es = scenario_ensemble(short)
    nl, ns = length(base.tgrid), length(short.tgrid)
    Δ = step(short.tgrid)
    nmasked = 0
    for (a, b) in zip(el, es)
        @test b.grid == a.grid[:, 1:ns]                                  # the same runs up to t = 8
        isnan(b.crossing) && continue
        @test b.crossing == a.crossing
        for j in -(ns - 1):(ns - 1)
            x, y = b.aligned[:, j + ns], a.aligned[:, j + nl]
            if b.crossing + j * Δ > 8 && !b.absorbed
                @test all(==(-1), x)                                     # masked, never held
                nmasked += 1
            else
                @test x == y                                             # equal to the long runs where observed
            end
        end
    end
    @test nmasked > 0
    ss = summarise(es)
    @test all(!isnan, ss.cond[:I].mean) && minimum(ss.extras[:alignment]["cond_n"]) >= 1

    # a subcritical epidemic (R₀ ≈ 0.24) from 5% seeds: final_size ≥ 5% in every run, but P(major) ≈ 0 because the
    # rule counts new infections only
    sub = derive(base; id = :wp27_subcritical, params = Dict(:τ => 0.0126), tspan = (0, 60), align = NoAlignment(),
                 N = 500, nsims = 40)
    s = summarise(scenario_ensemble(sub))
    @test all(>=(0.05), s.final_size)
    @test s.p_major < 0.1
end

@testset "N02: conditioned and aligned ensembles against independent references" begin
    N = 5000
    one = Scenario(:wp27_one_seed; model = sir_model(), network = ConfigurationNetwork(PoissonDegree(5)),
                   params = ANCHOR, initial = SeedFraction(:I => 1 / N), tspan = (0, 100), tstep = 0.5,
                   sim = SimConfig(; N, nsims = 300, align = CumulativeCrossing(0.02)), expected = (:final_size,))
    ens = scenario_ensemble(one; parallel = true)
    s = summarise(ens)
    # P(major | 1 seed) = 1 − q for Exp(γ) infectious periods (not the bond-percolation value R∞): q = 0.3906
    q = extinction_probability(5.0, 1 / 6, 1 / 4)
    @test isapprox(q, 0.39059; atol = 2e-5)
    se = sqrt((1 - q) * q / s.nsims)
    @test abs(s.p_major - (1 - q)) < 4se
    c = s.extras[:conditioning]
    @test c["largest_unselected"] < 0.02 && c["smallest_selected"] > 0.5  # the 5% threshold sits in the gap
    # the conditioned final size is the edge-based final size (NetworkEpiCore's final-size equation)
    fs = s.final_size[s.major]
    @test abs(mean(fs) - one.expected[:final_size]) < 0.01
    # the aligned, conditioned prevalence is the edge-based curve shifted to the same crossing
    eb = ebcm_poisson(5.0, 1 / 6, 1 / 4, 1 / N, one.tgrid)
    tab = compare(s, aligned_curves(eb, s); observables = [:I, :cumulative])
    @test tab["EBCM (independent RK4)", :I].D∞ < 0.008
    @test abs(maximum(s.cond[:I].mean) / maximum(eb[:I]) - 1) < 0.03
    # without conditioning and alignment the mean is biased (the report's symptom), and conditioning alone still
    # smears the peak
    su = summarise(ens; scenario = unaligned_scenario(one))
    @test maximum(s.uncond[:I].mean) < maximum(s.cond[:I].mean)
    @test maximum(summarise(ens; scenario = derive(unaligned_scenario(one); condition = Unconditioned())).cond[:I].mean) <
          0.8 * maximum(eb[:I])
    @test maximum(su.cond[:I].mean) < maximum(s.cond[:I].mean)
    @test compare(su, eb; observables = [:I])["EBCM (independent RK4)", :I].D∞ >
          2 * tab["EBCM (independent RK4)", :I].D∞
end

@testset "regeneration is bit-identical; cache, hashes and strict mode" begin
    tiny = derive(scenario(:sir_pois5); id = :wp27_tiny, N = 500, nsims = 20)
    tiny_al = Scenario(:wp27_tiny_aligned; model = sir_model(), network = ConfigurationNetwork(PoissonDegree(5)),
                       params = ANCHOR, initial = SeedFraction(:I => 2 / 500), tspan = (0, 80), tstep = 0.5,
                       sim = SimConfig(; N = 500, nsims = 20, align = CumulativeCrossing(0.02)))
    mktempdir() do root
        d1, d2, d3, cache = (joinpath(root, x) for x in ("d1", "d2", "d3", "cache"))
        res = regenerate_scenarios([tiny, tiny_al]; dir = d1, parallel = false, io = nothing)
        @test [r.id for r in res] == [:wp27_tiny, :wp27_tiny_aligned, :wp27_tiny_aligned_unaligned]
        @test length(readdir(d1)) == 9
        b1 = files_bytes(d1)
        regenerate_scenarios([tiny, tiny_al]; dir = d2, parallel = true, io = nothing)
        @test files_bytes(d2) == b1                                       # bit-identical, also with threads
        regenerate_scenarios([tiny, tiny_al]; dir = d1, io = nothing)
        @test files_bytes(d1) == b1                                       # and again in place
        toml = read(joinpath(d1, summary_basename(tiny.id, scenario_hash(tiny)) * ".toml"), String)
        @test !occursin(r"date|wall|time =", toml)                         # no timestamp in the provenance
        @test isempty(missing_scenario_summaries([tiny, tiny_al]; dir = d1))
        @test length(missing_scenario_summaries([tiny, tiny_al]; dir = d3)) == 3
        # each summary once: repeated ids, and a companion also given on its own
        @test length(missing_scenario_summaries([tiny, tiny_al, tiny, unaligned_scenario(tiny_al)]; dir = d3)) == 3

        # committed summaries load, identical to a fresh summary
        s = scenario_summary(tiny; dir = d1)
        @test s == summarise(scenario_ensemble(tiny))
        @test scenario_summary(unaligned_scenario(tiny_al); dir = d1).id === :wp27_tiny_aligned_unaligned

        # a hash mismatch is refused: another scenario with the same id, a file whose stored hash differs, and a
        # summary from another algorithm revision
        other = derive(tiny; id = :wp27_tiny, nsims = 21)
        err = try
            scenario_summary(other; dir = d1); nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("stale", sprint(showerror, err))
        mkpath(d3)
        src = summary_basename(tiny.id, scenario_hash(tiny))
        dst = summary_basename(other.id, scenario_hash(other))
        for ext in (".toml", ".curves.csv", ".runs.csv")
            cp(joinpath(d1, src * ext), joinpath(d3, dst * ext))
        end
        err = try
            scenario_summary(other; dir = d3); nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("summarises scenario hash", sprint(showerror, err))
        old = EnsembleSummary(s.id, s.scenario_hash, "0", s.N, s.nsims, s.n_major, s.p_major, s.p_major_ci, s.t,
                              s.observables, s.cond, s.uncond, s.final_size, s.peak, s.major, s.shifts, s.realised,
                              s.extras, s.provenance)
        d4 = joinpath(root, "d4")
        save_summary(d4, old)
        err = try
            scenario_summary(tiny; dir = d4); nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("algorithm revision", sprint(showerror, err))
        # and one made by other summarising code (another SUMMARY_REVISION in the provenance, or none), which
        # --check (missing_scenario_summaries) reports too
        d5, d6 = joinpath(root, "d5"), joinpath(root, "d6")
        with_provenance(p) = EnsembleSummary(s.id, s.scenario_hash, s.algorithm_revision, s.N, s.nsims, s.n_major,
                                             s.p_major, s.p_major_ci, s.t, s.observables, s.cond, s.uncond,
                                             s.final_size, s.peak, s.major, s.shifts, s.realised, s.extras, p)
        @test s.provenance["summary_revision"] == NO.SUMMARY_REVISION
        save_summary(d5, with_provenance(merge(s.provenance, Dict("summary_revision" => "0"))))
        save_summary(d6, with_provenance(filter(p -> first(p) != "summary_revision", s.provenance)))
        for (d, rev) in ((d5, "0"), (d6, "none"))
            err = try
                scenario_summary(tiny; dir = d); nothing
            catch e
                e
            end
            @test err isa ArgumentError &&
                  occursin("summary revision $(rev), not $(NO.SUMMARY_REVISION)", sprint(showerror, err))
            bad = missing_scenario_summaries([tiny]; dir = d)
            @test length(bad) == 1 && first(only(bad)) === :wp27_tiny && occursin("summary revision", last(only(bad)))
        end

        # policies and the strict cache
        empty = joinpath(root, "empty")
        @test_throws ArgumentError scenario_summary(tiny; dir = empty, cache_dir = cache)
        withenv("NETEPI_STRICT_CACHE" => "1") do
            @test_throws ArgumentError scenario_summary(tiny; policy = :auto, dir = empty, cache_dir = cache)
            @test_throws ArgumentError scenario_summary(tiny; policy = :auto, dir = d4, cache_dir = cache)  # stale
            @test !isdir(cache)                                            # nothing was computed
        end
        withenv("NETEPI_STRICT_CACHE" => nothing) do
            a = @test_logs (:info, r"no valid summary") scenario_summary(tiny; policy = :auto, dir = empty,
                                                                          cache_dir = cache)
            @test a == s && isfile(joinpath(cache, src * ".toml"))
            @test (@test_logs scenario_summary(tiny; policy = :auto, dir = empty, cache_dir = cache)) == s  # cached
            @test scenario_summary(tiny; policy = :auto, dir = d4, cache_dir = cache) == s  # stale committed: cache
            @test scenario_summary(tiny; policy = :auto, dir = d5, cache_dir = cache) == s
            # strict mode uses the committed summaries only: a valid copy in the user cache (which a restored depot
            # cache could supply) does not stand in for a missing or stale committed summary
            withenv("NETEPI_STRICT_CACHE" => "1") do
                for (d, why) in ((empty, "no summary"), (d3, "other (stale) hashes"), (d4, "algorithm revision"),
                                 (d5, "summary revision 0"), (d6, "summary revision none"))
                    err = try
                        scenario_summary(tiny; policy = :auto, dir = d, cache_dir = cache); nothing
                    catch e
                        e
                    end
                    msg = err === nothing ? "" : sprint(showerror, err)
                    @test err isa ArgumentError && occursin("NETEPI_STRICT_CACHE", msg) && occursin(why, msg)
                end
                @test_throws ArgumentError scenario_summary(tiny; policy = :auto, dir = empty, cache_dir = nothing)
                @test scenario_summary(tiny; policy = :auto, dir = d1, cache_dir = cache) == s   # committed: fine
                @test_logs scenario_summary(tiny; policy = :auto, dir = d1, cache_dir = cache)
            end
            @test scenario_summary(tiny; policy = :recompute, cache_dir = nothing) == s
        end
        @test_throws ArgumentError scenario_summary(tiny; policy = :sometimes, dir = d1)

        # regeneration removes the stale files of the regenerated ids
        cp(joinpath(d3, dst * ".toml"), joinpath(d1, dst * ".toml"))
        regenerate_scenarios([tiny]; dir = d1, io = nothing)
        @test !isfile(joinpath(d1, dst * ".toml")) && isfile(joinpath(d1, src * ".toml"))

        # every summary is written once: a companion given on its own (before or after its aligned scenario) is
        # written from the aligned scenario's ensemble, and repeated ids are simulated once
        ual = unaligned_scenario(tiny_al)
        plan = NO._regeneration_plan([ual, tiny_al, tiny, tiny_al, tiny], true)
        @test [first(p).id for p in plan] == [:wp27_tiny_aligned, :wp27_tiny]
        @test [[t.id for t in last(p)] for p in plan] ==
              [[:wp27_tiny_aligned, :wp27_tiny_aligned_unaligned], [:wp27_tiny]]
        @test [first(p).id for p in NO._regeneration_plan([ual, tiny_al], false)] == [:wp27_tiny_aligned_unaligned,
                                                                                      :wp27_tiny_aligned]
        d7 = joinpath(root, "d7")
        res = regenerate_scenarios([ual, tiny_al, tiny_al]; dir = d7, io = nothing)
        @test [r.id for r in res] == [:wp27_tiny_aligned, :wp27_tiny_aligned_unaligned]
        b7 = files_bytes(d7)
        @test length(b7) == 6 && all(f -> b7[f] == b1[f], keys(b7))    # the same files as a regeneration of both
    end
    @test isdir(dirname(scenario_data_dir())) && endswith(scenario_data_dir(), joinpath("data", "scenarios"))
    withenv("NETEPI_CACHE_DIR" => "/some/where") do
        @test scenario_cache_dir() == "/some/where"
    end
end
