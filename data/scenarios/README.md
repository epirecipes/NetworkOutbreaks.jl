# Committed reference summaries

This directory holds the reference stochastic ensembles of the NetworkEpiCore validation scenarios
(`scenario_ids()`; design §E). It is the only copy. EdgeBasedModels and NodeBasedModels compare their deterministic
curves with these files in tests and vignettes. They read them through NetworkOutbreaks, which is a test-only
dependency for them:

```julia
using NetworkOutbreaks
ref = scenario_summary(:sir_pois5)          # EnsembleSummary: N = 10⁴, 200 runs, fresh graph per run
compare(ref, model_curves(sys, sol; t = ref.t, label = "edge-based"))
```

## Files

Each scenario has three files named `<id>__<hash8>`, where `hash8` is the first 8 hex digits of
`scenario_hash(scenario(id))`. They are written by NetworkEpiCore `save_summary` and read by `load_summary`, using only
the standard library.

| file | content |
|---|---|
| `<id>__<hash8>.toml` | `summary_format`, `id`, the full 64-hex `scenario_hash`, `algorithm_revision` (NetworkOutbreaks `ALGORITHM_REVISION`), `N`, `nsims`, `n_major`, `p_major`, `p_major_ci` (95% Wilson), `observables`, `statistics`, `uncond_equals_cond`, `[provenance]`, `[extras]` |
| `<id>__<hash8>.curves.csv` | `t` (the scenario's `tgrid`), then `cond.<obs>.<stat>` and `uncond.<obs>.<stat>` for every observable and each of `mean, sd, se, q025, q25, q50, q75, q975`. The `uncond` columns are omitted when every run is kept, i.e. when they equal the `cond` ones. |
| `<id>__<hash8>.runs.csv` | per run: `run`, `final_size`, `peak_t`, `peak_value`, `major` (0/1), `shift` (aligned scenarios only) and `realised.<stat>` of the run's graph (`mean_degree`, `excess_degree`, `clustering`, `erased_fraction`, where they apply) |

Observables are population fractions: the species of the model, `:infectious` (the sum of the infectors) and
`:cumulative`. `:cumulative` counts the seeds plus the entries into an infected compartment. For SIR-type models it is
the fraction ever infected and ends at `final_size`; for SIS and SIRS it is cumulative incidence.

## How they are made

`regenerate_scenarios` runs `summarise(scenario_ensemble(sc))` and then `save_summary`
(`src/validation/*.jl`). The rules are as follows.

- **Runs.** There are `sc.sim.nsims` runs on `sc.sim.N` nodes. By default each run gets a fresh graph
  (`graphs = :per_run`), so runs are iid and se = sd/√n. Seeding places exactly ρ_X·N nodes per compartment. On a
  multitype network each stratum is seeded on its own node type, with ρ a fraction of all N nodes.
- **Random streams** (design §J.7). With b = `sc.sim.base_seed`:
  - graph j is `sample_graph(sc.network, N; rng = NetworkOutbreaks.stable_rng(b + j))`;
  - run r uses `stable_rng(b + 2^32 + r)`;
  - run r uses graph j = r (`:per_run`), j = 1 (`:fixed`) or j = mod1(r, G) (`(:pool, G)`).

  These are the streams of `simulate(sc)`. `scenario_graph(sc, r)` returns run r's graph and `scenario_run(sc, r)`
  re-simulates run r alone.
- **Conditioning.** `MajorOutbreak(c)` keeps runs whose new infections by t_end, excluding the seeds, number at least
  c·N. `Survival()` keeps runs with prevalence > 0 at t_end. `[extras.conditioning]` records the quantity the rule
  selects on (`measure`) and its largest value over the discarded runs and smallest over the kept runs. For
  `MajorOutbreak` this is the new-infection fraction, and the two values show whether the threshold falls in the gap
  of the bimodal final-size distribution. For `Survival` it is the prevalence at t_end.
- **Alignment** (the small-seed scenarios, `CumulativeCrossing(ℓ)`):
  - Run i crosses at t_i, the time its cumulative incidence (excluding seeds) first reaches ℓ. It is shifted by
    `shift_i = t_i − t*`, where t* is the grid time nearest the median crossing time of the kept runs
    (`[extras.alignment]`).
  - Samples before a run starts are missing, and so are samples after it ends unless the run is absorbed.
    `cond_n` and `uncond_n` give the number of runs at each grid time.
  - To compare a deterministic curve, shift it to cross ℓ at t* with `aligned_curves(curves, ref)`.
  - Every aligned scenario also has an unaligned companion `<id>_unaligned`
    (`unaligned_scenario(sc)`, same runs), for the aligned-vs-unaligned comparison.
- **SIS/SIRS.** `[extras.reinfection_histogram]` holds the mean fraction of nodes infected p = 0, 1, … times.
- **Precision** (`NetworkOutbreaks.SUMMARY_DIGITS`). Means and standard errors are rounded to 6 decimals; standard
  deviations and quantiles to 5; peak times and shifts to 6; realised graph statistics to 6 significant digits.
  Final sizes and peak values are exact counts/N. Each rounding is far below the Monte Carlo error of its statistic.
- **Provenance** records the Julia, NetworkOutbreaks, NetworkEpiCore, Graphs and StableRNGs versions, the sampler,
  `ALGORITHM_REVISION` and `SUMMARY_REVISION`. It records no date or wall time, so the files are a pure function of
  the scenario and the code, and regeneration is bit-identical. `regenerate_scenarios` prints wall times instead.

## Cache key and loading

The key is (`scenario_hash`, `ALGORITHM_REVISION`), and the `summary_revision` in the provenance must be the current
`NetworkOutbreaks.SUMMARY_REVISION`. A stale summary is never loaded:

- `scenario_summary(sc)` (policy `:committed`) loads from this directory. If the summary is missing, was made for
  another scenario hash, or comes from another algorithm or summary revision, it throws an error.
- `policy = :auto` falls back to a user cache (`scenario_cache_dir()`, or `NETEPI_CACHE_DIR`). If the summary is
  missing there too, it computes the summary. It never writes to this directory.
- With `NETEPI_STRICT_CACHE=1`, which CI sets, `:auto` uses this directory only. It does not read the user cache
  (which a restored depot cache could supply) and it never computes. A missing or stale committed summary is
  therefore an error, even when the user cache holds a valid copy.
- `policy = :recompute` always computes.

## Regenerating

From the `NetworkOutbreaks.jl` directory:

```sh
julia --project=. -t auto scripts/regenerate_scenarios.jl              # every scenario not tagged :deferred
julia --project=. -t auto scripts/regenerate_scenarios.jl sir_pois5 sir_bim
julia --project=. scripts/regenerate_scenarios.jl --list               # ids, hashes, sizes, nothing simulated
julia --project=. scripts/regenerate_scenarios.jl --check              # exit status 1 if a summary is missing or stale
```

Regenerating a scenario removes its files with other hashes. The unaligned companion of a time-aligned scenario is
written from the aligned scenario's ensemble. If the companion is also listed (or registered), it is not simulated a
second time.

Regenerate after any change that alters a scenario's hash (NetworkEpiCore registry), `ALGORITHM_REVISION` (samplers,
generators, streams) or `SUMMARY_REVISION` (summarising). Scenarios tagged `:deferred` have no summary, because
NetworkOutbreaks cannot simulate them yet.

Ownership (design §G.2): the files here belong to WP30; this README, the runner and the script belong to WP27.

## Current summaries against the deterministic limit

Generated by WP30 on 2026-09-27 with `ALGORITHM_REVISION = 2` and `SUMMARY_REVISION = 2` (53 summaries from 51
ensembles; the two `_unaligned` companions share the runs of their aligned scenarios and are not listed separately).
Every file is under 200 kB, and `regenerate_scenarios.jl --check` reports all of them current.

Columns: N and runs are `sc.sim`; P(major) is the conditioned fraction (`MajorOutbreak` or, for SIS/SIRS, `Survival`);
the simulated value is the mean ± SE over the kept runs of the final size R∞ (fraction ever infected, seeds included)
or, for SIS/SIRS, of the prevalence I(t_end). The deterministic value is the first admissible declared back end, in
the order edge-based (EB), then NodeBasedModels pairwise, solved with `reltol = 1e-9` on the scenario's grid, with its
verdict in brackets (aligned scenarios use `aligned_curves`). Δ = deterministic − simulated. D∞(I) is `compare`'s
max_t |I_det − Ī|.

| scenario | N | runs | P(major) | simulated | deterministic [verdict] | Δ | D∞(I) |
|---|---|---|---|---|---|---|---|
| `sir_reg6` | 10000 | 200 | 1.000 | R∞ 0.9292 ± 0.0003 | EB [exact_limit] 0.9295 | +0.0003 | 0.0017 |
| `sir_pois5` | 10000 | 200 | 1.000 | R∞ 0.8001 ± 0.0005 | EB [exact_limit] 0.8002 | +0.0001 | 0.0022 |
| `sir_nb4` | 10000 | 200 | 1.000 | R∞ 0.6405 ± 0.0007 | EB [exact_limit] 0.6408 | +0.0003 | 0.0011 |
| `sir_bim` | 10000 | 200 | 1.000 | R∞ 0.4959 ± 0.0009 | EB [exact_limit] 0.4956 | -0.0003 | 0.0023 |
| `sir_pl` | 10000 | 200 | 1.000 | R∞ 0.2881 ± 0.0012 | EB [exact_limit] 0.2898 | +0.0017 | 0.0026 |
| `sir_wm5` | 10000 | 200 | 1.000 | R∞ 0.8001 ± 0.0006 | EB [exact_limit] 0.7997 | -0.0004 | 0.0014 |
| `seir_pois5` | 10000 | 200 | 1.000 | R∞ 0.8002 ± 0.0005 | EB [exact_limit] 0.8002 | +0.0000 | 0.0003 |
| `sir_erl3_pois5` | 10000 | 200 | 1.000 | R∞ 0.8574 ± 0.0004 | EB [exact_limit] 0.8577 | +0.0003 | 0.0023 |
| `seair_pois5` | 10000 | 200 | 1.000 | R∞ 0.6986 ± 0.0008 | EB [exact_limit] 0.6976 | -0.0010 | 0.0006 |
| `twostrain_pois5` | 10000 | 200 | 1.000 | R∞ 0.8386 ± 0.0005 | EB [exact_limit] 0.8381 | -0.0006 | 0.0013 |
| `sir_vax_pois5` | 10000 | 200 | 1.000 | R∞ 0.5560 ± 0.0012 | EB [exact_limit] 0.5581 | +0.0021 | 0.0019 |
| `sis_reg3` | 10000 | 200 | 1.000 | I(t_end) 0.8171 ± 0.0003 | PW(pairwise_bernoulli) [approximate] 0.8182 | +0.0011 | 0.1456 |
| `sirs_pois5` | 10000 | 200 | 0.990 | I(t_end) 0.0353 ± 0.0006 | PW(pairwise_const) [approximate] 0.0351 | -0.0002 | 0.0047 |
| `sir_clust_s2t2` | 10000 | 200 | 1.000 | R∞ 0.9226 ± 0.0004 | EB [exact_limit] 0.9229 | +0.0003 | 0.0013 |
| `sir_clust_pois12` | 10000 | 200 | 1.000 | R∞ 0.7173 ± 0.0006 | EB [exact_limit] 0.7167 | -0.0006 | 0.0009 |
| `sir_sbm2` | 10000 | 200 | 1.000 | R∞ 0.7738 ± 0.0006 | EB [exact_limit] 0.7749 | +0.0011 | 0.0017 |
| `sir_unstr2` | 10000 | 200 | 1.000 | R∞ 0.9296 ± 0.0003 | EB [exact_limit] 0.9295 | -0.0000 | 0.0009 |
| `sir_mpx` | 10000 | 200 | 1.000 | R∞ 0.8783 ± 0.0004 | EB [exact_limit] 0.8781 | -0.0002 | 0.0008 |
| `sir_ne_reg6_eta01` | 5000 | 100 | 1.000 | R∞ 0.5815 ± 0.0025 | EB [exact_limit] 0.5835 | +0.0020 | 0.0022 |
| `sir_ne_reg6_eta1` | 5000 | 100 | 1.000 | R∞ 0.7472 ± 0.0014 | EB [exact_limit] 0.7469 | -0.0002 | 0.0019 |
| `sir_ne_reg6_eta10` | 5000 | 100 | 1.000 | R∞ 0.7947 ± 0.0014 | EB [exact_limit] 0.7941 | -0.0005 | 0.0019 |
| `sir_dense_pois5` | 10000 | 200 | 1.000 | R∞ 0.5458 ± 0.0013 | EB [exact_limit] 0.5440 | -0.0018 | 0.0008 |
| `sir_dense_pois20` | 10000 | 200 | 1.000 | R∞ 0.7439 ± 0.0007 | EB [exact_limit] 0.7434 | -0.0005 | 0.0008 |
| `sir_dense_pois100` | 10000 | 200 | 1.000 | R∞ 0.7890 ± 0.0007 | EB [exact_limit] 0.7889 | -0.0001 | 0.0010 |
| `sir_pois5_5seeds` | 10000 | 1000 | 0.990 | R∞ 0.7968 ± 0.0003 | EB [exact_limit] 0.7970 | +0.0002 | 0.0011 |
| `sir_pois5_1seed` | 10000 | 2000 | 0.596 | R∞ 0.7966 ± 0.0002 | EB [exact_limit] 0.7968 | +0.0002 | 0.0015 |
| `sir_reg6_fixed` | 1000 | 200 | 1.000 | R∞ 0.9289 ± 0.0009 | EB (annealed, undeclared) [none] 0.9295 | +0.0006 | 0.0072 |
| `sir_erl2_pois5` | 10000 | 200 | 1.000 | R∞ 0.8437 ± 0.0005 | EB [exact_limit] 0.8436 | -0.0001 | 0.0019 |
| `sir_erl5_pois5` | 10000 | 200 | 1.000 | R∞ 0.8690 ± 0.0004 | EB [exact_limit] 0.8688 | -0.0003 | 0.0014 |
| `sir_pois5_N1000` | 1000 | 2000 | 1.000 | R∞ 0.7995 ± 0.0006 | EB [exact_limit] 0.8002 | +0.0007 | 0.0136 |
| `sir_pois5_N100000` | 100000 | 20 | 1.000 | R∞ 0.8000 ± 0.0007 | EB [exact_limit] 0.8002 | +0.0002 | 0.0009 |
| `sir_bim_N1000` | 1000 | 2000 | 0.986 | R∞ 0.4912 ± 0.0009 | EB [exact_limit] 0.4956 | +0.0043 | 0.0177 |
| `sir_bim_N100000` | 100000 | 20 | 1.000 | R∞ 0.4959 ± 0.0009 | EB [exact_limit] 0.4956 | -0.0003 | 0.0009 |
| `sir_pl_N1000` | 1000 | 2000 | 0.856 | R∞ 0.2750 ± 0.0016 | EB [exact_limit] 0.2898 | +0.0148 | 0.0118 |
| `sir_pl_N100000` | 100000 | 20 | 1.000 | R∞ 0.2901 ± 0.0011 | EB [exact_limit] 0.2898 | -0.0003 | 0.0014 |
| `sir_clust_s2t2_N1000` | 1000 | 2000 | 1.000 | R∞ 0.9205 ± 0.0003 | EB [exact_limit] 0.9229 | +0.0024 | 0.0134 |
| `sir_clust_s2t2_N100000` | 100000 | 20 | 1.000 | R∞ 0.9227 ± 0.0004 | EB [exact_limit] 0.9229 | +0.0002 | 0.0012 |
| `sir_dc_bim_r0` | 10000 | 200 | 1.000 | R∞ 0.4952 ± 0.0007 | EB [exact_limit] 0.4956 | +0.0004 | 0.0025 |
| `sir_dc_bim_r05` | 10000 | 200 | 1.000 | R∞ 0.3869 ± 0.0005 | EB [exact_limit] 0.3869 | +0.0001 | 0.0027 |
| `sir_dc_bim_rn05` | 10000 | 200 | 1.000 | R∞ 0.5269 ± 0.0012 | EB [exact_limit] 0.5285 | +0.0016 | 0.0010 |
| `sir_dormant_msv` | 10000 | 200 | 1.000 | R∞ 0.6259 ± 0.0007 | EB [exact_limit] 0.6266 | +0.0007 | 0.0016 |
| `sir_dormant_dvd` | 10000 | 200 | 1.000 | R∞ 0.6435 ± 0.0006 | EB [exact_limit] 0.6440 | +0.0005 | 0.0013 |
| `sir_dormant_fast` | 10000 | 200 | 1.000 | R∞ 0.7560 ± 0.0004 | EB [exact_limit] 0.7564 | +0.0004 | 0.0023 |
| `sir_mfsh_pois5` | 10000 | 200 | 1.000 | R∞ 0.6680 ± 0.0007 | EB [exact_limit] 0.6687 | +0.0007 | 0.0025 |
| `sir_mfsh_msv` | 10000 | 200 | 1.000 | R∞ 0.7846 ± 0.0004 | EB [exact_limit] 0.7846 | +0.0000 | 0.0014 |
| `sir_ne_pois5_eta10` | 5000 | 100 | 1.000 | R∞ 0.6607 ± 0.0017 | EB [exact_limit] 0.6621 | +0.0015 | 0.0023 |
| `seir_clust_s2t2` | 10000 | 200 | 1.000 | R∞ 0.9235 ± 0.0003 | EB [exact_limit] 0.9229 | -0.0006 | 0.0008 |
| `seair_clust_s2t2` | 10000 | 200 | 1.000 | R∞ 0.8327 ± 0.0007 | EB [exact_limit] 0.8330 | +0.0003 | 0.0008 |
| `sir_age2` | 10000 | 200 | 1.000 | R∞ 0.7352 ± 0.0007 | EB [exact_limit] 0.7357 | +0.0005 | 0.0020 |
| `sir_hetsus_bim` | 10000 | 200 | 1.000 | R∞ 0.4226 ± 0.0009 | EB [exact_limit] 0.4239 | +0.0013 | 0.0022 |
| `seirv_hetsus_pois5` | 10000 | 200 | 1.000 | R∞ 0.5948 ± 0.0010 | EB [exact_limit] 0.5956 | +0.0008 | 0.0008 |

Notes.

- Every `:exact_limit` back end at N ≥ 5000 has |Δ| ≤ 3 SE + 10/N and D∞(I) < 0.005 (design §E.2).
- The N = 1000 variants are finite-size checks, not limit checks: D∞(I) is 0.012–0.018 there and falls to about
  0.001 at N = 10⁵. The largest gap is `sir_pl_N1000` (Δ = +0.0148, just over 3 SE + 10/N = 0.0147). It shrinks as
  about 1/N (+0.0148, +0.0017, −0.0003 at N = 10³, 10⁴, 10⁵), so it is a finite-size effect and not a wrong limit.
  Two causes are visible in the files. (1) Erasing multi-edges trims the hubs: 1.1% of stubs are erased at N = 10³,
  and the realised excess degree is 8.19 instead of 8.66. EB on the pooled realised degree distribution of runs 1–100
  gives 0.2848, not 0.2898. (2) With 10 seeds and R∞ ≈ 0.29, the final-size distribution has no gap at the 5%
  threshold (largest discarded 0.049, smallest kept 0.050), so the kept runs include a tail of intermediate outbreaks.
- `sis_reg3` and `sirs_pois5` are compared with approximate pairwise back ends. The endemic prevalence of `sis_reg3` agrees to
  0.001, so its large D∞(I) (0.146) lies in the transient.
- `sir_reg6_fixed` (quenched, one graph with N = 1000) declares only the graph-level `:individual` and `:pair` back
  ends. It is shown against the annealed 6-regular EB limit for reference only.
