# Reference ensembles

`data/scenarios/` holds **53 committed summaries**. There is one for every registered NEC
scenario and its N-scaling variants, plus the unaligned companions of the small-seed
scenarios. `missing_scenario_summaries()` is empty. Each summary follows the protocol of design
§E:

- **Size.** N = 10⁴ nodes and 200 runs, with a **fresh graph per run**. The neighbour-exchange
  scenarios use N = 5000 and 100 runs.
- **Streams.** `stable_rng(base_seed + r)` for graph r and `stable_rng(base_seed + 2³² + r)`
  for run r. These are StableRNGs seeded through splitmix64, because adjacent raw seeds are
  correlated.
- **Seeding.** Exactly ρN seeds, placed uniformly.
- **Conditioning.** SIR runs must be major outbreaks (`MajorOutbreak(0.05)`), and SIS runs must
  survive. Both the conditioned and unconditioned statistics are stored, together with
  P(major) and its 95% Wilson interval.
- **Keys.** Summaries are keyed by `scenario_hash(sc)` and `ALGORITHM_REVISION` (currently
  "2"). With `NETEPI_STRICT_CACHE=1`, a missing or stale summary is an error.

`regenerate_scenarios` (or `scripts/regenerate_scenarios.jl`) rebuilds them.

The table shows values read from the committed summaries. "Mean final size" is taken over the
major runs, with its standard error. "Expected R∞" is the NEC value stored in the scenario:
the edge-based fixed point, which is the large-N limit.

| scenario | network | N | runs | P(major) (95% CI) | mean final size ± SE | expected R∞ |
|---|---|---:|---:|---|---|---:|
| `:sir_reg6` | 6-regular | 10⁴ | 200 | 1.000 (0.981, 1.000) | 0.9292 ± 0.0003 | 0.9295 |
| `:sir_pois5` | Poisson(5) | 10⁴ | 200 | 1.000 (0.981, 1.000) | 0.8001 ± 0.0005 | 0.8002 |
| `:sir_nb4` | negative binomial | 10⁴ | 200 | 1.000 (0.981, 1.000) | 0.6405 ± 0.0007 | 0.6408 |
| `:sir_bim` | bimodal | 10⁴ | 200 | 1.000 (0.981, 1.000) | 0.4959 ± 0.0009 | 0.4956 |
| `:sir_pl` | power law | 10⁴ | 200 | 1.000 (0.981, 1.000) | 0.2881 ± 0.0012 | 0.2898 |
| `:seir_pois5` | Poisson(5) | 10⁴ | 200 | 1.000 (0.981, 1.000) | 0.8002 ± 0.0005 | 0.8002 |
| `:sir_wm5` | well mixed (`MassActionSSA`) | 10⁴ | 200 | 1.000 (0.981, 1.000) | 0.8001 ± 0.0006 | 0.8002 |
| `:sir_pois5_1seed` | Poisson(5), one seed | 10⁴ | 2000 | 0.5955 (0.574, 0.617) | 0.7966 ± 0.0002 | 0.7968 |

With a single seed, P(major) is the **infector-side** branching-process probability. It is
0.6094 for this scenario, which is not R∞ = 0.7968. The ensemble value 0.5955 lies below it at
N = 10⁴ (vignette 05, §2).

**N-scaling.** `:sir_pois5`, `:sir_bim`, `:sir_pl` and `:sir_clust_s2t2` are also committed at
N = 10³ (2000 runs) and N = 10⁵ (20 runs). An exact limit shows a D∞ that falls towards the
Monte Carlo floor, while a biased approximation levels off. For example, the edge-based model on
`:sir_pl` has D∞(I) = 0.0118, 0.0027 and 0.0014, while the constant-closure pairwise model has
0.0311, 0.0217 and 0.0200 (vignette 05, §4).

