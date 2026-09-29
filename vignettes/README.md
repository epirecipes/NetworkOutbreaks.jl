# NetworkOutbreaks.jl vignettes

| Page | Scenarios | Content |
|---|---|---|
| `01_sis_reinfection` | `:sis_reg3` | SIS on a 3-regular network: pairwise and reinfection-counting pairwise against the committed ensemble; reinfection histograms |
| `02_algorithms_dynamic_networks` | `:sir_pois5`, `:sir_ne_reg6_eta{01,1,10}`, `:sir_wm5`, `:sir_ne_reg6_eta001_N1e5`, `:sir_reg6_tau12_N1e5` | cost per event of the four samplers; neighbour exchange (`NeighbourExchangeProcess`) against the edge-based DFD model; the final-size dip below the static value at small η |
| `03_interventions` | `:sir_vax_pois5`, `:sir_pois5`, `:sirv_pulse0_pois5` | continuous vaccination; vaccination pulses (`ScheduledStateChange(...; from = [:S])`) |
| `04_contact_tracing` | `:sir_pois5`, `:sirq_pois5_a{005,01,02}` | neighbour-triggered quarantine S + I → Q + I against the edge-based model |
| `05_validation` | `:sir_pois5`, `:sir_pois5_1seed`, `:sir_pois5_5seeds`, N-scaling, sampler variants | the scenario protocol, conditioning, alignment, N-scaling, sampler agreement |
| `06_processes` | `:sir_dormant_*`, `:sir_mfsh_*`, `:sir_wm5` | dormant contacts, fleeting contacts (MFSH), `MassActionSSA` against the complete graph |

Render with `quarto render` in this directory (the pages listed in `_quarto.yml`; `freeze: auto`, and
`_freeze/` is kept). Set `NETEPI_STRICT_CACHE=1` so that a missing reference summary is an error.
A multi-format project render can leave a frozen page's `index.md` without its `index_files/`
figures (quarto 1.9 removes them while embedding the html); finish with
`quarto render <page>/index.qmd --to gfm` for each page and check that every
`index_files/figure-commonmark/*.svg` that `index.md` links to exists.

Reference summaries: registered scenarios load the committed files of `../data/scenarios/`; the
vignette-local scenarios of `_shared/scenarios.jl` (each `derive`d from a registered scenario, with
its own hash) load `data/`, which `data/generate.jl` regenerates (all of them, or only the ids given
as arguments).

The PDFs are built with lualatex, for the font fallbacks (STIX Two Math, Menlo) of the glyphs that
STIX Two Text and Fira Code lack; fvextra wraps long code and output lines.
