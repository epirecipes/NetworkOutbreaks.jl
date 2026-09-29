# Vignettes

The vignettes are Quarto pages under `vignettes/` in the repository, rendered with a frozen
cache. `vignettes/README.md` lists the scenarios that each page uses.

- [SIS on a 3-regular network: pairwise and reinfection counting](vignettes/01_sis_reinfection/index.html): scenario `:sis_reg3`
- [Samplers and dynamic networks: neighbour exchange](vignettes/02_algorithms_dynamic_networks/index.html): scenarios `:sir_pois5`, `:sir_ne_reg6_eta{01,1,10}`, `:sir_wm5`; vignette-local `:sir_ne_reg6_eta001_N1e5`, `:sir_reg6_tau12_N1e5`
- [Interventions: continuous vaccination and vaccination pulses](vignettes/03_interventions/index.html): scenarios `:sir_vax_pois5`, `:sir_pois5`, `:sirv_pulse0_pois5`
- [Neighbour-triggered quarantine](vignettes/04_contact_tracing/index.html): scenarios `:sir_pois5`, `:sirq_pois5_a{005,01,02}`
- [The validation protocol: scenarios, conditioning, alignment, N-scaling and samplers](vignettes/05_validation/index.html)
- [Contact processes: dormant contacts, fleeting contacts and mass action](vignettes/06_processes/index.html): scenarios `:sir_dormant_*`, `:sir_mfsh_*`, `:sir_wm5`

The pages are also rendered as Markdown (`vignettes/<page>/index.md`) and PDF.
