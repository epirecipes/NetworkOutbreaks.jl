# NetworkOutbreaks.jl vignettes


- [Overview](#overview)
- [Pages](#pages)
- [Conventions](#conventions)

## Overview

[NetworkOutbreaks.jl](https://github.com/epirecipes/NetworkOutbreaks.jl)
simulates compartmental epidemic models exactly, as continuous-time
Markov chains, on explicit graphs. You can pass a graph directly or give
a network descriptor from
[NetworkEpiCore.jl](https://github.com/epirecipes/NetworkEpiCore.jl),
which NetworkOutbreaks then samples: configuration, multitype,
clustered, degree-correlated or multiplex networks. It can also simulate
network *processes*, namely neighbour exchange, dormant contacts,
fleeting contacts and well-mixed mass action.

Models are the same `ContactModel`s that
[EdgeBasedModels.jl](https://github.com/epirecipes/EdgeBasedModels.jl)
and
[NodeBasedModels.jl](https://github.com/epirecipes/NodeBasedModels.jl)
lift to deterministic approximations, so one model definition drives all
three packages. NetworkOutbreaks also generates, hashes and commits the
**reference ensembles** (`data/scenarios/`) for NetworkEpiCore’s
canonical scenarios. The vignettes of the other two packages compare
their approximations against these ensembles.

## Pages

1.  [SIS on a 3-regular network: pairwise and reinfection
    counting](01_sis_reinfection/index.md): endemic SIS on scenario
    `:sis_reg3`, compared with the pairwise and reinfection-counting
    approximations, and reinfection histograms.
2.  [Samplers and dynamic networks: neighbour
    exchange](02_algorithms_dynamic_networks/index.md): agreement
    between the exact samplers (next reaction, direct,
    composition–rejection, HAS), and the degree-preserving
    neighbour-exchange process, interpolating between a static network
    and well-mixed mass action.
3.  [Interventions: continuous vaccination and vaccination
    pulses](03_interventions/index.md): S → V vaccination, which
    edge-based models admit as an exit, and scheduled pulses restricted
    to susceptibles.
4.  [Neighbour-triggered quarantine](04_contact_tracing/index.md):
    quarantine triggered by infected neighbours, a contact whose product
    is not infectious.
5.  [The validation protocol: scenarios, conditioning, alignment,
    N-scaling and samplers](05_validation/index.md): how the reference
    ensembles are built, hashed, conditioned on major outbreaks,
    time-aligned and cached, and the N-scaling test that separates
    approximations that are exact in the limit from structural bias.
6.  [Contact processes: dormant contacts, fleeting contacts and mass
    action](06_processes/index.md): the dormant-contact and
    fleeting-contact (MFSH) processes, and `MassActionSSA` checked
    against the complete graph.

## Conventions

Every comparison uses the same seeding, conditioning and band
definitions:

- a fresh graph for each run, drawn from `stable_rng` streams;
- conditioning on major outbreaks for SIR-type models, and on survival
  for SIS and SIRS;
- the spread band is the pointwise 2.5–97.5% quantile of individual
  runs, and the mean band is ±1.96 standard errors.

The mathematics relating stochastic simulation, edge-based models,
pairwise models and mass action (morphisms, limits and calibrations) is
described in the NetworkEpiCore.jl documentation.
