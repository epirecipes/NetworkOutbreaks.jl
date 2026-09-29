# API reference

NetworkOutbreaks re-exports every NetworkEpiCore name (`ContactModel`, `sir_model`,
`ConfigurationNetwork`, `SeedFraction`, `scenario`, …). Those names are documented in
NetworkEpiCore. The docstrings below are NetworkOutbreaks' own.

## Models, networks, specs and trajectories

```@autodocs
Modules = [NetworkOutbreaks]
Pages   = ["NetworkOutbreaks.jl", "model.jl", "network.jl", "spec.jl", "state.jl", "events.jl"]
```

## Samplers

```@autodocs
Modules = [NetworkOutbreaks]
Pages   = ["algorithms/common.jl", "algorithms/direct.jl", "algorithms/next_reaction.jl",
           "algorithms/composition_rejection.jl", "algorithms/has.jl", "algorithms/mass_action.jl"]
```

## Graph processes and graph generators

```@autodocs
Modules = [NetworkOutbreaks]
Pages   = ["processes/common.jl", "processes/neighbour_exchange.jl", "processes/dormant.jl",
           "processes/mfsh.jl", "generators/common.jl", "generators/configuration.jl",
           "generators/wellmixed.jl", "generators/multitype.jl", "generators/clustered.jl",
           "generators/multiplex.jl", "generators/joint_degree.jl"]
```

## Interventions

```@autodocs
Modules = [NetworkOutbreaks]
Pages   = ["interventions.jl"]
```

## Ensembles, observables and the high-level `simulate`

```@autodocs
Modules = [NetworkOutbreaks]
Pages   = ["ensemble.jl", "analysis.jl", "convenience.jl"]
```

## Reference ensembles of the shared scenarios

```@autodocs
Modules = [NetworkOutbreaks]
Pages   = ["validation/conditioning.jl", "validation/runner.jl", "validation/summarise.jl",
           "validation/cache.jl"]
```

## Index

```@index
Modules = [NetworkOutbreaks]
```
