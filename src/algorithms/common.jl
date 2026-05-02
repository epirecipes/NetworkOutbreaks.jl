#=
algorithms/common.jl

Algorithm dispatch hierarchy.
=#

abstract type OutbreakAlgorithm end

"""
    simulate(spec::OutbreakSpec; algorithm = DirectSSA(),
             seed::Integer = rand(UInt64), keep::Symbol = :counts)
        :: OutbreakTrajectory

Run a single stochastic simulation. `keep` is one of:

- `:counts`    — only times + compartment counts at each event (default).
- `:events`    — additionally store the full event log.

`seed` is forwarded to a fresh `Xoshiro` RNG; pass the same seed to
reproduce a run exactly (single-threaded).
"""
function simulate end
