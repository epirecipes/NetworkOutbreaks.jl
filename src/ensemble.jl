#=
ensemble.jl

Run many trajectories with independent random streams, either sequentially or with threaded task parallelism.
=#

"""
    OutbreakEnsemble

The result of `simulate_ensemble`: the `spec` and a vector of `OutbreakTrajectory`s. It is indexable and iterable
over its trajectories.
"""
struct OutbreakEnsemble
    spec::OutbreakSpec
    trajectories::Vector{OutbreakTrajectory}
end

"""
    simulate_ensemble(spec; nsims, seed = nothing, rng = nothing, algorithm = DirectSSA(),
                      keep = :counts, parallel = false, interventions = InterventionPlan())

Run `nsims` independent simulations of `spec`.

Trajectory `k` is `simulate(spec; seed = s_k, …)` with a child seed `s_k` derived from the master seed:
- with `seed::Integer`, `s_k = NetworkOutbreaks._ensemble_child_seed(seed, k)` (a splitmix64 mix);
- with `rng::AbstractRNG`, the child seeds are `nsims` draws `rand(rng, UInt64)` (the generator is advanced);
- with neither, a master seed is drawn from the global RNG.

Each child therefore has its own `StableRNG` stream (`traj.seed == s_k`), so a given master seed gives the same
ensemble in sequential mode and with `parallel = true`, independently of the number of threads. Every run works on
its own copy of the transition rates, so interventions in one run never affect another. An error in a run is raised
as itself (e.g. an `ArgumentError`), also with `parallel = true`, after every run has finished.
"""
function simulate_ensemble(spec::OutbreakSpec;
                           nsims::Integer,
                           seed::Union{Nothing, Integer} = nothing,
                           rng::Union{Nothing, AbstractRNG} = nothing,
                           algorithm::OutbreakAlgorithm = DirectSSA(),
                           keep::Symbol = :counts,
                           parallel::Bool = false,
                           interventions::InterventionPlan = InterventionPlan())
    nsims >= 1 || throw(ArgumentError("nsims must be ≥ 1"))
    sub_seeds = _ensemble_seeds(seed, rng, nsims)
    one_run(k) = simulate(spec; algorithm = algorithm, seed = sub_seeds[k], keep = keep,
                          interventions = interventions)
    return OutbreakEnsemble(spec, _run_all(one_run, Int(nsims), parallel))
end

function _ensemble_seeds(seed, rng, nsims::Integer)
    if rng !== nothing
        seed === nothing ||
            throw(ArgumentError("pass either `seed` or `rng` to simulate_ensemble, not both"))
        return UInt64[rand(rng, UInt64) for _ in 1:nsims]
    end
    parent = seed === nothing ? rand(UInt64) : _seed_u64(seed)
    return UInt64[_ensemble_child_seed(parent, k) for k in 1:nsims]
end

# splitmix64 of seed + i·φ (φ = 2^64 / golden ratio): well-separated child seeds for any master seed.
_ensemble_child_seed(seed::UInt64, i::Integer) = _splitmix64_mix(seed + UInt64(i) * _GOLDEN64)

Base.length(e::OutbreakEnsemble) = length(e.trajectories)
Base.getindex(e::OutbreakEnsemble, i::Integer) = e.trajectories[i]
Base.iterate(e::OutbreakEnsemble, args...) = iterate(e.trajectories, args...)
