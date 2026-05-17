#=
ensemble.jl

Run many trajectories with different seeds, either sequentially or via
threaded task parallelism.
=#

struct OutbreakEnsemble
    spec::OutbreakSpec
    trajectories::Vector{OutbreakTrajectory}
end

"""
    simulate_ensemble(spec; nsims, seed = rand(UInt64), parallel = false, kwargs...)

Run `nsims` simulations of `spec`. Each child receives a deterministic
sub-seed derived from the master `seed` and its trajectory index, so a fixed
`seed` gives the same ensemble in sequential mode and with `parallel = true`,
independent of Julia's thread count.
"""
function simulate_ensemble(spec::OutbreakSpec;
                           nsims::Integer,
                           seed::Integer = rand(UInt64),
                           algorithm::OutbreakAlgorithm = DirectSSA(),
                           keep::Symbol = :counts,
                           parallel::Bool = false,
                           interventions::InterventionPlan = InterventionPlan())
    nsims >= 1 || throw(ArgumentError("nsims must be ≥ 1"))
    parent_seed = UInt64(seed)
    sub_seeds = [_ensemble_child_seed(parent_seed, k) for k in 1:nsims]
    trajs = Vector{OutbreakTrajectory}(undef, nsims)
    if parallel
        tasks = [Threads.@spawn simulate(spec; algorithm = algorithm,
                                         seed = sub_seeds[k], keep = keep,
                                          interventions = interventions)
                 for k in 1:nsims]
        for k in 1:nsims
            trajs[k] = fetch(tasks[k])
        end
    else
        for k in 1:nsims
            trajs[k] = simulate(spec; algorithm = algorithm,
                                seed = sub_seeds[k], keep = keep,
                                          interventions = interventions)
        end
    end
    return OutbreakEnsemble(spec, trajs)
end

function _ensemble_child_seed(seed::UInt64, i::Integer)
    z = seed + UInt64(i) * 0x9e3779b97f4a7c15
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    return z ⊻ (z >> 31)
end

Base.length(e::OutbreakEnsemble) = length(e.trajectories)
Base.getindex(e::OutbreakEnsemble, i::Integer) = e.trajectories[i]
Base.iterate(e::OutbreakEnsemble, args...) = iterate(e.trajectories, args...)
