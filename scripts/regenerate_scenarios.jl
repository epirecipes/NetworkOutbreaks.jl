#!/usr/bin/env julia
# Owner: WP27 (DESIGN_NetworkEpiCore.md §G.2 WP27, §E.4).
#
# Regenerate the committed reference summaries of the NetworkEpiCore scenarios in data/scenarios/ (see its README.md),
# or check that they are all present and current.
#
# Usage, from the NetworkOutbreaks.jl directory:
#
#   julia --project=. -t auto scripts/regenerate_scenarios.jl                  # every scenario not tagged :deferred
#   julia --project=. -t auto scripts/regenerate_scenarios.jl sir_pois5 sir_bim
#   julia --project=. -t auto scripts/regenerate_scenarios.jl --tags=n_scaling
#   julia --project=. scripts/regenerate_scenarios.jl --list                   # ids, hashes, sizes; simulates nothing
#   julia --project=. scripts/regenerate_scenarios.jl --check                  # exit status 1 if any is missing/stale
#
# Options:
#   --dir=PATH           write to (or check) PATH instead of data/scenarios
#   --tags=a,b           only scenarios with all of these tags (combined with the default exclusion of :deferred)
#   --exclude=a,b        also leave out scenarios with any of these tags
#   --include-deferred   do not leave out :deferred scenarios (they fail if NetworkOutbreaks cannot simulate them)
#   --no-companions      do not write the unaligned companions of the time-aligned scenarios
#   --no-prune           keep files of the same ids with other (stale) hashes
#   --serial             run the simulations of each ensemble on one thread (the files are the same)
#
# The files depend only on the scenarios, ALGORITHM_REVISION, SUMMARY_REVISION and the package versions (no date or
# wall time), so running this twice gives bit-identical files; the wall time of each ensemble is printed instead.

if Base.find_package("NetworkOutbreaks") === nothing
    import Pkg
    Pkg.activate(dirname(@__DIR__); io = devnull)
end
using NetworkOutbreaks

function parse_args(args)
    opts = Dict{String, String}()
    ids = Symbol[]
    for a in args
        if startswith(a, "--")
            k, v = occursin('=', a) ? split(a[3:end], '='; limit = 2) : (a[3:end], "")
            k in ("dir", "tags", "exclude", "include-deferred", "no-companions", "no-prune", "serial", "list",
                  "check", "help") || error("regenerate_scenarios.jl: unknown option --$(k) (see the header of this script)")
            opts[k] = v
        else
            push!(ids, Symbol(a))
        end
    end
    return opts, ids
end

symbols(s) = Symbol[Symbol(strip(x)) for x in split(s, ',') if !isempty(strip(x))]

function selected_ids(opts, ids)
    isempty(ids) || return ids
    exclude = haskey(opts, "include-deferred") ? Symbol[] : [:deferred]
    append!(exclude, symbols(get(opts, "exclude", "")))
    return scenario_ids(; tags = symbols(get(opts, "tags", "")), exclude)
end

function main(args)
    opts, given = parse_args(args)
    if haskey(opts, "help")
        println("see the header of ", @__FILE__)
        return 0
    end
    dir = get(opts, "dir", scenario_data_dir())
    companions = !haskey(opts, "no-companions")
    ids = selected_ids(opts, given)
    # the ensembles to simulate and the summaries each is written as (companions and repeated ids once)
    plan = NetworkOutbreaks._regeneration_plan(ids, companions)
    nsummaries = sum(length ∘ last, plan; init = 0)
    if haskey(opts, "list")
        for (_, targets) in plan, sc in targets
            base = summary_basename(sc.id, scenario_hash(sc))
            files = [joinpath(dir, base * ext) for ext in (".toml", ".curves.csv", ".runs.csv")]
            size = all(isfile, files) ? string(round(sum(filesize, files) / 1024; digits = 1), " kB") : "missing"
            println(rpad(string(sc.id), 30), " ", first(scenario_hash(sc), 8), "  N = ", rpad(string(sc.sim.N), 7),
                    " runs ", rpad(string(sc.sim.nsims), 5), " ", size)
        end
        return 0
    end
    if haskey(opts, "check")
        bad = missing_scenario_summaries(ids; dir, companions)
        for (id, why) in bad
            println(rpad(string(id), 30), " ", why)
        end
        println(isempty(bad) ? "all $(nsummaries) summaries ($(length(plan)) ensembles) are current in $(dir)" :
                "$(length(bad)) of $(nsummaries) summaries are missing or stale in $(dir)")
        return isempty(bad) ? 0 : 1
    end
    println("regenerating $(nsummaries) summaries from $(length(plan)) ensembles into $(dir) with ",
            "$(Threads.nthreads()) thread(s); ",
            "ALGORITHM_REVISION = ", ALGORITHM_REVISION, ", SUMMARY_REVISION = ", NetworkOutbreaks.SUMMARY_REVISION)
    total = @elapsed results = regenerate_scenarios(ids; dir, companions, prune = !haskey(opts, "no-prune"),
                                                    parallel = !haskey(opts, "serial") && Threads.nthreads() > 1)
    big = String[]
    for r in results, f in values(r.files)
        filesize(f) > 200 * 1024 && push!(big, "$(basename(f)) ($(round(filesize(f) / 1024; digits = 1)) kB)")
    end
    isempty(big) || println("files over 200 kB: ", join(big, ", "))
    println("wrote $(length(results)) summaries in $(round(total; digits = 1)) s")
    return 0
end

exit(main(ARGS))
