# Regenerate the vignette-local reference summaries (owner: WP34, V-NO).
#
#     cd NetworkOutbreaks.jl/vignettes
#     julia --project=. --threads=auto data/generate.jl            # all
#     julia --project=. data/generate.jl --check                     # report missing or stale summaries
#     julia --project=. --threads=auto data/generate.jl ID [ID ...]  # only the named vignette scenarios
#
# The scenarios are those of `_shared/scenarios.jl` (each derived from a registered NetworkEpiCore scenario, so it
# has its own hash); the files are written by NetworkOutbreaks' `regenerate_scenarios`, with the streams of design
# §J.7, into this directory, and the vignettes load them with `scenario_summary(sc; dir = VIGNETTE_DATA)`.
include(joinpath(@__DIR__, "..", "_shared", "scenarios.jl"))

scs = vignette_scenarios()
ids = Symbol.(filter(a -> !startswith(a, "--"), ARGS))
if !isempty(ids)
    unknown = setdiff(ids, [sc.id for sc in scs])
    isempty(unknown) || error("generate.jl: not vignette scenarios: ", join(unknown, ", "))
    filter!(sc -> sc.id in ids, scs)
end
if "--check" in ARGS
    miss = missing_scenario_summaries(scs; dir = VIGNETTE_DATA)
    isempty(miss) ? println("all $(length(scs)) vignette summaries are present and current") :
                    foreach(m -> println(first(m), ": ", last(m)), miss)
    exit(isempty(miss) ? 0 : 1)
end
regenerate_scenarios(scs; dir = VIGNETTE_DATA)
