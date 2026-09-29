# NetworkOutbreaks test harness.
#
# Every file in test/suites/*.jl is a self-contained suite, included in sorted order, each in its own anonymous
# module (so helper definitions cannot collide) and inside its own testset.
#
# Run a subset by passing name fragments, e.g.
#     julia --project=. -e 'using Pkg; Pkg.test(test_args = ["regressions"])'
# or by setting NETWORKOUTBREAKS_TEST_SUITES="regressions,00_legacy_core" (comma-separated fragments).
# A suite runs if its file name (without .jl) contains any of the fragments.

using Test

const SUITE_DIR = joinpath(@__DIR__, "suites")

function selected_fragments()
    frags = String[a for a in ARGS if !isempty(a)]
    env = get(ENV, "NETWORKOUTBREAKS_TEST_SUITES", "")
    append!(frags, filter(!isempty, strip.(split(env, ','))))
    return frags
end

function suite_files(frags)
    files = sort(filter(f -> endswith(f, ".jl"), readdir(SUITE_DIR)))
    isempty(frags) && return files
    return filter(f -> any(fr -> occursin(fr, first(splitext(f))), frags), files)
end

let frags = selected_fragments(), files = suite_files(frags)
    isempty(files) && error("no test suite matches $(frags); suites: $(readdir(SUITE_DIR))")
    @testset "NetworkOutbreaks" begin
        for f in files
            name = first(splitext(f))
            @testset "$name" begin
                mod = Module(Symbol("Suite_", name))
                Base.include(mod, joinpath(SUITE_DIR, f))
            end
        end
    end
end
