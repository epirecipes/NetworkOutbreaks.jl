using Documenter
using NetworkOutbreaks

DocMeta.setdocmeta!(NetworkOutbreaks, :DocTestSetup, :(using NetworkOutbreaks);
                    recursive = true)

makedocs(
    sitename = "NetworkOutbreaks.jl",
    modules  = [NetworkOutbreaks],
    authors  = "Simon Frost",
    repo     = "https://github.com/epirecipes/NetworkOutbreaks.jl/blob/{commit}{path}#{line}",
    format   = Documenter.HTML(;
        canonical    = "https://epirecip.es/NetworkOutbreaks.jl/",
        edit_link    = "main",
        assets       = String[],
        prettyurls   = false,
        size_threshold = nothing,
    ),
    pages = [
        "Home"      => "index.md",
        "Vignettes" => "vignettes.md",
        "API"       => "api.md",
    ],
    checkdocs = :none,
    warnonly  = true,
)
