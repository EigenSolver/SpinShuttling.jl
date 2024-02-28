using Documenter, SpinShuttling

makedocs(
    modules=[SpinShuttling],
    sitename="SpinShuttling.jl",
    checkdocs = :exports,
    format = Documenter.HTML(
        edit_link = nothing,
        canonical = "https://github.com/EigenSolver/SpinShuttling.jl.git",
        assets = [ asset("assets/favicon.png", class=:ico, islocal = true) ]),
    pages=[
        "index.md",
        # "manual.md"
        # "quickstart.md",
        # "installation.md",
        # "examples.md",
        # "stochastics.md",
        # "analytics.md",
        # "integration.md",
        # "reference.md",
        # "about.md"
    ],
    
)

deploydocs(
    repo = "https://github.com/EigenSolver/SpinShuttling.jl.git",
    )