using Documenter, SpinShuttling

makedocs(
    modules=[SpinShuttling],
    sitename="SpinShuttling.jl",
    checkdocs = :exports,
    format = Documenter.HTML(
        edit_link = nothing,
        canonical = "https://github.com/EigenSolver/SpinShuttling.jl.git",
        assets = [ asset("assets/logo.svg", class=:ico, islocal = true) ]),
    pages=[
        "Home"=>"index.md",
        # "Tutorial"=>Any[
            # "Quick Start"=>"guide.md",
        # ],
        "Quick Start"=>"guide.md",
        "Manual"=>"manual.md"
    ],
)

deploydocs(
    repo = "https://github.com/EigenSolver/SpinShuttling.jl.git",
    )