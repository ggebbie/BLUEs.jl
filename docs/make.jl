using BLUEs
using UnitfulLinearAlgebra # incl b.c. non-registered
using Documenter

DocMeta.setdocmeta!(BLUEs, :DocTestSetup, :(using BLUEs); recursive=true)

makedocs(;
    modules=[BLUEs],
    authors="G Jake Gebbie <ggebbie@whoi.edu>",
    repo="https://github.com/ggebbie/BLUEs.jl/blob/{commit}{path}#{line}",
    sitename="BLUEs.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ggebbie.github.io/BLUEs.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ggebbie/BLUEs.jl",
    devbranch="main",
)
