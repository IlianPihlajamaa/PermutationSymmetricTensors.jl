using Documenter
using PermutationSymmetricTensors

makedocs(
    sitename = "PermutationSymmetricTensors.jl",
    modules = [PermutationSymmetricTensors],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    pages=[
        "Home" => "index.md",
    ],
    checkdocs = :exported,
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/IlianPihlajamaa/PermutationSymmetricTensors.jl.git"
)
