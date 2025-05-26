using Documenter
using PermutationSymmetricTensors

makedocs(
    sitename = "PermutationSymmetricTensors",
    format = Documenter.HTML(),
    modules = [PermutationSymmetricTensors]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/IlianPihlajamaa/PermutationSymmetricTensors.jl.git"
)