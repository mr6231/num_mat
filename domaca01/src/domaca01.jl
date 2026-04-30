module domaca01

using LinearAlgebra
using Graphs


export RedkaMatrika, toRedka, toDense, sor, rowTerms, sparseMatrix


include("redka_matrika.jl")
include("sor.jl")

end