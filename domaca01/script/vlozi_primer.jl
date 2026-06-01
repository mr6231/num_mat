include("vlozi_algoritmi.jl")

using GraphRecipes
using Plots

# -------------------------
# Example 1
# -------------------------

G = krozna_lestev(8)

t = range(0, 2pi, 9)[1:end-1]

x = cos.(t)
y = sin.(t)

tocke = hcat(hcat(x, y)', zeros(2, 8))

fix = 1:8

p = graphplot(G)
savefig(p, "./doc/pictures/krozna_lestev.png")

vlozi!(G, fix, tocke)

#println("Coordinates of all vertices:")
#println(tocke)

p = graphplot(
    G,
    x = tocke[1, :],
    y = tocke[2, :],
    curves = false,
    markersize = 0.21
)

savefig(p, "./doc/pictures/vlozena_krozna_lestev.png")

# -------------------------
# Example 2
# -------------------------

m, n = 6, 6

G = Graphs.grid((m, n), periodic=false)

vogali = filter(v -> degree(G, v) <= 2, vertices(G))

tocke = zeros(2, n * m)

tocke[:, vogali] = [0 0 1 1;
                    0 1 0 1]

vlozi!(G, vogali, tocke)

p = graphplot(
    G,
    x = tocke[1, :],
    y = tocke[2, :],
    curves = false
)

savefig(p, "./doc/pictures/vlozena_2d_mreza.png")