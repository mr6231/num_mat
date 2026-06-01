include("vlozi_algoritmi.jl")

using Plots

# ---------------------------------
# Build example graph
# ---------------------------------

G = krozna_lestev(20)

t = range(0, 2pi, 21)[1:end-1]

x = cos.(t)
y = sin.(t)

fix = 1:20

# ---------------------------------
# Test many ω values
# ---------------------------------

ω_values = 0.1:0.05:1.95

iterations = Int[]

for ω in ω_values

    tocke = hcat(hcat(x, y)', zeros(2, 20))

    k = vlozi!(G, fix, tocke; ω = ω)

    push!(iterations, k)

    println("ω = $(round(ω, digits=2))   iterations = $k")
end

# ---------------------------------
# Find optimal ω
# ---------------------------------

best_idx = argmin(iterations)

best_ω = ω_values[best_idx]

println()
println("Optimal ω = ", best_ω)
println("Minimum iterations = ", iterations[best_idx])

# ---------------------------------
# Plot
# ---------------------------------

p = plot(
    ω_values,
    iterations,
    xlabel = "ω",
    ylabel = "Iterations",
    title = "SOR convergence depending on ω",
    marker = :circle,
    legend = false
)

savefig(p, "./doc/pictures/optimal_omega.png")