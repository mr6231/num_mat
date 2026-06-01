using Plots

include("../src/cdf_of_nd.jl")

# Range for plotting
x_vals = range(-6, 8, length=600)

y_vals = Float64[]
colors = String[]

for x in x_vals
    if x <= 5.0 && x >= -5.0
        push!(colors, "GL")
    else
        push!(colors, "ASYM")
    end
    push!(y_vals, cdf_ND(x))
end

# GL region (continuous middle part)
x_gl = [x_vals[i] for i in eachindex(x_vals) if colors[i] == "GL"]
y_gl = [y_vals[i] for i in eachindex(y_vals) if colors[i] == "GL"]

# Asymptotic region split into TWO parts
x_asym_left  = [x_vals[i] for i in eachindex(x_vals) if x_vals[i] < -5]
y_asym_left  = [y_vals[i] for i in eachindex(y_vals) if x_vals[i] < -5]

x_asym_right = [x_vals[i] for i in eachindex(x_vals) if x_vals[i] > 5]
y_asym_right = [y_vals[i] for i in eachindex(y_vals) if x_vals[i] > 5]

# Plot
p = plot(x_gl, y_gl,
    label="Gauss–Legendre",
    lw=2,
    color=:blue)

plot!(p, x_asym_left, y_asym_left,
    label="Asymptotic expansion",
    lw=2,
    color=:red)

plot!(p, x_asym_right, y_asym_right,
    lw=2,
    color=:red,
    label=false)   # avoid duplicate legend entry

xlabel!("x")
ylabel!("Φ(x)")
title!("Normal CDF (Hybrid Method)")

# Save
savefig(p, "doc/cdf_plot.png")