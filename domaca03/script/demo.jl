# ===========================================================================
# Script entry point — analysis and plots
# Uses the module defined above.
# ===========================================================================
include("../src/domaca03.jl")
import .domaca03
using Plots
using Printf

# ---------------------------------------------------------------------------
# Analytical harmonic period (independent of amplitude)
# ---------------------------------------------------------------------------
T_harm = 2π * sqrt(domaca03.l / domaca03.g)

println("="^65)
println("Matematično nihalo — DOPRI5, natančnost 1e-11 (rtol = atol)")
println("="^65)
@printf "Harmonični nihajni čas  T₀ = %.10f s\n" T_harm
println()

# ---------------------------------------------------------------------------
# Plot 1: small angle (0.2 rad ≈ 11.5°) — curves nearly identical
# ---------------------------------------------------------------------------
println("Računam Graf 1 (θ₀ = 0.2 rad)...")

θ_small = 0.2
ts_m1, θs_m1, Tp_m1 = domaca03.integrate(domaca03.f_math, θ_small, 0.0; T=20.0)
ts_h1, θs_h1, Tp_h1 = domaca03.integrate(domaca03.f_harm, θ_small, 0.0; T=20.0)

@printf "  T_mat = %.10f s,  T_harm = %.10f s,  Δ = %.2e s\n" Tp_m1 Tp_h1 abs(Tp_m1 - Tp_h1)

p1 = plot(ts_m1, θs_m1,
    label  = "Matematično nihalo",
    color  = :darkorange,
    lw     = 2,
    xlabel = "Čas t [s]",
    ylabel = "Odmik θ [rad]",
    title  = "Primerjava nihanja (θ₀ = 0.2 rad ≈ 11.5°)",
    legend = :topright,
    dpi    = 150)
plot!(p1, ts_h1, θs_h1,
    label = "Harmonično nihalo",
    color = :steelblue,
    lw    = 2,
    ls    = :dash)

savefig(p1, "plot1_primerjava_mali_odmik.png")
println("  Shranjeno: plot1_primerjava_mali_odmik.png")

# ---------------------------------------------------------------------------
# Plot 2: large angle (2.5 rad ≈ 143°) — phase drift clearly visible
# ---------------------------------------------------------------------------
println("Računam Graf 2 (θ₀ = 2.5 rad)...")

θ_large = 2.5
ts_m2, θs_m2, Tp_m2 = domaca03.integrate(domaca03.f_math, θ_large, 0.0; T=20.0)
ts_h2, θs_h2, Tp_h2 = domaca03.integrate(domaca03.f_harm, θ_large, 0.0; T=20.0)

@printf "  T_mat = %.10f s,  T_harm = %.10f s,  Δ = %.4f s\n" Tp_m2 Tp_h2 abs(Tp_m2 - Tp_h2)

p2 = plot(ts_m2, θs_m2,
    label  = "Matematično nihalo",
    color  = :darkorange,
    lw     = 2,
    xlabel = "Čas t [s]",
    ylabel = "Odmik θ [rad]",
    title  = "Primerjava nihanja (θ₀ = 2.5 rad ≈ 143°)",
    legend = :topright,
    dpi    = 150)
plot!(p2, ts_h2, θs_h2,
    label = "Harmonično nihalo",
    color = :steelblue,
    lw    = 2,
    ls    = :dash)

savefig(p2, "plot2_primerjava_veliki_odmik.png")
println("  Shranjeno: plot2_primerjava_veliki_odmik.png")

# ---------------------------------------------------------------------------
# Plot 3: Phase portrait (θ, ω) for several initial angles
# ---------------------------------------------------------------------------
println("Računam Graf 3: fazni portret...")

p3 = plot(
    title  = "Fazni portret matematičnega nihala",
    xlabel = "Odmik θ [rad]",
    ylabel = "Kotna hitrost ω [rad/s]",
    legend = :topright,
    dpi    = 150)

angles_phase = [0.5, 1.0, 1.5, 2.0, 2.5, 2.9]
colors_phase = [:steelblue, :teal, :seagreen, :goldenrod, :darkorange, :crimson]

for (θ0, col) in zip(angles_phase, colors_phase)
    ts_p, θs_p, _ = domaca03.integrate(domaca03.f_math, θ0, 0.0; T=80.0)
    # Reconstruct ω from the adaptive trajectory is not stored — re-run
    # keeping both components. We use a thin wrapper to capture ω too.
    # (integrate only stores θ for memory efficiency, so we collect both here.)
    u = [θ0, 0.0]
    thetas_pp = Float64[θ0]
    omegas_pp = Float64[0.0]
    h = 1e-3
    t = 0.0
    T_run = 80.0
    rtol = 1e-11; atol = 1e-11
    hmax = 0.1; hmin = 1e-14
    fac = 0.9; facmax = 5.0; facmin = 0.2
    while t < T_run
        h = min(h, T_run - t)
        h < hmin && break
        u_new, err_vec = domaca03.dopri5_step(domaca03.f_math, u, h)
        sc  = @. atol + rtol * max(abs(u), abs(u_new))
        err = sqrt(sum((err_vec ./ sc) .^ 2) / length(u))
        if err <= 1.0
            t += h; u = u_new
            push!(thetas_pp, u[1]); push!(omegas_pp, u[2])
            h = min(h * min(facmax, fac * (1.0 / max(err, 1e-15))^0.2), hmax)
        else
            h = h * max(facmin, fac * (1.0 / err)^0.2)
        end
    end
    plot!(p3, thetas_pp, omegas_pp,
          label = "θ₀ = $(θ0) rad",
          color = col,
          lw    = 1.2)
end

savefig(p3, "plot3_fazni_portret.png")
println("  Shranjeno: plot3_fazni_portret.png")

# ---------------------------------------------------------------------------
# Plot 4: Period T vs energy E  (main required result)
# ---------------------------------------------------------------------------
println("Računam Graf 4: nihajni čas v odvisnosti od energije...")

# Sample initial angles from nearly 0 to nearly π
# Near π the period → ∞, so we stop at 2.99 rad
θ0_values = range(0.05, 2.99, length=80)
energies  = Float64[]
periods_m = Float64[]

for θ0 in θ0_values
    E = domaca03.energy(θ0)
    # Simulate long enough to capture at least ~10 full oscillations
    T_approx = 2π * sqrt(domaca03.l / domaca03.g) * (1 + θ0^2 / 16)  # rough estimate
    T_sim    = max(80.0, 15 * T_approx)
    _, _, Tper = domaca03.integrate(domaca03.f_math, θ0, 0.0; T = T_sim)
    if !isnan(Tper) && Tper > 0.0
        push!(energies,  E)
        push!(periods_m, Tper)
    end
end

# Maximum energy when θ₀ → π (pendulum balanced upright)
E_max = domaca03.g * domaca03.l * 2.0   # = 2gl

p4 = plot(energies, periods_m,
    label   = "Matematično nihalo (DOPRI5)",
    color   = :darkorange,
    lw      = 2.5,
    marker  = :circle,
    ms      = 3,
    xlabel  = "Energija E [J]",
    ylabel  = "Nihajni čas T [s]",
    title   = "Nihajni čas matematičnega nihala v odvisnosti od energije",
    legend  = :topleft,
    dpi     = 150)

hline!(p4, [T_harm],
    label = @sprintf("Harmonični T₀ = %.6f s", T_harm),
    color = :steelblue,
    lw    = 2,
    ls    = :dash)

vline!(p4, [E_max],
    label = "E_max (θ₀ → π, T → ∞)",
    color = :crimson,
    lw    = 1.5,
    ls    = :dot)

savefig(p4, "plot4_nihajni_cas_energija.png")
println("  Shranjeno: plot4_nihajni_cas_energija.png")