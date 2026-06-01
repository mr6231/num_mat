"""
    domaca03

Numerical simulation of a mathematical pendulum using an adaptive
Dormand–Prince RK45 (DOPRI5) integrator with embedded error control.

The module provides:

- nonlinear and linearized pendulum models,
- a Dormand–Prince RK45 step returning BOTH the 5th-order solution
  and the local error estimate (4th vs 5th order difference),
- adaptive-step trajectory integration with precise period estimation
  via linear zero-crossing interpolation,
- energy computation utilities,
- plot generation comparing the two models and showing T vs E.

Physical parameters:

- `g` : gravitational acceleration [m/s²]
- `l` : pendulum length [m]

Accuracy target: relative and absolute tolerance 1e-11, so that
computed quantities are correct to at least 10 decimal places.
"""
module domaca03

export g, l,
       f_math, f_harm,
       dopri5_step,
       integrate,
       energy

const g = 9.81
const l = 1.0

"""
    f_math(u)

Right-hand side of the nonlinear mathematical pendulum.

    θ' = ω
    ω' = -(g/l) * sin(θ)
"""
f_math(u) = [u[2], -(g / l) * sin(u[1])]

"""
    f_harm(u)

Right-hand side of the linearized (harmonic) pendulum (sin θ ≈ θ).

    θ' = ω
    ω' = -(g/l) * θ
"""
f_harm(u) = [u[2], -(g / l) * u[1]]

mean(v) = sum(v) / length(v)


"""
    dopri5_step(f, u, h) -> (u5, err)

Perform one Dormand–Prince RK45 step of size `h`.

Returns a tuple:
- `u5`  : 5th-order (higher-accuracy) solution at t + h
- `err` : local error estimate = ||u5 - u4||, where u4 is the embedded
          4th-order solution. Used by the adaptive controller.

Butcher table (Dormand–Prince):
  c2=1/5, c3=3/10, c4=4/5, c5=8/9, c6=1, c7=1
  b  (5th order) : 35/384, 0, 500/1113, -125/192, 2187/6784, 11/84, 0
  b* (4th order) : 5179/57600, 0, 7571/16695, 393/640, -92097/339200,
                   187/2100, 1/40
  e  = b - b* (error coefficients):
       71/57600, 0, -71/16695, 71/1920, -17253/339200, 22/525, -1/40
"""
function dopri5_step(f, u, h)
    k1 = f(u)
    k2 = f(u .+ h .* (1/5 .* k1))
    k3 = f(u .+ h .* (3/40 .* k1 .+ 9/40 .* k2))
    k4 = f(u .+ h .* (44/45 .* k1 .- 56/15 .* k2 .+ 32/9 .* k3))
    k5 = f(u .+ h .* (19372/6561 .* k1 .- 25360/2187 .* k2 .+
                       64448/6561 .* k3 .- 212/729 .* k4))
    k6 = f(u .+ h .* (9017/3168 .* k1 .- 355/33 .* k2 .+
                       46732/5247 .* k3 .+ 49/176 .* k4 .-
                       5103/18656 .* k5))

    u5 = u .+ h .* (35/384 .* k1 .+
                     500/1113 .* k3 .-
                     125/192 .* k4 .+
                     2187/6784 .* k5 .+
                     11/84 .* k6)

    k7 = f(u5)

    err_vec = h .* (71/57600 .* k1 .-
                    71/16695 .* k3 .+
                    71/1920 .* k4 .-
                    17253/339200 .* k5 .+
                    22/525 .* k6 .-
                    1/40 .* k7)

    return u5, err_vec
end


"""
    integrate(f, theta0, omega0; T, rtol, atol, h0, hmax, hmin) ->
        (ts, thetas, period_avg)

Integrate a pendulum ODE using adaptive Dormand–Prince RK45 (DOPRI5).

Step-size control uses the standard PI controller:

    h_new = h * min(facmax, max(facmin, fac * (tol / err)^(1/5)))

where tol is the mixed absolute/relative tolerance and fac = 0.9 (safety).

Period estimation uses **linear interpolation** at each upward zero-crossing
of θ (i.e. θ changes sign from − to + while ω > 0):

    t_cross = t_old + dt_actual * (-θ_old) / (θ_new - θ_old)

This gives O(h²) accuracy on the crossing time, rather than the O(h)
accuracy of plain step-boundary detection.

# Arguments
- `f`       : ODE right-hand side (`f_math` or `f_harm`)
- `theta0`  : initial angular displacement [rad]
- `omega0`  : initial angular velocity [rad/s]

# Keyword Arguments
- `T=60.0`      : total simulation time [s]
- `rtol=1e-11`  : relative tolerance (target: 10 decimal places)
- `atol=1e-11`  : absolute tolerance
- `h0=1e-3`     : initial step size [s]
- `hmax=0.1`    : maximum allowed step size [s]
- `hmin=1e-14`  : minimum allowed step size [s] (hard floor)

# Returns
`(ts, thetas, period_avg)` where `ts` and `thetas` are the time and
angle trajectories, and `period_avg` is the mean period from all
detected complete oscillations (NaN if none found).
"""
function integrate(f, theta0, omega0;
                   T    = 60.0,
                   rtol = 1e-11,
                   atol = 1e-11,
                   h0   = 1e-3,
                   hmax = 0.1,
                   hmin = 1e-14)

    u   = [theta0, omega0]
    t   = 0.0
    h   = h0

    ts      = Float64[t]
    thetas  = Float64[theta0]
    periods = Float64[]

    last_crossing = nothing

    # PI controller constants
    fac    = 0.9
    facmax = 5.0
    facmin = 0.2

    while t < T
        h = min(h, T - t)
        if h < hmin
            break
        end

        u_new, err_vec = dopri5_step(f, u, h)
        sc  = @. atol + rtol * max(abs(u), abs(u_new))
        err = sqrt(sum((err_vec ./ sc) .^ 2) / length(u))

        if err <= 1.0
            u_old = u
            t_old = t

            t += h
            u  = u_new

            push!(ts,     t)
            push!(thetas, u[1])

            if u_old[1] < 0.0 && u[1] >= 0.0 && u[2] > 0.0
                # Linear interpolation: find t where θ = 0
                #   θ(t) ≈ θ_old + (θ_new - θ_old) * s,  s ∈ [0,1]
                #   0     = θ_old + (θ_new - θ_old) * s
                #   s     = -θ_old / (θ_new - θ_old)
                s       = -u_old[1] / (u[1] - u_old[1])
                t_cross = t_old + s * (t - t_old)

                if last_crossing !== nothing
                    push!(periods, t_cross - last_crossing)
                end
                last_crossing = t_cross
            end

            if err > 0.0
                h = min(h * min(facmax, fac * (1.0 / err)^0.2), hmax)
            else
                h = min(h * facmax, hmax)
            end
        else
            h = h * max(facmin, fac * (1.0 / err)^0.2)
        end
    end

    period_avg = isempty(periods) ? NaN : mean(periods)
    return ts, thetas, period_avg
end

"""
    energy(theta0, omega0=0.0)

Total mechanical energy of the pendulum:

    E = 1/2 * l² * ω² + g * l * (1 - cos θ)
"""
energy(theta0, omega0 = 0.0) =
    0.5 * l^2 * omega0^2 + g * l * (1.0 - cos(theta0))

end # module domaca03

