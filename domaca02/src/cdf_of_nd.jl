using FastGaussQuadrature

"""
Number of Gauss–Legendre quadrature points used in numerical integration
and in the asymptotic expansion.
"""
const N = 20

"""
Precomputed Gauss–Legendre quadrature nodes and weights on [-1, 1].

Used for efficient numerical approximation of the normal CDF.
"""
const GL_abscissas, GL_weights = gausslegendre(N)

"""
    asymptotic_Q(x)

Compute the asymptotic approximation of the Gaussian tail probability

    Q(x) = P(Z > x),  Z ~ N(0,1)

for large positive `x`.

# Arguments
- `x::Float64`: Point at which to evaluate (assumes `x > 0`).

# Returns
- Approximation of the tail probability using an asymptotic series expansion.

# Notes
- Uses a truncated series with `N-1` terms.
- Accurate for large `x`, but unstable for small values.
"""
function asymptotic_Q(x::Float64)

    factor = exp(-x^2 / 2) / (x * sqrt(2π))

    series = 1.0
    term = 1.0

    for n in 1:(N-1)
        term *= -(2n - 1) / (x^2)
        series += term
    end

    return factor * series
end

"""
    asymptotic_cdf(x)

Compute the standard normal cumulative distribution function Φ(x)
using an asymptotic expansion for large |x|.

# Arguments
- `x::Float64`: Evaluation point.

# Returns
- Approximation of Φ(x).

# Notes
- Uses symmetry:
  - x ≥ 0: Φ(x) = 1 - Q(x)
  - x < 0 : Φ(x) = Q(-x)
"""
function asymptotic_cdf(x::Float64)
    if x >= 0
        return 1.0 - asymptotic_Q(x)
    else
        return asymptotic_Q(-x)
    end
end

"""
    gauss_legendre_cdf(x)

Compute the standard normal CDF Φ(x) using Gauss–Legendre quadrature.

# Arguments
- `x::Float64`: Upper integration limit.

# Returns
- Approximation of Φ(x) using numerical integration.

# Notes
- Transforms the integral to [-1, 1] for quadrature.
- Most accurate for moderate values of `x`.
"""
function gauss_legendre_cdf(x::Float64)
    s = 0.0

    for (a, w) in zip(GL_abscissas, GL_weights)
        t = 0.5 * x * (a + 1)
        s += w * exp(-t^2 / 2)
    end

    integral = 0.5 * x * s
    return 0.5 + integral / sqrt(2π)
end

"""
    cdf_ND(x)

Compute the standard normal cumulative distribution function Φ(x)
using a hybrid numerical method.

# Arguments
- `x::Float64`: Evaluation point.

# Returns
- Approximation of Φ(x).

# Method
- For `x ≤ 5`: uses Gauss–Legendre quadrature (accurate in central region)
- For `x > 5`: uses asymptotic expansion (stable in tails)

# Notes
This hybrid approach balances accuracy and numerical stability across
the full real line.
"""
function cdf_ND(x::Float64)

    if x <= 5.0
        return gauss_legendre_cdf(x)
    else
        return asymptotic_cdf(x)
    end
end