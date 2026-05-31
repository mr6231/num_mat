using FastGaussQuadrature

const N = 20

const GL_abscissas, GL_weights = gausslegendre(N)

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

function asymptotic_cdf(x::Float64)
    if x >= 0
        return 1.0 - asymptotic_Q(x)
    else
        return asymptotic_Q(-x)
    end
end

function gauss_legendre_cdf(x::Float64)
    s = 0.0

    for (a, w) in zip(GL_abscissas, GL_weights)
        t = 0.5 * x * (a + 1)
        s += w * exp(-t^2 / 2)
    end

    integral = 0.5 * x * s
    return 0.5 + integral / sqrt(2π)
end

#Combined cdf
function cdf_ND(x::Float64)

    if x <= 5.0
        return gauss_legendre_cdf(x)

    else
        return asymptotic_cdf(x)
    end
end
