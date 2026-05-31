using Test
using Distributions

include("../src/cdf_of_nd.jl")

ref(x) = cdf(Normal(), x)

const TOL = 5e-11


function rel_err(approx, truth)
    return abs(approx - truth) / max(abs(truth), abs(1 - truth), 1e-300)
end

function run_report(name, f, xs)
    println()
    println("="^100)
    println(name)
    println("="^100)

    println(rpad("x",12),
            rpad("true value",20),
            rpad("computed value",20),
            rpad("abs err",18),
            rpad("rel err",18),
            "status")

    println("-"^100)

    all_passed = true

    for x in xs
        truth  = ref(x)
        approx = f(x)

        abs_err = abs(approx - truth)
        relerr  = rel_err(approx, truth)

        passed = relerr ≤ TOL

        if !passed
            all_passed = false
        end

        println(
            rpad(string(round(x, digits=3)), 6),
            rpad(string(truth), 22),
            rpad(string(approx), 22),
            rpad(string(abs_err), 25),
            rpad(string(relerr), 25),
            passed ? "PASS" : "FAIL"
        )
    end

    println("-"^100)
    println("Overall result: ", all_passed ? "PASS" : "FAIL")

    @test all_passed
end


@testset "Gauss-Legendre CDF" begin
    run_report(
        "Gauss-Legendre CDF",
        gauss_legendre_cdf,
        Float64.(-5.0:0.25:5.0)
    )
end

@testset "Asymptotic CDF" begin
    run_report(
        "Asymptotic CDF",
        asymptotic_cdf,
        vcat(Float64.(-15.0:0.5:-5.0), Float64.(5.0:0.5:15.0))
    )
end