using Test
using LinearAlgebra
using domaca01

# Helpers

"""Build a RedkaMatrika directly from a dense Julia Matrix{Float64}."""
function make_sparse(M::Matrix{Float64})
    return toRedka(M)
end

"""Diagonally-dominant n×n matrix that guarantees SOR convergence."""
function dd_matrix(n::Int)
    M = zeros(Float64, n, n)
    for i in 1:n
        for j in 1:n
            if i != j
                M[i, j] = 0.1
            end
        end
        M[i, i] = Float64(n)          # dominant diagonal
    end
    return M
end


@testset "RedkaMatrika" begin

    @testset "Construction and size" begin
        A = toRedka(Matrix{Float64}(I, 3, 3))
        @test size(A) == (3, 3)
        @test A.n == 3
    end

    @testset "toRedka / toDense round-trip" begin
        M = [1.0 0.0 3.0;
             0.0 5.0 0.0;
             2.0 0.0 4.0]
        A = toRedka(M)
        @test toDense(A) ≈ M
    end

    @testset "toRedka skips zeros" begin
        M = [0.0 1.0; 1.0 0.0]
        A = toRedka(M)
        # Only off-diagonal entries should be stored
        @test length(A.V[1]) == 1
        @test length(A.V[2]) == 1
    end

    @testset "getindex – stored entry" begin
        M = [2.0 0.0; 0.0 3.0]
        A = toRedka(M)
        @test A[1, 1] == 2.0
        @test A[2, 2] == 3.0
    end

    @testset "getindex – missing entry returns 0" begin
        M = [1.0 0.0; 0.0 1.0]
        A = toRedka(M)
        @test A[1, 2] == 0.0
        @test A[2, 1] == 0.0
    end

    @testset "setindex! – update existing entry" begin
        M = [1.0 0.0; 0.0 1.0]
        A = toRedka(M)
        A[1, 1] = 9.0
        @test A[1, 1] == 9.0
    end

    @testset "setindex! – insert new entry" begin
        M = [1.0 0.0; 0.0 1.0]
        A = toRedka(M)
        A[1, 2] = 7.0
        @test A[1, 2] == 7.0
        # Original entries untouched
        @test A[1, 1] == 1.0
    end

    @testset "Matrix-vector multiply" begin
        M = [2.0 1.0; 1.0 3.0]
        A = toRedka(M)
        x = [1.0, 2.0]
        @test A * x ≈ M * x
    end

    @testset "Matrix-vector multiply – identity" begin
        A = toRedka(Matrix{Float64}(I, 4, 4))
        x = [1.0, 2.0, 3.0, 4.0]
        @test A * x ≈ x
    end

    @testset "Matrix-vector multiply – zero vector" begin
        M = [3.0 1.0; 2.0 4.0]
        A = toRedka(M)
        @test A * zeros(2) ≈ zeros(2)
    end

    @testset "Scalar multiply (left)" begin
        M = [1.0 2.0; 3.0 4.0]
        A = toRedka(M)
        B = 2.0 * A
        @test toDense(B) ≈ 2.0 .* M
    end

    @testset "Scalar multiply (right)" begin
        M = [1.0 2.0; 3.0 4.0]
        A = toRedka(M)
        B = A * 3.0
        @test toDense(B) ≈ 3.0 .* M
    end

    @testset "Scalar multiply by zero returns empty sparse" begin
        M = [1.0 2.0; 3.0 4.0]
        A = toRedka(M)
        B = 0.0 * A
        @test toDense(B) ≈ zeros(2, 2)
        @test all(isempty.(B.V))
    end

    @testset "Unary negation" begin
        M = [1.0 -2.0; 3.0 4.0]
        A = toRedka(M)
        B = -A
        @test toDense(B) ≈ -M
    end

    @testset "Addition" begin
        M1 = [1.0 0.0; 0.0 2.0]
        M2 = [3.0 1.0; 1.0 4.0]
        A = toRedka(M1)
        B = toRedka(M2)
        C = A + B
        @test toDense(C) ≈ M1 + M2
    end

    @testset "Addition – commutativity" begin
        M1 = [2.0 1.0; 0.0 3.0]
        M2 = [1.0 0.0; 2.0 1.0]
        A = toRedka(M1)
        B = toRedka(M2)
        @test toDense(A + B) ≈ toDense(B + A)
    end

    @testset "Addition – with zero matrix" begin
        M = [1.0 2.0; 3.0 4.0]
        A = toRedka(M)
        Z = toRedka(zeros(Float64, 2, 2))
        @test toDense(A + Z) ≈ M
    end

    @testset "Subtraction" begin
        M1 = [5.0 1.0; 2.0 6.0]
        M2 = [1.0 1.0; 1.0 1.0]
        A = toRedka(M1)
        B = toRedka(M2)
        C = A - B
        @test toDense(C) ≈ M1 - M2
    end

    @testset "Subtraction – self is zero" begin
        M = [3.0 1.0; 2.0 4.0]
        A = toRedka(M)
        @test toDense(A - A) ≈ zeros(2, 2)
    end

    @testset "Addition dimension mismatch throws" begin
        A = toRedka(Matrix{Float64}(I, 2, 2))
        B = toRedka(Matrix{Float64}(I, 3, 3))
        @test_throws AssertionError A + B
    end

    @testset "Large sparse round-trip" begin
        n = 50
        M = zeros(Float64, n, n)
        for i in 1:n
            M[i, i] = Float64(i)
            if i < n; M[i, i+1] = 1.0; end
            if i > 1; M[i, i-1] = -1.0; end
        end
        A = toRedka(M)
        @test toDense(A) ≈ M
    end

end 


@testset "SOR solver" begin

    @testset "1×1 system" begin
        A = toRedka([2.0;;])
        b = [4.0]
        x, iters = sor(A, b)
        @test x ≈ [2.0] atol=1e-8
        @test iters >= 1
    end

    @testset "2×2 diagonally dominant" begin
        M = [4.0 1.0; 1.0 3.0]
        A = toRedka(M)
        b = [1.0, 2.0]
        x, _ = sor(A, b)
        @test x ≈ M \ b atol=1e-8
    end

    @testset "3×3 diagonally dominant" begin
        M = [10.0 -1.0  2.0;
             -1.0 11.0 -1.0;
              2.0 -1.0 10.0]
        A = toRedka(M)
        b = [6.0, 25.0, -11.0]
        x, _ = sor(A, b)
        @test x ≈ M \ b atol=1e-8
    end

    @testset "Identity system" begin
        n = 5
        A = toRedka(Matrix{Float64}(I, n, n))
        b = collect(1.0:n)
        x, iters = sor(A, b)
        @test x ≈ b atol=1e-10
        @test iters == 1        # should converge in one sweep
    end

    @testset "Zero right-hand side gives zero solution" begin
        M = dd_matrix(4)
        A = toRedka(M)
        b = zeros(4)
        x, _ = sor(A, b)
        @test x ≈ zeros(4) atol=1e-10
    end

    @testset "omega=1 equals Gauss-Seidel" begin
        M = dd_matrix(5)
        A = toRedka(M)
        b = ones(5)
        x_sor, _  = sor(A, b, 1.0)
        x_ref = M \ b
        @test x_sor ≈ x_ref atol=1e-8
    end

    @testset "Various omega values converge to same solution" begin
        M = dd_matrix(6)
        A = toRedka(M)
        b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        x_ref = M \ b
        for omega in [0.5, 1.0, 1.2, 1.5]
            x, _ = sor(A, b, omega)
            @test x ≈ x_ref atol=1e-7
        end
    end

    @testset "Tight tolerance is respected" begin
        M = dd_matrix(4)
        A = toRedka(M)
        b = [1.0, -1.0, 2.0, -2.0]
        x, _ = sor(A, b, 1.0, 1e-14)
        residual = toDense(toRedka(M)) * x - b
        @test norm(residual, Inf) < 1e-14
    end

    @testset "Loose tolerance converges faster (fewer iterations)" begin
        M = dd_matrix(10)
        A = toRedka(M)
        b = ones(10)
        _, iters_tight = sor(A, b, 1.0, 1e-12)
        _, iters_loose = sor(A, b, 1.0, 1e-2)
        @test iters_loose <= iters_tight
    end

    @testset "Returns iteration count" begin
        M = dd_matrix(3)
        A = toRedka(M)
        b = [1.0, 2.0, 3.0]
        result = sor(A, b)
        @test length(result) == 2          # (x, iterations)
        @test result[2] isa Int
        @test result[2] >= 1
    end

    @testset "Zero diagonal throws" begin
        # Row 2 has a zero on the diagonal
        M = [2.0 0.0; 1.0 0.0]
        A = toRedka(M)
        b = [1.0, 1.0]
        @test_throws ErrorException sor(A, b)
    end

    @testset "rowTerms – purely diagonal row" begin
        M = [3.0 0.0; 0.0 5.0]
        A = toRedka(M)
        x = [1.0, 2.0]
        diag, off = rowTerms(A, 1, x)
        @test diag == 3.0
        @test off  == 0.0
    end

    @testset "rowTerms – mixed row" begin
        M = [4.0 2.0; 1.0 3.0]
        A = toRedka(M)
        x = [1.0, 1.0]
        diag, off = rowTerms(A, 1, x)
        @test diag == 4.0
        @test off  == 2.0    # 2.0 * x[2] = 2.0
    end

    @testset "Large system" begin
        n = 100
        M = dd_matrix(n)
        A = toRedka(M)
        b = randn(n)
        x, iters = sor(A, b, 1.0, 1e-10)
        residual = M * x - b
        @test norm(residual, Inf) < 1e-8
        @test iters < 10_000
    end

end