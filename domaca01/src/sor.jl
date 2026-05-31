"""
    sor(A, b; omega=1.0, tol=1e-10)

Solve the linear system `A * x = b` using the Successive Over-Relaxation (SOR) method.

# Arguments
- `A::RedkaMatrika`: Sparse matrix in row-wise format.
- `b::Vector{Float64}`: Right-hand side vector.
- `omega::Float64=1.0`: Relaxation parameter (ω = 1 corresponds to Gauss–Seidel).
- `tol::Float64=1e-10`: Convergence tolerance based on the infinity norm of the residual.

# Returns
- `(x, iterations)`:
  - `x::Vector{Float64}`: Approximate solution.
  - `iterations::Int`: Number of iterations used.

# Notes
- Maximum iterations is fixed at 10,000.
- Convergence is checked using `‖A*x - b‖∞ < tol`.
- Requires that each row contains a nonzero diagonal element.
"""
function sor(A::RedkaMatrika, b::Vector{Float64}, omega::Float64=1.0, tol=1e-10)
    max_iterations = 10_000
    n, _ = size(A)
    x = zeros(n)

    for iteration in 1:max_iterations
        for row in 1:n
            diagonal_entry, row_sum = rowTerms(A, row, x)

            if diagonal_entry == 0.0
                error("Diagonal element is zero in row $row")
            end

            x[row] = (1 - omega) * x[row] +
                     (omega / diagonal_entry) * (b[row] - row_sum)
        end

        residual = A * x - b
        if norm(residual, Inf) < tol
            return x, iteration
        end
    end

    error("Convergence was not achieved within $max_iterations iterations")
end

"""
    rowTerms(A, row, x)

Compute the diagonal entry and the off-diagonal contribution of a given row
for use in iterative solvers such as SOR.

# Arguments
- `A::RedkaMatrika`: Sparse matrix in row-wise format.
- `row::Int`: Row index to process.
- `x::Vector{Float64}`: Current solution estimate.

# Returns
- `(diagonal, off_sum)`:
  - `diagonal::Float64`: Value of `A[row, row]` (0.0 if absent).
  - `off_sum::Float64`: Sum of all off-diagonal contributions
    `∑ A[row, j] * x[j]` for `j ≠ row`.
"""
function rowTerms(A::RedkaMatrika, row::Int, x::Vector{Float64})
    diagonal = 0.0
    off_sum = 0.0

    for (value, col) in zip(A.V[row], A.I[row])
        if col == row
            diagonal = value
        else
            off_sum += value * x[col]
        end
    end

    return diagonal, off_sum
end