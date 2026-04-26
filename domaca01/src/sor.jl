function sor(A::RedkaMatrika, b::Vector{Float64}, omega::Float64=1.0, tol=1e-10)
    max_iterations=10_000
    n, _ = size(A)
    x = zeros(n)

    for iteration in 1:max_iterations
        for row in 1:n
            diagonal_entry, row_sum = row_terms(A, row, x)

            if diagonal_entry == 0.0
                error("Diagonal element is zero in row $row")
            end

            x[row] = (1 - omega) * x[row] + (omega / diagonal_entry) * (b[row] - row_sum)
        end

        # Check residual infinity norm
        residual = A * x - b
        if norm(residual, Inf) < tol
            return x, iteration
        end
    end

    error("Convergence was not achieved within $max_iterations iterations")
end

function row_terms(A::RedkaMatrika, row::Int, x::Vector{Float64})
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