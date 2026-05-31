"""
    RedkaMatrika

A simple sparse matrix representation using row-wise storage.

# Fields
- `V::Vector{Vector{Float64}}`: Nonzero values stored row by row.
- `I::Vector{Vector{Int}}`: Column indices corresponding to each value in `V`.
- `n::Int`: Matrix dimension (assumed square `n × n`).

Each row `i` stores its nonzero entries as pairs `(V[i][k], I[i][k])`.
"""
struct RedkaMatrika
    V::Vector{Vector{Float64}}
    I::Vector{Vector{Int}}
    n::Int
end

"""
    size(A::RedkaMatrika)

Return the dimensions of the sparse matrix as `(n, n)`.
"""
Base.size(A::RedkaMatrika) = (A.n, A.n)

"""
    getindex(A::RedkaMatrika, i, j)

Return the element `A[i, j]`.

If the element is not explicitly stored, returns `0.0`.
"""
function Base.getindex(A::RedkaMatrika, i::Int, j::Int)
    for (k, col) in enumerate(A.I[i])
        if col == j
            return A.V[i][k]
        end
    end
    return 0.0
end

"""
    setindex!(A::RedkaMatrika, val, i, j)

Set the element `A[i, j] = val`.

If the entry already exists, it is overwritten. Otherwise, it is appended
to the sparse structure.
"""
function Base.setindex!(A::RedkaMatrika, val::Float64, i::Int, j::Int)
    for (k, col) in enumerate(A.I[i])
        if col == j
            A.V[i][k] = val
            return
        end
    end

    push!(A.I[i], j)
    push!(A.V[i], val)
end

"""
    *(A::RedkaMatrika, x::Vector{Float64})

Multiply sparse matrix `A` with dense vector `x`.

Returns the dense result vector `y = A * x`.
"""
function Base.:*(A::RedkaMatrika, x::Vector{Float64})
    y = zeros(A.n)
    for i in 1:A.n
        for (val, j) in zip(A.V[i], A.I[i])
            y[i] = y[i] + val * x[j]
        end
    end
    return y
end

"""
    *(c, A::RedkaMatrika)

Scalar multiplication `c * A`.

Returns a new sparse matrix with all values scaled by `c`.
If `c == 0`, returns an empty sparse matrix.
"""
function Base.:*(c::Real, A::RedkaMatrika)
    if c == 0
        return RedkaMatrika(
            [Float64[] for _ in 1:A.n],
            [Int[]     for _ in 1:A.n],
            A.n
        )
    end
    newV = [A.V[i] .* c for i in 1:A.n]
    newI = [copy(A.I[i]) for i in 1:A.n]
    return RedkaMatrika(newV, newI, A.n)
end

"""
    *(A, c)

Right scalar multiplication `A * c`.
"""
Base.:*(A::RedkaMatrika, c::Real) = c * A

"""
    -(A::RedkaMatrika)

Unary negation of a sparse matrix.

Returns a new matrix where all values are negated.
"""
function Base.:-(A::RedkaMatrika)
    newV = [(-).(row) for row in A.V]
    newI = [copy(row) for row in A.I]
    return RedkaMatrika(newV, newI, A.n)
end

"""
    +(A::RedkaMatrika, B::RedkaMatrika)

Add two sparse matrices of the same size.

Returns a new sparse matrix `C = A + B`.
"""
function Base.:+(A::RedkaMatrika, B::RedkaMatrika)
    @assert A.n == B.n "Matrix dimensions must match"
    n = A.n
    newV = [Float64[] for _ in 1:n]
    newI = [Int[]     for _ in 1:n]
    C = RedkaMatrika(newV, newI, n)

    for i in 1:n
        for (k, j) in enumerate(A.I[i])
            C[i, j] = A.V[i][k]
        end
        for (k, j) in enumerate(B.I[i])
            C[i, j] = C[i, j] + B.V[i][k]
        end
    end
    return C
end

"""
    -(A, B)

Matrix subtraction implemented as `A + (-B)`.
"""
Base.:-(A::RedkaMatrika, B::RedkaMatrika) = A + (-B)

"""
    toRedka(M::Matrix{Float64})

Convert a dense matrix `M` into sparse `RedkaMatrika` format.

Only nonzero entries are stored.
"""
function toRedka(M::Matrix{Float64})
    n = size(M, 1)
    V = Vector{Vector{Float64}}(undef, n)
    I = Vector{Vector{Int}}(undef, n)
    for i in 1:n
        V[i] = Float64[]
        I[i] = Int[]
        for j in 1:n
            if M[i,j] != 0
                push!(V[i], M[i,j])
                push!(I[i], j)
            end
        end
    end
    return RedkaMatrika(V, I, n)
end

"""
    toDense(A::RedkaMatrika)

Convert a sparse `RedkaMatrika` into a dense `Matrix{Float64}`.
"""
function toDense(A::RedkaMatrika)
    M = zeros(Float64, A.n, A.n)
    for i in 1:A.n
        for (k, j) in enumerate(A.I[i])
            M[i, j] = A.V[i][k]
        end
    end
    return M
end