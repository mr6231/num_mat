struct RedkaMatrika
    V::Vector{Vector{Float64}}
    I::Vector{Vector{Int}}
    n::Int
end

Base.size(A::RedkaMatrika) = (A.n, A.n)

function Base.getindex(A::RedkaMatrika, i::Int, j::Int)
    for (k, col) in enumerate(A.I[i])
        if col == j
            return A.V[i][k]
        end
    end
    return 0.0
end

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

function Base.:*(A::RedkaMatrika, x::Vector{Float64})
    y = zeros(n)
    for i in 1:n
        for (val, j) in zip(A.V[i], A.I[i])
            y[i] = y[i] + val * x[j]
        end
    end
    return y
end

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

Base.:*(A::RedkaMatrika, c::Real) = c * A

function Base.:-(A::RedkaMatrika)
    newV = [(-).(row) for row in A.V]
    newI = [copy(row) for row in A.I]
    return RedkaMatrika(newV, newI, A.n)
end

function Base.:+(A::RedkaMatrika, B::RedkaMatrika)
    @assert A.n == B.n "Matrix dimensions must match"
    n = A.n
    newV = [Float64[] for _ in 1:n]
    newI = [Int[]     for _ in 1:n]
    C = RedkaMatrika(newV, newI, n)

    for i in 1:n
        # Copy all entries from A
        for (k, j) in enumerate(A.I[i])
            C[i, j] = A.V[i][k]
        end
        # Add entries from B (getindex returns 0 for missing entries)
        for (k, j) in enumerate(B.I[i])
            C[i, j] = C[i, j] + B.V[i][k]
        end
    end
    return C
end

Base.:-(A::RedkaMatrika, B::RedkaMatrika) = A + (-B)

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

function toDense(A::RedkaMatrika)
    M = zeros(Float64, A.n, A.n)
    for i in 1:A.n
        for (k, j) in enumerate(A.I[i])
            M[i, j] = A.V[i][k]
        end
    end
    return M
end