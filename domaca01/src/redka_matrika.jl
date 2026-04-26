
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


function Base.:-(A::RedkaMatrika)
    newV = [(-).(row) for row in A.V]
    newI = [copy(row) for row in A.I]
    return RedkaMatrika(newV, newI, A.n)
end

function getRedkaFromMatrix(M::Matrix{Float64})
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