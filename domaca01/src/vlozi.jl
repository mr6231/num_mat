function laplacianMatrix(G::AbstractGraph, sprem)
    n = length(sprem)
    M = zeros(n, n)

    pos = Dict(v => i for (i, v) in enumerate(sprem))

    for (row, vertex) in enumerate(sprem)
        deg = 0

        for neighbor in neighbors(G, vertex)
            deg += 1

            col = get(pos, neighbor, nothing)
            if col !== nothing
                M[row, col] = 1
            end
        end

        M[row, row] = -deg
    end

    return M
end

function rhsVector(G::AbstractGraph, free_vertices, coords)
    free_set = Set(free_vertices)
    b = zeros(length(free_vertices))

    for (i, v) in enumerate(free_vertices)
        for nbr in neighbors(G, v)
            if !(nbr in free_set)
                b[i] -= coords[nbr]
            end
        end
    end

    return b
end

function vlozi!(G::AbstractGraph, fixed, points; ω = 1.0)
    free = setdiff(vertices(G), fixed)
    A = laplacianMatrix(G, free)

    dims = size(points, 1)
    iters = Int[]

    for d in 1:dims
        b = rhsVector(G, free, view(points, d, :))

        x, k = sor(-A, -b, ω)

        push!(iters, k)
        points[d, free] = x
    end

    return maximum(iters)
end

function krožna_lestev(n)
    G = SimpleGraph(2 * n)
    # prvi cikel
    for i = 1:n-1
    add_edge!(G, i, i + 1)
    end
    add_edge!(G, 1, n)
    # drugi cikel
    for i = n+1:2n-1
    add_edge!(G, i, i + 1)
    end
    add_edge!(G, n + 1, 2n)
    # povezave med obema cikloma
    for i = 1:n
    add_edge!(G, i, i + n)
    end
    return G
end
