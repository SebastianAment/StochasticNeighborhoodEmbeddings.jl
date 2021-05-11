# implements sparsity-exploiting versions of the main functions that
# take advantage of neighborhoods in the data

# computes a sparse approximation to the conditional neighbor matrix based on
# neighborhood graph "neighbors" computed with a fast nearsest neighbor algorithm (see util)
function conditional_neighbor(X::AbstractMatrix, σ²::AbstractVector{<:Real},
        neighbors::AbstractVecOfVec{Int}, kernel = gaussian, distance = euclidean)
    x = [c for c in eachcol(X)]
    n = length(x)
    K = spzeros(n, n)
    for i in 1:n
        xn, sn = @views x[neighbors[i]], σ²[neighbors[i]]
        @. K[neighbors[i], i] = kernel(distance(xn, (x[i],)) / sqrt(σ²[i]))
    end
    sumK = sum(K, dims = 1) # this is now a sparse operation!
    # @. K ./= sumK # this yields a dense array!
    for (i, k) in enumerate(eachcol(K))
        k ./= sumK[i]
    end
    return K
end

function optimize_variances(X::AbstractMatrix, u::Real, neighbors::AbstractVecOfVec{Int}, kernel = gaussian, distance = euclidean;
                            tol::Real = 1e-6, max_iter::Int = 32)
    n = size(X, 2)
    σ² = fill(5one(eltype(X)), n) # if this is too small initially, Newton has problems
    @threads for i in 1:n
        σ²[i] = optimize_variances(X, u, i, σ²[i], kernel, distance, tol = tol, max_iter = max_iter)
    end
    any(isnan, σ²) && throw(DomainError("Warning: variance optimization yielded NaN: ", σ²))
    return σ²
end

# optimizing variances using sparse operations
# neighbors_i are the indices of the neighbors of x[i]
function optimize_variances(X::AbstractMatrix, u::Real, i::Int, σ²::Real,
                            neighbors_i::AbstractVector{Int},
                            kernel = gaussian, distance = euclidean;
                            tol::Real = 1e-6, max_iter::Int = 32)
    x = [c for c in eachcol(X)]
    n = length(x)
    xn = @view x[neighbors_i]
    function objective(log_σ²::Real) # allocating version, is faster for high d, equal for low
        local σ² = exp(log_σ²)
        p = @. kernel(distance((x[i],), xn)/σ²) # only evaluate kernel function on neighbors
        p ./= sum(p)
        p = SparseVector(n, neighbors_i, p) # create sparse vectors
        entropy(p) - log2(u)
    end
    log_σ² = newtons_method(objective, log(σ²), tol = tol, max_iter = max_iter)
    return exp(log_σ²)
end
