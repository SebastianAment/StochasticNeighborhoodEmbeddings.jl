# conditional neighbor probability for generally asymmetric SNE
# generalizes kernel and distance
function conditional_neighbor(X::AbstractMatrix, σ²::AbstractVector{<:Real},
                    kernel = gaussian, distance = euclidean)
    x = [c for c in eachcol(X)]
    xt, vt = permutedims(x), permutedims(σ²)
    K = @. kernel(distance(x, xt) / sqrt(vt)) # lazy?
    for i in diagind(K) # "Because we are only interested in modeling pairwise similarities, we set the value of pi|i to zero."
        K[i] = 0
    end
    sumK = sum(K, dims = 1) # this is an O(n²) operation!
    @. K /= sumK # columns are normalized probability distributions now
end

# adjusts variances of conditional neighbor distribution to achieve perplexity u
function optimize_variances(X::AbstractMatrix, u::Real, kernel = gaussian, distance = euclidean;
                            tol::Real = 1e-6, max_iter::Int = 32)
    n = size(X, 2)
    σ² = fill(10one(eltype(X)), n) # if this is too small initially, Newton has problems
    @threads for i in 1:n
        σ²[i] = optimize_variances(X, u, i, σ²[i], kernel, distance, tol = tol, max_iter = max_iter)
    end
    any(isnan, σ²) && throw(DomainError("Warning: variance optimization yielded NaN: $σ²"))
    return σ²
end

# optimizing variances using dense operations
function optimize_variances(X::AbstractMatrix, u::Real, i::Int, σ²::Real,
                            kernel = gaussian, distance = euclidean;
                            tol::Real = 1e-6, max_iter::Int = 32)
    x = [c for c in eachcol(X)]
    n = length(x)
    function objective(log_σ²::Real) # allocating version, is faster for high d, equal for low
        local σ² = exp(log_σ²)
        p = @. kernel(distance(x[i:i], x) / sqrt(σ²)) # this has sparse approximation (see sparse.jl)
        p[i] = 0
        p ./= sum(p)
        entropy(p) - log2(u)
    end
    log_σ² = newtons_method(objective, log(σ²), tol = tol, max_iter = max_iter)
    return exp(log_σ²)
end
