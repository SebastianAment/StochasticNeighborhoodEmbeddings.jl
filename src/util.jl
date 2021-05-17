# Shannon entropy of discrete distribution with probabilities given by p
entropy(p::Real) = iszero(p) ? p : -p*log2(p)
entropy(p::AbstractVector) = sum(entropy, p)
perplexity(p::AbstractVector) = 2^entropy(p)

# Kullback-Leibler divergence
kl(p::Real, q::Real) = iszero(p) ? zero(p) : p*log(p/q) # what about q = 0?
kl(pq::NTuple{2}) = kl(pq...) # unpack tuple argument
# Z is assumed to be normalization constant of Q
function kl(P::AbstractMatOrFac, Q::AbstractMatOrFac, Z::Real = 1)
    val = zero(promote_type(eltype(P), eltype(Q)))
    for i in 1:size(P, 1), j in 1:size(Q, 2)
        val += kl(P[i, j], Q[i, j] / Z)
    end
    return val
end
# specialization for sparse matrix P
# Z is assumed to be normalization constant of Q
function kl(P::SparseMatrixCSC, Q::AbstractMatOrFac, Z::Real = 1)
    I, J, _ = findnz(P)
    val = zero(promote_type(eltype(P), eltype(Q)))
    for (i, j) in zip(I, J) # TODO: parallelize?
        val += kl(P[i, j], Q[i, j] / Z)
    end
    return val
end

# euclidean distance
_d2(x::Real, y::Real) = (x-y)^2
_d2(x::NTuple{2}) = _d2(x...)
_euclidean(x, y) = sqrt(sum(_d2, zip(x, y)))
euclidean(x::AbstractVector, y::AbstractVector) = _euclidean(x, y) # avoids temporary array
euclidean(x::NTuple, y::NTuple) = _euclidean(x, y) # avoids temporary array

# kernels
gaussian(x::Real) = exp(-x^2/2)
cauchy(x::Real) = inv(1+x^2)

# linear algebra
function hadamard_product(P::AbstractMatrix, Q::AbstractMatOrFac)
    n, m = size(P)
    H = zeros(eltype(P), n, m)
    for j in 1:n, i in 1:m
        H[i, j] = P[i, j] * Q[i, j]
    end
    return H
end

function hadamard_product(P::SparseMatrixCSC, Q::AbstractMatOrFac)
    PQ = similar(P)
    I, J, _ = findnz(P)
     for (i, j) in zip(I, J)
          PQ[i, j] = P[i, j] * Q[i, j]
     end
     return PQ
end


# synthetic data generators
function two_bump_data(d::Int, n::Int)
    X = randn(d, n)
    r = rem(n, 2)
    n = div(n, 2)
    @. X[:, 1:n] = .1X[:, 1:n] + 1
    @. X[:, n+1:end] = .1X[:, n+1:end] - 1
    return X
end

############################# optimization #####################################
function newtons_method(objective, x; tol::Real = 1e-6, max_iter::Int = 32)
    g(x) = ForwardDiff.derivative(objective, x)
    for t in 1:max_iter # Newton's method
        ft, gt = value_derivative(objective, x)
        gt = sign(gt) * maximum(abs, (ft, gt)) # results in abs(ft/gt) ≤ 1 which leads to more robust convergence
        abs(ft) > tol || break
        x -= ft / gt
    end
    return x
end

# auto-diff helper functions to calculate value and gradient with one call
function value_derivative(f, x::Real, diffresults::Val{true} = Val(true))
    r = DiffResults.DiffResult(zero(x), zero(x))
    r = ForwardDiff.derivative!(r, f, x)
    r.value, r.derivs[1]
end
function value_derivative(f, x::Real, diffresults::Val{false})
    f(x), ForwardDiff.derivative(f, x)
end

############################### neighborhoods ##################################
# defaults to euclidean distance
function k_nearest_neighbors(X::AbstractMatrix, k::Int)
    metric = Euclidean()
    println(size(X))
    println(k)
    tree = BallTree(X, metric; leafsize = 32, reorder = true)
    neighbors, distances = knn(tree, X, k + 1) # plus one because we are putting in the original points as well
    for i in 1:size(X, 2) # delete self-references
         filter!(!=(i), neighbors[i])
    end
    return neighbors
end

# ε = sqrt(-2log(δ)) # based on Gaussian kernel, minimum distance for kernel to be smaller than δ
function in_range_neighbors(X::AbstractMatrix, r::Real)
    reorder = true # reorder copies the data and puts nearby points close in memory, think about using Distance.jl for the rest of the code too
    leafsize = 8 # to optimize
    metric = Euclidean()
    tree = BallTree(X, metric; leafsize = 8, reorder = true)
    neighbors = inrange(tree, X, r) # plus one because we are putting in the original points as well
    for i in 1:size(X, 2) # delete self-references
         filter!(!=(i), neighbors[i])
    end
    return neighbors
end
