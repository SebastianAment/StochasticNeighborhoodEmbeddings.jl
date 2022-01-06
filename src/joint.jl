# joint neighbor probability for symmetric SNE
# function joint_neighbor(x::VecOfVec{<:Number}, kernel = cauchy, distance = euclidean)
function joint_neighbor(X::AbstractMatrix, kernel = cauchy, distance = euclidean)
    n = size(X, 2)
    joint_neighbor!(zeros(n, n), X, kernel, distance)
end
function joint_neighbor!(K::AbstractMatrix, X::AbstractMatrix,
                         kernel = cauchy, distance = euclidean)
    unnormalized_joint_neighbor!(K, X, kernel, distance)
    K ./= sum(K) # entire matrix is normalized probability distribution now
end

# sum over all elements of Q via matrix-vector products to work with fast transforms
function normalization_constant(Q::AbstractMatOrFac)
    n, m = size(Q)
    x, y = ones(n), ones(m)
    return dot(x, Q*y)
end

function unnormalized_joint_neighbor!(K::AbstractMatrix, X::AbstractMatrix,
                         kernel = cauchy, distance = euclidean)
    x = [c for c in eachcol(X)]
    xt = permutedims(x)
    @. K = kernel(distance(x, xt)) # IDEA: lazy: LazyMatrixSum(gramian(kernel, x), kernel(0)*I(length(x)))
    for i in diagind(K) # "Because we are only interested in modeling pairwise similarities, we set the value of pi|i to zero."
        K[i] = 0
    end
end

# lazy representation
function unnormalized_joint_neighbor(X::AbstractMatrix, kernel = cauchy, distance = euclidean, dofast::Bool = false)
    d, n = size(X)
    k = Cauchy()
    k² = k^2
    Q = gramian(k, X)
    Q² = gramian(k², X) # squared kernel
    D = -1I(n) # to subtract diagonal
    Q = Matrix(Q) + D # IDEA: Lazy
    Q² = Matrix(Q²) + D
    return Q, Q²
end
