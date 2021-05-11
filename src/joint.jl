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
    return dot(x, Q*y) # implicitly calls FKT through mul!
end

function unnormalized_joint_neighbor!(K::AbstractMatrix, X::AbstractMatrix,
                         kernel = cauchy, distance = euclidean)
    x = [c for c in eachcol(X)]
    xt = permutedims(x)
    @. K = kernel(distance(x, xt)) # lazy?
    for i in diagind(K) # "Because we are only interested in modeling pairwise similarities, we set the value of pi|i to zero."
        K[i] = 0
    end
end

# lazy representation
function unnormalized_joint_neighbor(X::AbstractMatrix, kernel = cauchy, distance = euclidean)
    d, n = size(X)
    # k(x, y) = kernel(distance(x, y))
    # k = Cauchy()
    k(x, y) = cauchy(distance(x, y))
    k²(x, y) = k(x, y)^2
    Q = gramian(k, X)
    Q² = gramian(k², X) # squared kernel
    D = -1I(n) # to subtract diagonal
    if n > fast_algorithm_min_size # factorize with fast kernel transform, if data is large enough to warrant it
        # IDEA: reuse tree replace kernel function: k, k^2
        Q = fkt(Q; max_dofs_per_leaf = 256, precond_param = 2*256, trunc_param = 5) # fast kernel transform
        Q = LazyMatrixSum(Q, D)
        Q² = fkt(Q²; max_dofs_per_leaf = 256, precond_param = 2*256, trunc_param = 5)
        Q² = LazyMatrixSum(Q², D)
    else # dense
        Q = Matrix(Q) + D
        Q² = Matrix(Q²) + D
    end
    return Q, Q²
end

# TODO: have function that changes the kernel function of a FMM, while maintaining
# the same domain decomposition
# function change_kernel(F::MultipoleFactorization, k)
#     to = TimerOutput()
#     G = MultipoleFactorization(k, F.trunc_param, to, multi_to_singnle)
#
#     if tgt_points === src_points && precond_param > 0
#         @timeit fact.to "Get diag inv for precond" compute_preconditioner!(fact, precond_param, variance)
#     end
#     return G
# end
