module StochasticNeighborhoodEmbeddings
using Base.Threads
using LinearAlgebra
using SparseArrays
using Statistics
using FillArrays
using ForwardDiff
export tsne

using LazyLinearAlgebra
using CovarianceFunctions
using CovarianceFunctions: gramian, Cauchy, Gramian
using FastKernelTransform

# IDEA: have repulsive_force, attractive_force for SNE
fast_algorithm_min_size = 0 # minimum number of data points for fast algorithm

using NearestNeighbors

const AbstractVecOfVec{T} = AbstractVector{<:AbstractVector{<:T}}
const AbstractMatOrFac = Union{AbstractMatrix, Factorization}

function conditional_neighbor end
function optimize_variances end
function joint_neighbor end

include("util.jl") # definition of statistical functions and kernels
include("dense.jl") # standard routines based on dense calculations
include("sparse.jl") # accelerated sparse implementations
include("joint.jl")

# X is the data matrix whose columns are data points
# u is perplexity
# d is embedding dimensionality, usually 2 or 3
# implementation follows "VISUALIZING DATA USING T-SNE" closely
# missing: adaptive learning rate
function tsne(X::AbstractMatrix, u::Real; max_iter::Int = 1024, d::Int = 2,
            verbose::Bool = false, lr::Real = 1, momentum::Real = 0, weight_decay::Real = 0,
            # early_compression::Real = 1, early_exaggeration::Real = 12)
            early_compression::Real = 0, early_exaggeration::Real = 1)

    n = size(X, 2)
    Pij = nothing # introduce symbol in scope before if statement
    if n ≥ fast_algorithm_min_size
        k = floor(3*u) # k-nearest neighbor cut-off
        neighbors = k_nearest_neighbors(X, k)
        σ² = optimize_variances(X, u, neighbors)
        Pij = conditional_neighbor(X, σ², neighbors, gaussian, euclidean)
    else
        σ² = optimize_variances(X, u)
        Pij = conditional_neighbor(X, σ², gaussian, euclidean)
    end
    P = @. (Pij + Pij') / 2n # symmetrized distribution - technically only super-diagonal matters
    Y = .1randn(d, n) # initialize embedded points
    G = zeros(d, n) # gradient of Y
    if early_compression > 0 || early_exaggeration > 1
        initialize_embedding!(Y, G, P, early_exaggeration, early_compression, lr, momentum, max_iter)
    end
    tsne!(Y, G, P; lr = lr, momentum = momentum, weight_decay = weight_decay,
                    max_iter = max_iter, verbose = verbose)
    return Y
end

# it's good practice to do a few iterations with weight decay on Y
# see 3.4 Optimization Methods for t-SNE of "VISUALIZING DATA USING T-SNE"
# called "early compression"
# "early exaggeration" multiplies all Pij by a constant,
# e.g. 4 in "VISUALIZING DATA USING T-SNE", 12 in "Accelerating t-SNE using Tree-Based Algorithms"
function initialize_embedding!(Y::AbstractMatrix, G::AbstractMatrix, P::AbstractMatrix,
    early_exaggeration::Real, early_compression::Real, lr::Real = 1, momentum::Real = 1/2, max_iter::Int = 128)
    early_exaggeration ≥ 1 || throw(DomainMismatch("early_exaggeration < 1"))
    P .*= early_exaggeration
    tsne!(Y, G, P; lr = lr, momentum = momentum, weight_decay = early_compression, max_iter = max_iter)
    P ./= early_exaggeration
    return Y
end

function tsne!(Y::AbstractMatrix, G::AbstractMatrix, P::AbstractMatrix; max_iter::Int = 1024,
               lr::Real = 1, momentum::Real = 0, weight_decay::Real = 0, verbose::Bool = false)
    α, β, γ = lr, momentum, weight_decay
    for i in 1:max_iter
        # NOTE: compared to "Accelerating t-SNE using Tree-Based Algorithms", Q is un-normalized
        Q, Q² = unnormalized_joint_neighbor(Y, cauchy, euclidean)
        Z = normalization_constant(Q)
        if verbose
            println("iteration ", i)
            println("divergence ", kl(P, Q, Z))
        end
        gradient!(G, P, Q, Q², Y, Z, β) # gradient + momentum increment
        @. Y = (1 - γ) * Y - α * G # increment by objective + weight decay gradients
    end
    return Y
end

function gradient!(G::AbstractMatrix, P::AbstractMatOrFac, Q::AbstractMatOrFac,
                   Q²::AbstractMatOrFac, Y::AbstractMatrix, Z::Real, β::Real = 0)
    @. G *= β
    y = [c for c in eachcol(Y)]
    attractive_force!(G, P, Q, Y, Z) # for t-SNE
    repulsive_force!(G, P, Q², Y, Z)
    return G
end

# attractive force can be calculated quickly just based on sparse approximation to P
# Z is normalization constant of Q
function attractive_force!(G::AbstractMatrix, P::SparseMatrixCSC, Q::AbstractMatOrFac, Y::AbstractMatrix, Z::Real)
    y = [c for c in eachcol(Y)]
    I, J, _ = findnz(P)
    for (i, j) in zip(I, J)
        @. G[:, i] += 4Z * P[i, j] * Q[i, j] * (y[i] - y[j])
    end
    return G
end

# generic fallback
function attractive_force!(G::AbstractMatrix, P::AbstractMatOrFac, Q::AbstractMatOrFac, Y::AbstractMatrix, Z::Real)
    y = [c for c in eachcol(Y)]
    for i in eachindex(y)
        for j in eachindex(y)
            @. G[:, i] += 4 * P[i, j] * Q[i, j] * (y[i] - y[j])
        end
    end
    # PQ = P .* Q # fast through sparsity in P, TODO: check if special case is necessary
    # sumPQ = sum(PQ, dims = 2)
    # @. G += 4Z * sumPQ' * Y
    # PQY = PQ * Y'
    # @. G -= PQY'
    return G
end

function repulsive_force!(G::AbstractMatrix, P::AbstractMatOrFac, Q²::AbstractMatOrFac, Y::AbstractMatrix, Z::Real)
    n, m = size(P)
    # 1. row-wise sum of Q²
    # IDEA: could pre-allocate
    sumQ² = Q² * ones(m) # FKT: n * log(n) - fast multiply with n vector
    @. G -= 4/Z * sumQ²' * Y

    # 2. for each dimension, run fast transform once -> d * n * log(n) complexity
    # IDEA: could pre-allocate
    QY = Q² * Y' # FKT: d * n * log(n) - fast multiply with n x d matrix
    @. G += 4/Z * QY'
    return G
end

end # module

# IDEA: represent P, Q lazily
# struct JointDistribution{K, X, T}
#     kernel::K
#     x::X
#     normalization::T
# end
# function JointDistribution(kernel, x::VecOfVec{<:Number})
#     K = gramian(kernel, x)
#     K = fkt(K)
#     normalization = sum(K * Ones(length(x)))
# end
