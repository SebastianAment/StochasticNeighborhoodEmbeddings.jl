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
fast_algorithm_min_size = 2^14 # minimum number of data points for fast algorithm

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
include("gradient.jl")

# X is the data matrix whose columns are data points
# u is perplexity
# d is embedding dimensionality, usually 2 or 3
# implementation follows "VISUALIZING DATA USING T-SNE" closely
# missing: adaptive learning rate
function tsne(X::AbstractMatrix, u::Real; max_iter::Int = 1024, d::Int = 2,
            verbose::Bool = false, lr::Real = 1, beta_1::Real = .9, beta_2::Real = .999,
            weight_decay::Real = 0, dofast::Bool = size(X, 2) ≥ fast_algorithm_min_size,
            early_iter::Int = 64, early_lr::Real = lr,
            early_beta_1::Real = 0.5, early_beta_2::Real = 0.9,
            early_compression::Real = 1, early_exaggeration::Real = 12)

    n = size(X, 2)
    Pij = nothing # introduce symbol in scope before if statement
    if dofast
        k = floor(3*u) # k-nearest neighbor cut-off
        neighbors = k_nearest_neighbors(X, k)
        σ² = optimize_variances(X, u, neighbors)
        Pij = conditional_neighbor(X, σ², neighbors, gaussian, euclidean)
    else
        σ² = optimize_variances(X, u)
        Pij = conditional_neighbor(X, σ², gaussian, euclidean)
    end
    P = @. (Pij + Pij') / 2n # symmetrized distribution - technically only super-diagonal matters
    Y = 1e-2randn(d, n) # initialize embedding points
    G = zeros(d, n) # gradient of Y
    G² = zeros(d, n) # second moments of gradient
    if early_iter > 0 && (early_compression > 0 || early_exaggeration > 1)
        initialize_embedding!(Y, G, G², P, early_exaggeration, early_compression,
                        early_lr, early_beta_1, early_beta_2, early_iter, verbose, dofast)
    end
    @. G = 0 # clearing gradient storage
    @. G² = 0
    tsne!(Y, G, G², P; lr = lr, beta_1 = beta_1, beta_2 = beta_2,
    weight_decay = weight_decay, max_iter = max_iter, verbose = verbose, dofast = dofast)
    return Y
end

# it's good practice to do a few iterations with weight decay on Y
# see 3.4 Optimization Methods for t-SNE of "VISUALIZING DATA USING T-SNE"
# called "early compression"
# "early exaggeration" multiplies all Pij by a constant,
# e.g. 4 in "VISUALIZING DATA USING T-SNE", 12 in "Accelerating t-SNE using Tree-Based Algorithms"
function initialize_embedding!(Y::AbstractMatrix, G::AbstractMatrix, G²::AbstractMatrix,
    P::AbstractMatrix, early_exaggeration::Real, early_compression::Real, lr::Real = 1,
    beta_1::Real = .9, beta_2::Real = .999, early_iter::Int = 128, verbose::Bool = false,
    dofast::Bool = size(X, 2) ≥ fast_algorithm_min_size)
    early_exaggeration ≥ 1 || throw(DomainMismatch("early_exaggeration < 1"))
    P .*= early_exaggeration
    tsne!(Y, G, G², P; lr = lr, beta_1 = beta_1, beta_2 = beta_2,
    weight_decay = early_compression, max_iter = early_iter, verbose = verbose, dofast = dofast)
    P ./= early_exaggeration
    return Y
end

function tsne!(Y::AbstractMatrix, G::AbstractMatrix, G²::AbstractMatrix, P::AbstractMatrix;
        max_iter::Int = 1024, lr::Real = 1, beta_1::Real = .9, beta_2::Real = .999,
        weight_decay::Real = 0, verbose::Bool = false, eps::Real = 1e-8, dofast::Bool = size(X, 2) ≥ fast_algorithm_min_size)
    α, β₁, β₂, γ, ε = lr, beta_1, beta_2, weight_decay, eps
    println(max_iter)
    # G² keep track of second moments of gradient
    for t in 1:max_iter
        # NOTE: compared to "Accelerating t-SNE using Tree-Based Algorithms", Q is un-normalized
        Q, Q² = unnormalized_joint_neighbor(Y, cauchy, euclidean, dofast)
        Z = normalization_constant(Q)
        if verbose
            println("iteration ", t)
            println("divergence ", kl(P, Q, Z)) # bottleneck??
        end
        gradient!(G, G², P, Q, Q², Y, Z, β₁, β₂) # gradient + momentum increment
        αₜ = stepsize(α, β₁, β₂, t)
        @. Y = (1 - γ) * Y - αₜ * G / (sqrt(G²) + ε) # increment by objective + weight decay gradients
    end
    return Y
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


# # X is the data matrix whose columns are data points
# # u is perplexity
# # d is embedding dimensionality, usually 2 or 3
# # implementation follows "VISUALIZING DATA USING T-SNE" closely
# # missing: adaptive learning rate
# function tsne(X::AbstractMatrix, u::Real; max_iter::Int = 1024, d::Int = 2,
#             verbose::Bool = false, lr::Real = 1, momentum::Real = 0, weight_decay::Real = 0,
#             early_compression::Real = 1e-2, early_exaggeration::Real = 12, early_iter::Int = 64,
#             early_momentum::Real = 0.5, early_lr::Real = lr,
#             # early_compression::Real = 0, early_exaggeration::Real = 1,
#             dofast::Bool = size(X, 2) ≥ fast_algorithm_min_size)
#
#     n = size(X, 2)
#     Pij = nothing # introduce symbol in scope before if statement
#     if dofast
#         k = floor(3*u) # k-nearest neighbor cut-off
#         neighbors = k_nearest_neighbors(X, k)
#         σ² = optimize_variances(X, u, neighbors)
#         Pij = conditional_neighbor(X, σ², neighbors, gaussian, euclidean)
#     else
#         σ² = optimize_variances(X, u)
#         Pij = conditional_neighbor(X, σ², gaussian, euclidean)
#     end
#     P = @. (Pij + Pij') / 2n # symmetrized distribution - technically only super-diagonal matters
#     Y = 1e-2randn(d, n) # initialize embedding points
#     G = zeros(d, n) # gradient of Y
#     G² = zeros(d, n) # second moments of gradient
#     if early_iter > 0 && (early_compression > 0 || early_exaggeration > 1)
#         initialize_embedding!(Y, G, P, early_exaggeration, early_compression, early_lr, early_momentum, early_iter, verbose)
#     end
#     tsne!(Y, G, P; lr = lr, momentum = momentum, weight_decay = weight_decay,
#                     max_iter = max_iter, verbose = verbose)
#     return Y
# end
#
# # it's good practice to do a few iterations with weight decay on Y
# # see 3.4 Optimization Methods for t-SNE of "VISUALIZING DATA USING T-SNE"
# # called "early compression"
# # "early exaggeration" multiplies all Pij by a constant,
# # e.g. 4 in "VISUALIZING DATA USING T-SNE", 12 in "Accelerating t-SNE using Tree-Based Algorithms"
# function initialize_embedding!(Y::AbstractMatrix, G::AbstractMatrix, P::AbstractMatrix,
#     early_exaggeration::Real, early_compression::Real, lr::Real = 1, momentum::Real = 1/2, early_iter::Int = 128, verbose::Bool = false)
#     early_exaggeration ≥ 1 || throw(DomainMismatch("early_exaggeration < 1"))
#     P .*= early_exaggeration
#     tsne!(Y, G, P; lr = lr, momentum = momentum, weight_decay = early_compression, max_iter = early_iter, verbose = verbose)
#     P ./= early_exaggeration
#     return Y
# end
#
# function tsne!(Y::AbstractMatrix, G::AbstractMatrix, P::AbstractMatrix; max_iter::Int = 1024,
#                lr::Real = 1, momentum::Real = 0, weight_decay::Real = 0, verbose::Bool = false)
#     α, β, γ = lr, momentum, weight_decay
#     # G² keep track of second moments of gradient
#     for i in 1:max_iter
#         # NOTE: compared to "Accelerating t-SNE using Tree-Based Algorithms", Q is un-normalized
#         Q, Q² = unnormalized_joint_neighbor(Y, cauchy, euclidean)
#         Z = normalization_constant(Q)
#         if verbose
#             println("iteration ", i)
#             println("divergence ", kl(P, Q, Z)) # bottleneck??
#         end
#         gradient!(G, P, Q, Q², Y, Z, β) # gradient + momentum increment
#         @. Y = (1 - γ) * Y - α * G # increment by objective + weight decay gradients
#     end
#     return Y
# end
