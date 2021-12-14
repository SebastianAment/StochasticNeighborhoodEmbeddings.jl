module StochasticNeighborhoodEmbeddings
using Base.Threads
using LinearAlgebra
using SparseArrays
using Statistics
using FillArrays
using ForwardDiff
export tsne, tsne!

# using LazyLinearAlgebra
using CovarianceFunctions
using CovarianceFunctions: Cauchy

# # IDEA: have repulsive_force, attractive_force for SNE
# fast_algorithm_min_size = 2^14 # minimum number of data points for fast algorithm

using NearestNeighbors

const AbstractVecOfVec{T} = AbstractVector{<:AbstractVector{<:T}}
const AbstractMatOrFac = Union{AbstractMatrix, Factorization}

function conditional_neighbor end
function optimize_variances end
function joint_neighbor end

include("util.jl") # definition of statistical functions and kernels
include("parameters.jl")
include("dense.jl") # standard routines based on dense calculations
include("sparse.jl") # accelerated sparse implementations
include("joint.jl")
include("gradient.jl")

# X is the data matrix whose columns are data points
# u is perplexity
# d is embedding dimensionality, usually 2 or 3
# implementation follows "VISUALIZING DATA USING T-SNE" closely
# missing: adaptive learning rate
function tsne(X::AbstractMatrix, u::Real, params::OptimizationParameters = OptimizationParameters())

    n = size(X, 2)
    Y = 1e-2randn(params.d, n) # initialize embedding points
    tsne!(Y, X, u, params)
    return Y
end

# if Y is already allocated
function tsne!(Y::AbstractMatrix, X::AbstractMatrix, u::Real, params::OptimizationParameters = OptimizationParameters())
    P = optimized_symmetrized_neighbor(X, u, params.dofast)
    tsne!(Y, P, params)
    return Y
end

function optimized_symmetrized_neighbor(X::AbstractMatrix, u::Real, dofast::Bool)
    n = size(X, 2)
    Pij = nothing # introduce symbol in scope before if statement
    k = floor(3*u) # k-nearest neighbor cut-off for sparsification of P matrix
    if dofast && k < n
        neighbors = k_nearest_neighbors(X, k)
        σ² = optimize_variances(X, u, neighbors)
        Pij = conditional_neighbor(X, σ², neighbors, gaussian, euclidean)
    else
        σ² = optimize_variances(X, u)
        Pij = conditional_neighbor(X, σ², gaussian, euclidean)
    end
    P = @. (Pij + Pij') / 2n # symmetrized distribution - technically only super-diagonal matters
end

function tsne!(Y::AbstractMatrix, P::AbstractMatrix, params::OptimizationParameters = OptimizationParameters())
    n = size(X, 2)
    G = zeros(params.d, n) # gradient of Y
    G² = zeros(params.d, n) # second moments of gradient
    if params.early_iter > 0 && (params.early_compression > 0 || params.early_exaggeration > 1)
        initialize_embedding!(Y, G, G², P, params)
    end
    @. G = 0 # clearing gradient storage between
    @. G² = 0
    tsne!(Y, G, G², P, params)
    return Y
end

function tsne!(Y::AbstractMatrix, G::AbstractMatrix, G²::AbstractMatrix,
               P::AbstractMatrix, params::OptimizationParameters = OptimizationParameters())
    tsne!(Y, G, G², P, params.max_iter, params.lr, params.beta_1, params.beta_2,
          params.weight_decay, params.eps, params.dofast, params.verbose, params.debias)
    return Y
end

# it's good practice to do a few iterations with weight decay on Y
# see 3.4 Optimization Methods for t-SNE of "VISUALIZING DATA USING T-SNE"
# called "early compression"
# "early exaggeration" multiplies all Pij by a constant,
# e.g. 4 in "VISUALIZING DATA USING T-SNE", 12 in "Accelerating t-SNE using Tree-Based Algorithms"
function initialize_embedding!(Y::AbstractMatrix, G::AbstractMatrix, G²::AbstractMatrix,
                               P::AbstractMatrix, params::OptimizationParameters)
    P .*= params.early_exaggeration # multiplying all Pij by a certain constant
    tsne!(Y, G, G², P, params.early_iter, params.early_lr, params.early_beta_1,
          params.early_beta_2, params.early_compression, params.eps, params.dofast, params.verbose, params.debias)
    P ./= params.early_exaggeration
    return Y
end

function tsne!(Y::AbstractMatrix, G::AbstractMatrix, G²::AbstractMatrix, P::AbstractMatrix,
               max_iter::Int, α::Real, β₁::Real, β₂::Real, γ::Real, ε::Real, dofast::Bool, verbose::Bool, debias::Bool)
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
        αₜ = debias ? stepsize(α, β₁, β₂, t) : α
        @. Y = (1 - γ) * Y - αₜ * G / (sqrt(G²) + ε) # increment by objective + weight decay gradients
    end
    return Y
end

end # module
