function gradient!(G::AbstractMatrix, P::AbstractMatOrFac, Q::AbstractMatOrFac,
                   Q²::AbstractMatOrFac, Y::AbstractMatrix, Z::Real, β::Real = 0)
    @. G *= β
    attractive_force!(G, P, Q, Y) # for t-SNE
    repulsive_force!(G, P, Q², Y, Z)
    return G
end

# attractive force can be calculated quickly just based on sparse approximation to P
# Z is normalization constant of Q
# function attractive_force!(G::AbstractMatrix, P::SparseMatrixCSC, Q::AbstractMatOrFac, Y::AbstractMatrix)
#     y = [c for c in eachcol(Y)]
#     I, J, _ = findnz(P)
#     for (i, j) in zip(I, J) # IDEA: parallelize over columns?
#          @. G[:, i] += 4 * P[i, j] * Q[i, j] * (y[i] - y[j])
#     end
#     return G
# end

# generic fallback
function attractive_force!(G::AbstractMatrix, P::AbstractMatOrFac, Q::AbstractMatOrFac, Y::AbstractMatrix)
    PQ = hadamard_product(P, Q) # fast through sparsity in P, check if special case is necessary
    PQY = PQ * Y'
    sumPQ = sum(PQ, dims = 2)
    @. G += 4(sumPQ' * Y - PQY')
    return G
end

function repulsive_force!(G::AbstractMatrix, P::AbstractMatOrFac, Q²::AbstractMatOrFac, Y::AbstractMatrix, Z::Real)
    n, m = size(P) # is square
    # we compute in one product
    # 1. row-wise sum of Q²
    # 2. for each dimension, run fast transform once -> d * n * log(n) complexity
    Y1 = hcat(Y', ones(m)) # IDEA: lazy or re-allocate?
    Q²Y1 = Q²*Y1 # does all relevant computations in one matrix-matrix product
    Q²Y = @view Q²Y1[:, 1:2]
    sumQ² = @view Q²Y1[:, 3]
    @. G += 4/Z * (Q²Y' - sumQ²' * Y) # can this be re-written as a single product?
    return G
end

################################## ADAM ########################################
# implementation of adam optimizer
# gradient first and second moment increment using
function gradient!(G::AbstractMatrix, G²::AbstractMatrix,
                    P::AbstractMatOrFac, Q::AbstractMatOrFac, Q²::AbstractMatOrFac,
                    Y::AbstractMatrix, Z::Real, β₁::Real = 0, β₂::Real = 0)
    @. G *= β₁
    @. G² *= β₂
    attractive_force!(G, G², P, Q, Y, β₁, β₂) # for t-SNE
    repulsive_force!(G, G², P, Q², Y, Z, β₁, β₂)
    return G, G²
end

# generic fallback
function attractive_force!(G::AbstractMatrix, G²::AbstractMatrix, P::AbstractMatOrFac,
                           Q::AbstractMatOrFac, Y::AbstractMatrix, β₁::Real, β₂::Real)
    PQ = hadamard_product(P, Q) # fast through sparsity in P,  check if special case is necessary
    PQY = PQ * Y'
    sumPQ = sum(PQ, dims = 2)
    ∇ = @. 4(sumPQ' * Y - PQY')
    @. G += (1-β₁) * ∇ # can this be re-written as a single product?
    @. G² += (1-β₂) * ∇^2
    return G
end

function repulsive_force!(G::AbstractMatrix, G²::AbstractMatrix, P::AbstractMatOrFac,
                Q²::AbstractMatOrFac, Y::AbstractMatrix, Z::Real, β₁::Real, β₂::Real)
    n, m = size(P) # is square
    # we compute in one product
    # 1. row-wise sum of Q²
    # 2. for each dimension, run fast transform once -> d * n * log(n) complexity
    Y1 = hcat(Y', ones(m)) # IDEA: lazy or re-allocate?
    Q²Y1 = Q²*Y1 # does all relevant computations in one matrix-matrix product
    Q²Y = @view Q²Y1[:, 1:2]
    sumQ² = @view Q²Y1[:, 3]
    # @. G += 4/Z * (Q²Y' - sumQ²' * Y) # can this be re-written as a single product?
    # @. G² += (4/Z * (Q²Y' - sumQ²' * Y))^2
    ∇ = @. 4/Z * (Q²Y' - sumQ²' * Y)
    @. G += (1-β₁) * ∇ # can this be re-written as a single product?
    @. G² += (1-β₂) * ∇^2
    return G
end

# stepsize to debias moments for Adam
stepsize(α::Real, β₁::Real, β₂::Real, t::Int) = α * sqrt(1-β₂^t)/(1-β₁^t)
