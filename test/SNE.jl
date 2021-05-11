module TestSNE

using StochasticNeighborhoodEmbeddings
using StochasticNeighborhoodEmbeddings: conditional_neighbor, joint_neighbor,
        two_bump_data, gaussian, euclidean, k_nearest_neighbors

using Test

d = 16
n = 32
X = two_bump_data(d, n)
x = [c for c in eachcol(X)]
# TODO: standardize
@testset "neighbor matrices" begin
    σ² = 1:n
    P = conditional_neighbor(X, σ²)
    @test vec(sum(P, dims = 1)) ≈ ones(n)

    # test if sigma is constant for each column
    p1 = @. gaussian(euclidean(x[1:1], x[2:end]) / σ²[1])
    p1 ./= sum(p1)
    @test P[2:end, 1] ≈ p1
end

using StochasticNeighborhoodEmbeddings: optimize_variances, perplexity, entropy
@testset "variance optimization" begin
    u = 30
    σ² = optimize_variances(X, u)
    @test all(!isnan, σ²)
    P = conditional_neighbor(X, σ²)
    # println([perplexity(p) for p in eachcol(P)].-u)
    @test all(eachcol(P)) do p isapprox(perplexity(p), u, atol = 1e-3) end
end

using SparseArrays
@testset "sparse" begin
    # now sparse case
    σ² = 1:n
    u = 5 # perplexity
    neighbors = k_nearest_neighbors(X, floor(3u))
    P = conditional_neighbor(X, σ², neighbors)
    @test P isa SparseMatrixCSC
    @test vec(sum(P, dims = 1)) ≈ ones(n)

    σ² = optimize_variances(X, u, neighbors)
    P = conditional_neighbor(X, σ², neighbors)
    # println([perplexity(p) for p in eachcol(P)].-u)
    @test all(eachcol(P)) do p isapprox(perplexity(p), u, atol = 1e-3) end
end

@testset "gradient" begin

end

# u = 30 # perplexity: higher value means distribution P[:, i] has larger entropy -> more uneven
# Y = tsne(X, u)

end # module
