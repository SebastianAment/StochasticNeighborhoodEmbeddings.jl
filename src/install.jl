using Pkg

ssh = true
if ssh
    git = "git@github.com:SebastianAment/"
else
    git = "https://github.com/SebastianAment/"
end

add(x) = Pkg.add(Pkg.PackageSpec(url = git * x * ".git"))

add("LazyInverse.jl")
add("LazyLinearAlgebra.jl")
add("LinearAlgebraExtensions.jl")
add("KroneckerProducts.jl")
add("WoodburyIdentity.jl")
add("CovarianceFunctions.jl")

git = "git@github.com:jpryan1/"
add_fkt(x) = Pkg.add(Pkg.PackageSpec(url = git * x * ".git"))
add_fkt("FastKernelTransform.jl")
