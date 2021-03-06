using StochasticNeighborhoodEmbeddings

d = 16 # relatively high-dimensional
n = 64
X1 = .1randn(d, n) .+ ones(d)
X2 = .1randn(d, n) .- ones(d)
X = hcat(X1, X2)
u = 30 # perplexity: higher value means distribution P[:, i] has larger entropy -> more uneven
Y = tsne(X, u, max_iter = 64, early_iter = 0, verbose = true)

using Plots
@views scatter(Y[1,:], Y[2,:])
gui()
