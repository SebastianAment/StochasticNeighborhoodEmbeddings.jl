using MLDatasets
using HDF5
using StatsBase
using LinearAlgebra

# load full training set
train_x, train_y = MNIST.traindata()
# load full test set
# test_x,  test_y  = MNIST.testdata()

# u = 40 # from "VISUALIZING DATA USING T-SNE"
u = 50 # from "Accelerating t-SNE using Tree-Based Algorithms"
n = size(train_x, 3)

X = train_x
X = reshape(X, :, n) # vectorize images
labels = train_y
if false
    k = 1024
    subsample = sample(1:n, k, replace = false)
    X = X[:, subsample]
    labels = labels[subsample]
end

# reduce dimensionality using PCA to
U, S, Vt = svd(X)
V = Vt'
d = 50
X = V[1:d, :]
@. X *= S[1:d]

using StochasticNeighborhoodEmbeddings
using StochasticNeighborhoodEmbeddings: OptimizationParameters,
            optimized_symmetrized_neighbor, initialize_embedding!

total_iter = 64
early_iter = 64
max_iter = total_iter - early_iter

beta_1 = 0.9
beta_2 = 0.99
weight_decay = 0
lr = 3e-1
dofast = false

early_beta_1 = 0.5
early_beta_2 = 0.9
early_lr = 3e-1
early_compression = 0
early_exaggeration = 8

params = OptimizationParameters(max_iter = max_iter, beta_1 = beta_1,
        beta_2 = beta_2, lr = lr, weight_decay = weight_decay,
        early_iter = early_iter, early_beta_1 = early_beta_1, early_beta_2 = early_beta_2,
        early_lr = early_lr, early_compression = early_compression,
        early_exaggeration = early_exaggeration, dofast = dofast,
        verbose = true, debias = false)

Y = 1e-2randn(params.d, size(X, 2))
P = optimized_symmetrized_neighbor(X, u, params.dofast)
G = zeros(params.d, size(X, 2)) # gradient of Y
G² = zeros(params.d, size(X, 2)) # second moments of gradient
exaggerations = 8:-1:1
for i in exaggerations # slowly reducing early_exaggeration
    params.early_exaggeration = i
    @time initialize_embedding!(Y, G, G², P, params)
end
params.early_iter = 0
params.max_iter = 1024 - length(exaggerations) * early_iter
params.early_exaggeration = 0
@time tsne!(Y, G, G², P, params) # fine-tuning

f = h5open("MNIST_TSNE_full.h5", "w")
f["Y"] = Y
f["X"] = X
f["labels"] = labels
f["max_iter"] = max_iter
f["lr"] = lr
f["beta_1"] = beta_1
f["beta_2"] = beta_2
f["early_iter"] = early_iter
f["early_lr"] = early_lr
f["early_beta_1"] = early_beta_1
f["early_beta_2"] = early_beta_2
f["early_compression"] = early_compression
f["early_exaggeration"] = early_exaggeration
close(f)

# using Plots
# doplot = true
# if doplot
#     YC = [Y[:, ==(i).(labels)] for i in 1:10]
#     scatter()
#     for (i, yc) in enumerate(YC)
#         scatter!(yc[1, :], yc[2, :], label = "$(i-1)")
#     end
#     gui()
# end
