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
k = 512
if true
    subsample = sample(1:n, k, replace = false)
else
    subsample = 1:n
end

X = train_x[:, :, subsample]
X = reshape(X, :, length(subsample)) # vectorize image
labels = train_y[subsample]

# reduce dimensionality using PCA to
U, S, Vt = svd(X)
V = Vt'
d = 50
X = V[1:d, :]
@. X *= S[1:d]

using StochasticNeighborhoodEmbeddings
max_iter = 128
beta_1 = 0.9
beta_2 = 0.99
lr = 1

early_iter = 64
early_beta_1 = 0.5
early_beta_2 = 0.9
early_lr = 1e-1
early_compression = 1
early_exaggeration = 12

@time Y = tsne(X, u, max_iter = max_iter, beta_1 = beta_1, beta_2 = beta_2, lr = lr,
        early_iter = early_iter, early_beta_1 = early_beta_1, early_beta_2 = early_beta_2,
        early_lr = early_lr, early_compression = early_compression,
        early_exaggeration = early_exaggeration, verbose = true, dofast = true)

f = h5open("MNIST_TSNE.h5", "w")
f["Y"] = Y
f["X"] = X
f["labels"] = labels
f["max_iter"] = max_iter
f["lr"] = lr
f["beta_1"] = max_iter
f["beta_2"] = max_iter
f["early_iter"] = max_iter
f["early_lr"] = max_iter
f["early_beta_1"] = max_iter
f["early_beta_2"] = max_iter
f["early_compression"] = early_compression
f["early_exaggeration"] = early_exaggeration
close(f)

using Plots
doplot = false
if doplot
    f = h5open("MNIST_TSNE.h5", "r")
    labels = read(f["labels"])
    Y = read(f["Y"])
    close(f)

    YC = [Y[:, ==(i).(labels)] for i in 1:10]
    scatter()
    for (i, yc) in enumerate(YC)
        scatter!(yc[1, :], yc[2, :], label = "$(i-1)")
    end
    gui()
end
