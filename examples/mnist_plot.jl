using Plots
using HDF5

dir = "StochasticNeighborhoodEmbeddings/examples/"
f = h5open(dir * "MNIST_TSNE_full.h5", "r")
labels = read(f["labels"])
Y = read(f["Y"])
close(f)

YC = [Y[:, ==(i).(labels)] for i in 1:10]
scatter()
for (i, yc) in enumerate(YC)
    scatter!(yc[1, :], yc[2, :], label = "$(i-1)")
end
gui()
