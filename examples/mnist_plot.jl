using Plots
plotly()
using HDF5
using ColorSchemes

# 11750s
# Plots.scalefontsizes(2) # only execute this once

dir = "StochasticNeighborhoodEmbeddings/examples/"
name =  "MNIST_TSNE_full_4"
f = h5open(dir * name * ".h5", "r")
labels = read(f["labels"])
Y = read(f["Y"])
close(f)

colors = palette(:seaborn_bright)
# colors = palette(:seaborn_deep)
# colors = palette(:seaborn_colorblind)
# colors = palette(:seaborn_muted)
#

YC = [Y[:, ==(i).(labels)] for i in 0:9]
scatter()
for (i, yc) in enumerate(YC)
    scatter!(yc[1, :], yc[2, :], label = "$(i-1)", markersize = 1,
    markerstrokewidth = 0, axis = false, ticks = false, markercolor = colors[i],
    legend = :outertopright)
end
gui()
savefig(name * ".pdf")
