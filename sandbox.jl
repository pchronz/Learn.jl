include("jlearn.jl")
using jlearn
using PyPlot
import Clustering

srand(13)
N_2 = 150
X_1 = randn(N_2, 2) .- 3.0
X_2 = randn(N_2, 2) .+ 3.0
X = vcat(X_1, X_2)
clust = Kmeans(n_clusters=2)
fit!(clust, X)
@show all(predict(clust, X) .== Clustering.assignments(clust.estimator)')
scatter(X_1[:, 1], X_2[:, 2], color="red")
scatter(X_2[:, 1], X_2[:, 2], color="green")
#scatter(centers[1, 1], centers[2, 1], color="cyan")
#scatter(centers[1, 2], centers[2, 2], color="yellow")
#show()

