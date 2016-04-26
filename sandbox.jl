include("jlearn.jl")
using Base.Test
using jlearn
using PyPlot

####### ICA
N = 50.0
mt = MersenneTwister(42)
X = rand(mt, round(Int, N), 2)
ica = FastICA(n_components=2, whiten=false, max_iter=2)
X_ica = fit_transform!(ica, X)
dump(ica)

