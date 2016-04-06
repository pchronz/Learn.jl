include("jlearn.jl")
using Base.Test
using jlearn
using PyPlot

# TODO MinMax
# TODO Whitening
# TODO PCA
# TODO ICA
# TODO Pipeline

N = 50
X = [collect(1:N) collect(1:N)]
y = map(reshape(X[:, 1], size(X)[1])) do x
    if x < 0.3*N
        0
    elseif x > 0.6*N
        1
    else
        1
    end
end
params = Dict{ASCIIString, Vector}("C"=>[0.01, 0.1, 1., 10., 100., 1000., 10000., 100000., 1000000.], "kernel"=>["rbf", "linear", "polynomial", "sigmoid"])
gs = GridSearchCV(SVC(), params; scoring=precision_score, cv=y->stratified_kfold(y; k=10))
fit!(gs, X, y)
@show gs.scores
@show gs.best_score
@show gs.best_params

