include("jlearn.jl")
using Base.Test
using jlearn
using PyPlot

N = 50.0
X = [collect(1:N) collect(1:N)]
mt = MersenneTwister(42)
X[:, 2] += rand(mt, round(Int, N))
mt = MersenneTwister(42)
y = reshape(map(x->x + rand(mt), X[:, 1]), round(Int, N))
reg = DecisionTreeRegressor()
fit!(reg, X, y)
y_pred = predict(reg, X)
@show r2_score(y, y_pred)
@show score(reg, X, y)

