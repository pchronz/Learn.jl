include("jlearn.jl")
using jlearn
using PyPlot

srand(42)
N = 300
X = rand(N, 2)
y = Array(Any, N)
for r in 1:size(X, 1)
    slope = (X[r, 2]/X[r, 1])
    y[r] = if slope > 1.
        2.
    elseif slope < 0.5
        0.
    else
        1.
    end
end
X += randn(N, 2) ./ 20
clf = GaussianNB()
fit!(clf, X, y)
y_pred = predict(clf, X)
@show y
@show y_pred
@show f1_score(y, y_pred)

