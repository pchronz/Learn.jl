include("jlearn.jl")
using jlearn
using PyPlot

clf = LogisticRegression()
srand(42)
N = 300
X = rand(N, 2)
y = Array(Float64, N)
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
#scatter(X[y .== 0., 1], X[y .== 0., 2], color="blue")
#scatter(X[y .== 1., 1], X[y .== 1., 2], color="red")
#scatter(X[y .== 2., 1], X[y .== 2., 2], color="green")
#show()
fit!(clf, X, y)
y_pred = predict(clf, X)
@show y
@show y_pred
@show f1_score(y, y_pred)

