include("jlearn.jl")
using jlearn

clf = RandomForestClassifier()
X = [collect(1.:100.) collect(1.:100.)]
y = map(x->x?1:0, reshape(X[:, 1] .> 50, size(X)[1]))
fit!(clf, X, y)
y_pred = predict(clf, X)
@show y_pred
@show f1_score(y, y_pred)

