include("jlearn.jl")
using Base.Test
using jlearn

####### Metrics #######
y = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]
prec_dict = precision_score(y, y_pred)
@test_approx_eq prec_dict["0"] 2/3
@test prec_dict["1"] == 0.0
@test prec_dict["2"] == 0.0
@test precision_score(y, y_pred, ave_fun="macro") == 2/9
recall_dict = recall_score(y, y_pred)
@test recall_dict["0"] == 1.0
@test recall_dict["1"] == 0.0
@test recall_dict["2"] == 0.0
@test_approx_eq recall_score(y, y_pred, ave_fun="macro") 1/3
f1_dict = f1_score(y, y_pred)
@test f1_dict["0"] == 0.8
@test f1_dict["1"] == 0.
@test f1_dict["2"] == 0.
@test f1_score(y, y_pred, ave_fun="macro") == 0.

####### SVC #######
clf = SVC()
X = [collect(1:100) collect(1:100)]
y = map(x->x?1:0, reshape(X[:, 1] .> 50, size(X)[1]))
fit!(clf, X, y)
y_pred = predict(clf, X)
@test all(y .== y_pred)

