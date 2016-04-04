include("jlearn.jl")
using Base.Test
using jlearn

N = 200
kf = kfold(N, 2)
X = [collect(1:N) collect(1:N)]
y = map(x->x?1:0, reshape(X[:, 1] .> N/2, size(X)[1]))
for (idx_tr, idx_test) in kf
    X_tr = X[idx_tr, :]
    X_test = X[idx_test, :]
    y_tr = y[idx_tr]
    y_test = y[idx_test]

    clf = SVC()
    fit!(clf, X_tr, y_tr)
    y_pred = predict(clf, X_test)
    println(f1_score(y_test, y_pred))
end

