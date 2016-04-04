include("jlearn.jl")
using jlearn
#clf = SVC()
#X = [collect(1:100) collect(1:100)]
#y = map(x->x?1:0, reshape(X[:, 1] .> 50, size(X)[1]))
#fit(clf, X, y)
#y_pred = predict(clf, X)
#println(precision_score(y, y_pred))
#println(recall_score(y, y_pred))
#println(f1_score(y, y_pred))

y = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]
info("Precision...")
println(precision_score(y, y_pred))
println(precision_score(y, y_pred, ave_fun="macro"))
info("Recall...")
println(recall_score(y, y_pred))
println(recall_score(y, y_pred, ave_fun="macro"))
info("F1...")
println(f1_score(y, y_pred))
println(f1_score(y, y_pred, ave_fun="macro"))

