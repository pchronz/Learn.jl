include("jlearn.jl")
using jlearn
import PyPlot
import LIBSVM

#srand(13)
N = 150
X = reshape(collect(1:float(N)), N, 1)
y = reshape(map(x->x^2, X), N)
idx = collect(1:N)
shuffle!(idx)
X_tr = X[idx[1:120], :]
y_tr = y[idx[1:120]]
X_test = X[idx[121:end], :]
y_test = y[idx[121:end]]
svm = LIBSVM.svmtrain(y_tr, X_tr'; svm_type=Int32(3), C=1000.)
y_pr_tr = LIBSVM.svmpredict(svm, X_tr')[1]
y_pr_test = LIBSVM.svmpredict(svm, X_test')[1]
PyPlot.scatter(X_tr, y_tr; color="blue")
PyPlot.scatter(X_tr, y_pr_tr; color="red")
PyPlot.scatter(X_test, y_pr_test; color="green")
PyPlot.show()

