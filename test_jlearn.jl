include("jlearn.jl")
using Base.Test
using jlearn

######## Metrics #######
#y = [0, 1, 2, 0, 1, 2]
#y_pred = [0, 2, 1, 0, 0, 1]
######## Precision
#prec_dict = precision_score(y, y_pred)
#@test_approx_eq prec_dict["0"] 2/3
#@test prec_dict["1"] == 0.0
#@test prec_dict["2"] == 0.0
#@test precision_score(y, y_pred, ave_fun="macro") == 2/9
######## Recall
#recall_dict = recall_score(y, y_pred)
#@test recall_dict["0"] == 1.0
#@test recall_dict["1"] == 0.0
#@test recall_dict["2"] == 0.0
#@test_approx_eq recall_score(y, y_pred, ave_fun="macro") 1/3
######## F1
#f1_dict = f1_score(y, y_pred)
#@test f1_dict["0"] == 0.8
#@test f1_dict["1"] == 0.
#@test f1_dict["2"] == 0.
#@test f1_score(y, y_pred, ave_fun="macro") == 0.
######## R^2
#N = 50.
#y_true = collect(1:N)
#y_pred = ones(round(Int, N), 1) * mean(y_true)
#y_pred = reshape(y_pred, round(Int, N))
#@test r2_score(y_true, y_pred) == 0.
#@test r2_score(y_true, y_true) == 1.
######## MSE
#y_true = [3, -0.5, 2, 7] 
#y_pred = [2.5, 0.0, 2, 8] 
#@test mean_squared_error(y_true, y_pred) == 0.375
######## Explained variance ratio
#y_true = [3, -0.5, 2, 7] 
#y_pred = [2.5, 0.0, 2, 8] 
#@test_approx_eq explained_variance_score(y_true, y_pred) 0.9571734475374732
#
######## SVC #######
#clf = SVC()
#X = [collect(1.:100.) collect(1.:100.)]
#y = map(x->x?1:0, reshape(X[:, 1] .> 50, size(X)[1]))
#fit!(clf, X, y)
#y_pred = predict(clf, X)
#@test all(y .== y_pred)
#
######## SVR ######
#N = 50.0
#X = reshape(collect(1:N), round(Int, N), 1)
#y = reshape(map(x->x^2, X), round(Int, N))
#reg = SVR(C=100.0)
#fit!(reg, X, y)
#y_pred = predict(reg, X)
#@test_approx_eq r2_score(y, y_pred) 1.0
#
######## Cross validation #######
######## k-fold
#N = 9
#X = [collect(1:N) collect(1:N)]
#y = map(x->x?1:0, reshape(X[:, 1] .> 4, size(X)[1]))
#kf = kfold(y; k=3)
#i = 1
#for (idx_tr, idx_test) in kf
#    if i == 1
#        @test idx_tr == collect(1:6)
#        @test idx_test == collect(7:9)
#    elseif i == 2
#        @test idx_tr == collect(4:9)
#        @test idx_test == collect(1:3)
#    elseif i == 3
#        @test idx_tr == [1, 2, 3, 7, 8, 9]
#        @test idx_test == collect(4:6)
#    end
#    i += 1
#end
######## Stratified k-fold
#N = 30
#X = [collect(1:N) collect(1:N)]
#y = map(reshape(X[:, 1], size(X)[1])) do x
#    if x < 0.3*N
#        0
#    elseif x > 0.6*N
#        2
#    else
#        1
#    end
#end
#skf = stratified_kfold(y; k=3)
#i = 1
#for (idx_tr, idx_test) in skf
#    X_tr = X[idx_tr, :]
#    X_test = X[idx_test, :]
#    y_tr = y[idx_tr]
#    y_test = y[idx_test]
#    if i == 1
#        @test idx_tr == [1,2,3,4,5,9,10,11,12,13,14,15,19,20,21,22,23,24,25,26]
#        @test idx_test == [6,7,8,16,17,18,27,28,29,30]
#    elseif i == 2
#        @test idx_tr == [4,5,6,7,8,12,13,14,15,16,17,18,23,24,25,26,27,28,29,30]
#        @test idx_test == [1,2,3,9,10,11,19,20,21,22]
#    elseif i == 3
#        @test idx_tr == [1,2,3,7,8,9,10,11,15,16,17,18,19,20,21,22,27,28,29,30]
#        @test idx_test == [4,5,6,12,13,14,23,24,25,26]
#    end
#    i += 1
#end
#
######## cross_val_score
#N = 50.0
#X = [collect(1:N) collect(1:N)]
#y = map(reshape(X[:, 1], size(X)[1])) do x
#    if x < 0.3*N
#        0
#    elseif x > 0.6*N
#        2
#    else
#        1
#    end
#end
#scores = cross_val_score!(SVC(), X, y; cv=y->stratified_kfold(y; k=10))
#@test_approx_eq scores["0"] 1/1.2
#@test_approx_eq scores["1"] 1/1.2
#@test_approx_eq scores["2"] 0.9266666666666667 
#
######## GridSearchCV
#N = 50.0
#X = [collect(1:N) collect(1:N)]
#y = map(reshape(X[:, 1], size(X)[1])) do x
#    if x < 0.3*N
#        0
#    elseif x > 0.6*N
#        1
#    else
#        1
#    end
#end
#params = Dict{ASCIIString, Vector}("C"=>[0.01, 0.1, 1., 10., 100., 1000., 10000., 100000., 1000000.], "kernel"=>["rbf", "linear", "polynomial", "sigmoid"])
## XXX Why is julia not inferring the parametric type here?
##gs = GridSearchCV(SVC(), params; scoring=f1_score, cv=y->stratified_kfold(y; k=10))
#gs = GridSearchCV{SVC}(SVC(), params; scoring=f1_score, cv=y->stratified_kfold(y; k=10))
#fit!(gs, X, y)
#@test gs.best_score > 0.9
#
######## Preprocessing #######
#N = 50.0
#X = [collect(1:N) collect(1:N)]
#y = map(reshape(X[:, 1], size(X)[1])) do x
#    if x < 0.3*N
#        0
#    elseif x > 0.6*N
#        1
#    else
#        1
#    end
#end
#
######## MinMaxScaler
#mms = MinMaxScaler()
#X_mms = fit_transform!(mms, X)
#@test all(minimum(X_mms, 1) .== 0.0)
#@test all(maximum(X_mms, 1) .== 1.0)
#@test X != X_mms
#mms = MinMaxScaler(10.0, 50.0)
#X_mms = fit_transform!(mms, X)
#@test all(minimum(X_mms, 1) .== 10.0)
#@test all(maximum(X_mms, 1) .== 50.0)
#@test X != X_mms
#
######## StandardScaler
#ss = StandardScaler()
#X_ss = fit_transform!(ss, X)
#for m in mean(X_ss, 1)
#    @test_approx_eq_eps m 0.0 1e-12
#end
#for m in var(X_ss, 1)
#    @test_approx_eq m 1.0 
#end
#@test X != X_ss
#
######## PCA
#N = 50.0
#mt = MersenneTwister(42)
#X = rand(mt, round(Int, N), 2)
#pca = PCA()
#X_pca = fit_transform!(pca, X)
#@test_approx_eq pca.explained_variance_ratio_[1] 0.5420700035608417 
#@test_approx_eq pca.explained_variance_ratio_[2] 0.45792999643915844 
#@test_approx_eq pca.M.proj[1] -0.8435103551544426
#@test_approx_eq pca.M.proj[2] -0.5371129124748595 
#
######## ICA
#N = 50.0
#mt = MersenneTwister(42)
#X = rand(mt, round(Int, N), 2)
#ica = FastICA(;n_components=2, whiten=true)
#X_ica = fit_transform!(ica, X)
#@test_approx_eq ica.M.mean[1] 0.4831187641151334 
#@test_approx_eq ica.M.mean[2] 0.4641596857176869 
#
######## Pipeline #######
#N = 50.0
#X = [collect(1:N) collect(1:N)]
#y = map(reshape(X[:, 1], size(X)[1])) do x
#    if x < 0.3*N
#        0
#    elseif x > 0.6*N
#        1
#    else
#        1
#    end
#end
#
#pipe = Pipeline([("mms", MinMaxScaler()), ("ss", StandardScaler())], ("svc", SVC()))
#fit!(pipe, X, y)
#y_pred = predict(pipe, X)
#@test_approx_eq f1_score(y, y_pred)["1"] 1.0
#@test f1_score(y, y_pred)["0"] == 1.0
#
#params = Dict{ASCIIString, Vector}("mms__range_min"=>[0.0], "mms__range_max"=>[1.0], "svc__C"=>[0.01, 0.1, 1., 10., 100., 1000., 10000., 100000., 1000000.], "svc__kernel"=>["rbf", "linear", "polynomial", "sigmoid"])
## XXX Why doesn't type infernce work here?
##GridSearchCV(pipe, params)
#gs = GridSearchCV{Pipeline}(pipe, params)
#fit!(gs, X, y)
#@test gs.best_score == 1.0
#@test gs.best_params[:mms__range_min] == 0.0
#@test gs.best_params[:mms__range_max] == 1.0
#@test gs.best_params[:svc__kernel] == "linear"
#@test gs.best_params[:svc__C] == 0.1
#@test gs.best_estimator.preprocessors[1][1] == "mms"
#@test gs.best_estimator.preprocessors[1][2].range_min == 0.0
#@test gs.best_estimator.preprocessors[1][2].range_max == 1.0
#@test gs.best_estimator.preprocessors[2][1] == "ss"
#@test gs.best_estimator.estimator[1] == "svc"
#@test gs.best_estimator.estimator[2].svm.kernel == "linear"
#@test gs.best_estimator.estimator[2].svm.C == 0.1
#
######## Linear Models #######
######## Linear regression
#N = 50.0
#X = [collect(1:N) collect(1:N)]
#mt = MersenneTwister(42)
#X[:, 2] += rand(mt, round(Int, N))
#mt = MersenneTwister(42)
#y = reshape(map(x->x + rand(mt), X[:, 1]), round(Int, N))
#reg = LinearRegression()
#fit!(reg, X, y)
#y_pred = predict(reg, X)
#@test r2_score(y, y_pred) == 1.0
#@test score(reg, X, y) == 1.0
#
####### Logistic regression
#clf = LogisticRegression()
#srand(42)
#N = 100
#X = rand(N, 2)
#y = Array(Float64, N)
#for r in 1:size(X, 1)
#    y[r] = (X[r, 2]/X[r, 1]) > 1. ? 1. : 0.
#end
#X += randn(N, 2) ./ 10
##scatter(X[y .== 1., 1], X[y .== 1., 2], color="red")
##scatter(X[y .== 0., 1], X[y .== 0., 2], color="blue")
##show()
#fit!(clf, X, y)
#y_pred = predict(clf, X)
#@test_approx_eq f1_score(y, y_pred)["0.0"] 0.8913043478260869 
#@test_approx_eq f1_score(y, y_pred)["1.0"] 0.9074074074074074 
#
######## Ensemble Methods #######
######## RandomForestRegressor
#N = 50.0
#X = [collect(1:N) collect(1:N)]
#mt = MersenneTwister(42)
#X[:, 2] += rand(mt, round(Int, N))
#mt = MersenneTwister(42)
#y = reshape(map(x->x + rand(mt), X[:, 1]), round(Int, N))
#reg = RandomForestRegressor()
#fit!(reg, X, y)
#y_pred = predict(reg, X)
#@test r2_score(y, y_pred) > 0.98
#@test score(reg, X, y) > 0.98
#
######## RandomForestClassifier
#srand(42)
#clf = RandomForestClassifier()
#X = [collect(1.:100.) collect(1.:100.)]
#y = map(x->x?1:0, reshape(X[:, 1] .> 50, size(X)[1]))
#fit!(clf, X, y)
#y_pred = predict(clf, X)
#@test (y'*y_pred)[1] == 50
#
######## DecisionTreeRegressor
#N = 50.0
#X = [collect(1:N) collect(1:N)]
#mt = MersenneTwister(42)
#X[:, 2] += rand(mt, round(Int, N))
#mt = MersenneTwister(42)
#y = reshape(map(x->x + rand(mt), X[:, 1]), round(Int, N))
#reg = DecisionTreeRegressor()
#fit!(reg, X, y)
#y_pred = predict(reg, X)
#@test r2_score(y, y_pred) > 0.99
#@test score(reg, X, y) > 0.99
#
######## DecisionTreeClassifier
#clf = DecisionTreeClassifier()
#X = [collect(1.:100.) collect(1.:100.)]
#y = map(x->x?1:0, reshape(X[:, 1] .> 50, size(X)[1]))
#fit!(clf, X, y)
#y_pred = predict(clf, X)
#@test all(y_pred .== y)
#
######## Multilabel classification #######
######## OneVsOne classification via logistic regression
#clf = LogisticRegression()
#srand(42)
#N = 300
#X = rand(N, 2)
#y = Array(Float64, N)
#for r in 1:size(X, 1)
#    slope = (X[r, 2]/X[r, 1])
#    y[r] = if slope > 1.
#        2.
#    elseif slope < 0.5
#        0.
#    else
#        1.
#    end
#end
#X += randn(N, 2) ./ 20
##scatter(X[y .== 0., 1], X[y .== 0., 2], color="blue")
##scatter(X[y .== 1., 1], X[y .== 1., 2], color="red")
##scatter(X[y .== 2., 1], X[y .== 2., 2], color="green")
##show()
#fit!(clf, X, y)
#y_pred = predict(clf, X)
#@test_approx_eq f1_score(y, y_pred)["0.0"] 0.9264705882352942
#@test_approx_eq f1_score(y, y_pred)["1.0"] 0.8395061728395061
#@test_approx_eq f1_score(y, y_pred)["2.0"] 0.9403973509933775
#
######## OneVsOne classification via SVM
#srand(42)
#N = 300
#X = rand(N, 2)
#y = Array(Any, N)
#for r in 1:size(X, 1)
#    slope = (X[r, 2]/X[r, 1])
#    y[r] = if slope > 1.
#        :2
#    elseif slope < 0.5
#        '0'
#    else
#        "1"
#    end
#end
#X += randn(N, 2) ./ 20
#clf = SVC()
#fit!(clf, X, y)
#y_pred = predict(clf, X)
#f1 = f1_score(y, y_pred)
#@test_approx_eq f1["0"] 0.896 
#@test_approx_eq f1["1"] 0.7976878612716763
#@test_approx_eq f1["2"] 0.9205298013245033
#
######## OneVsOne classification via random forest
#srand(42)
#N = 300
#X = rand(N, 2)
#y = Array(Any, N)
#for r in 1:size(X, 1)
#    slope = (X[r, 2]/X[r, 1])
#    y[r] = if slope > 1.
#        :2
#    elseif slope < 0.5
#        '0'
#    else
#        "1"
#    end
#end
#X += randn(N, 2) ./ 20
#clf = RandomForestClassifier()
#fit!(clf, X, y)
#y_pred = predict(clf, X)
#f1 = f1_score(y, y_pred)
#@test_approx_eq f1["0"] 0.9618320610687023
#@test_approx_eq f1["1"] 0.8848484848484849
#@test_approx_eq f1["2"] 0.9539473684210527
#
######## OneVsOne classification via decision tree
#srand(42)
#N = 300
#X = rand(N, 2)
#y = Array(Any, N)
#for r in 1:size(X, 1)
#    slope = (X[r, 2]/X[r, 1])
#    y[r] = if slope > 1.
#        :2
#    elseif slope < 0.5
#        '0'
#    else
#        "1"
#    end
#end
#X += randn(N, 2) ./ 20
#clf = DecisionTreeClassifier()
#fit!(clf, X, y)
#y_pred = predict(clf, X)
#f1 = f1_score(y, y_pred)
#@test_approx_eq f1["0"] 1.0
#@test_approx_eq f1["1"] 1.0
#@test_approx_eq f1["2"] 1.0
#
######## OneVsOne classification via naive bayes
#srand(42)
#N = 300
#X = rand(N, 2)
#y = Array(Any, N)
#for r in 1:size(X, 1)
#    slope = (X[r, 2]/X[r, 1])
#    y[r] = if slope > 1.
#        :2
#    elseif slope < 0.5
#        '0'
#    else
#        "1"
#    end
#end
#X += randn(N, 2) ./ 20
#clf = GaussianNB()
#fit!(clf, X, y)
#y_pred = predict(clf, X)
#f1 = f1_score(y, y_pred)
#@test_approx_eq f1["0"] 0.916030534351145
#@test_approx_eq f1["1"] 0.8297872340425531
#@test_approx_eq f1["2"] 0.9252669039145908
#
######## OneVsAll classification via logistic regression
#clf = LogisticRegression(strategy=OneVsAllStrategy())
#srand(42)
#N = 300
#X = rand(N, 2)
#y = Array(Float64, N)
#for r in 1:size(X, 1)
#    slope = (X[r, 2]/X[r, 1])
#    y[r] = if slope > 1.
#        2.
#    elseif slope < 0.5
#        0.
#    else
#        1.
#    end
#end
#X += randn(N, 2) ./ 20
##scatter(X[y .== 0., 1], X[y .== 0., 2], color="blue")
##scatter(X[y .== 1., 1], X[y .== 1., 2], color="red")
##scatter(X[y .== 2., 1], X[y .== 2., 2], color="green")
##show()
#fit!(clf, X, y)
#y_pred = predict(clf, X)
#@test_approx_eq f1_score(y, y_pred)["0.0"] 0.5833333333333334
#@test_approx_eq f1_score(y, y_pred)["1.0"] 0.33070866141732286 
#@test_approx_eq f1_score(y, y_pred)["2.0"] 0.8693009118541033
#
######## OneVsAll classification via SVM
#srand(42)
#N = 300
#X = rand(N, 2)
#y = Array(Any, N)
#for r in 1:size(X, 1)
#    slope = (X[r, 2]/X[r, 1])
#    y[r] = if slope > 1.
#        :2
#    elseif slope < 0.5
#        '0'
#    else
#        "1"
#    end
#end
#X += randn(N, 2) ./ 20
#clf = SVC(strategy=OneVsAllStrategy())
#fit!(clf, X, y)
#y_pred = predict(clf, X)
#f1 = f1_score(y, y_pred)
#@test_approx_eq f1["0"] 0.7974683544303798
#@test_approx_eq f1["1"] 0.0
#@test_approx_eq f1["2"] 0.888888888888889
#
######## OneVsAll classification via random forest
#srand(42)
#N = 300
#X = rand(N, 2)
#y = Array(Any, N)
#for r in 1:size(X, 1)
#    slope = (X[r, 2]/X[r, 1])
#    y[r] = if slope > 1.
#        :2
#    elseif slope < 0.5
#        '0'
#    else
#        "1"
#    end
#end
#X += randn(N, 2) ./ 20
#clf = RandomForestClassifier(strategy=OneVsAllStrategy())
#fit!(clf, X, y)
#y_pred = predict(clf, X)
#f1 = f1_score(y, y_pred)
#@test_approx_eq f1["0"] 0.9503546099290779
#@test_approx_eq f1["1"] 0 
#@test_approx_eq f1["2"] 0.9666666666666667
#
######## OneVsAll classification via decision tree
#srand(42)
#N = 300
#X = rand(N, 2)
#y = Array(Any, N)
#for r in 1:size(X, 1)
#    slope = (X[r, 2]/X[r, 1])
#    y[r] = if slope > 1.
#        :2
#    elseif slope < 0.5
#        '0'
#    else
#        "1"
#    end
#end
#X += randn(N, 2) ./ 20
#clf = DecisionTreeClassifier(strategy=OneVsAllStrategy())
#fit!(clf, X, y)
#y_pred = predict(clf, X)
#f1 = f1_score(y, y_pred)
#@test_approx_eq f1["0"] 1.0
#@test_approx_eq f1["1"] 0.0
#@test_approx_eq f1["2"] 1.0
#
######## OneVsAll classification via GaussianNB
#srand(42)
#N = 300
#X = rand(N, 2)
#y = Array(Any, N)
#for r in 1:size(X, 1)
#    slope = (X[r, 2]/X[r, 1])
#    y[r] = if slope > 1.
#        :2
#    elseif slope < 0.5
#        '0'
#    else
#        "1"
#    end
#end
#X += randn(N, 2) ./ 20
#clf = GaussianNB(strategy=OneVsAllStrategy())
#fit!(clf, X, y)
#y_pred = predict(clf, X)
#f1 = f1_score(y, y_pred)
#@test_approx_eq f1["0"] 0.9117647058823528
#@test_approx_eq f1["1"] 0.0
#@test_approx_eq f1["2"] 0.9351535836177474
#
######## Discriminant Analysis #######
######## Linear Discriminant Analysis
#srand(42)
#N = 300
#X = rand(N, 2)
#y = Array(Any, N)
#for r in 1:size(X, 1)
#    slope = (X[r, 2]/X[r, 1])
#    y[r] = if slope > 1.
#        :2
#    elseif slope < 0.5
#        '0'
#    else
#        "1"
#    end
#end
#X += randn(N, 2) ./ 20
#clf = LinearDiscriminantAnalysis()
#fit!(clf, X, y)
#y_pred = predict(clf, X)
#f1 = f1_score(y, y_pred)
#@test_approx_eq f1["0"] 0.9022556390977443
#@test_approx_eq f1["1"] 0.7958115183246073
#@test_approx_eq f1["2"] 0.898550724637681

####### Clustering #######
import Clustering
####### Kmeans
srand(13)
N_2 = 150
X_1 = randn(N_2, 2) .- 3.0
X_2 = randn(N_2, 2) .+ 3.0
X_3 = randn(N_2, 2) .+ 6.0
X = vcat(X_1, X_2, X_3)
clust = Kmeans(n_clusters=3)
fit!(clust, X)
@test all(predict(clust, X) .== Clustering.assignments(clust.estimator)')
@test score(clust, X) == clust.estimator.totalcost

params = Dict{ASCIIString, Vector}("n_clusters"=>[2, 3, 4])
gs = GridSearchCV{Kmeans}(Kmeans(), params)
fit!(gs, X)
@test_approx_eq gs.best_score 0.6654593229441417
# XXX The best result is given as 2, but realy should be 3. The kmeans implementation 
# in module "Clustering" seem erroneous.
@test gs.best_params[:n_clusters] == 2
params = Dict{ASCIIString, Vector}("mms__range_min"=>[0.0], "mms__range_max"=>[1.0], "km__n_clusters"=>[2, 3, 4])
# XXX Why doesn't type infernce work here?
#GridSearchCV(pipe, params)
pipe = Pipeline([("mms", MinMaxScaler()), ("ss", StandardScaler())], ("km", Kmeans()))
gs = GridSearchCV{Pipeline}(pipe, params)
fit!(gs, X)
@test gs.best_params[:km__n_clusters] == 2

