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
X = [collect(1.:100.) collect(1.:100.)]
y = map(x->x?1:0, reshape(X[:, 1] .> 50, size(X)[1]))
fit!(clf, X, y)
y_pred = predict(clf, X)
@test all(y .== y_pred)

####### Cross validation #######
####### k-fold
N = 9
X = [collect(1:N) collect(1:N)]
y = map(x->x?1:0, reshape(X[:, 1] .> 4, size(X)[1]))
kf = kfold(y; k=3)
i = 1
for (idx_tr, idx_test) in kf
    if i == 1
        @test idx_tr == collect(1:6)
        @test idx_test == collect(7:9)
    elseif i == 2
        @test idx_tr == collect(4:9)
        @test idx_test == collect(1:3)
    elseif i == 3
        @test idx_tr == [1, 2, 3, 7, 8, 9]
        @test idx_test == collect(4:6)
    end
    i += 1
end
####### Stratified k-fold
N = 30
X = [collect(1:N) collect(1:N)]
y = map(reshape(X[:, 1], size(X)[1])) do x
    if x < 0.3*N
        0
    elseif x > 0.6*N
        2
    else
        1
    end
end
skf = stratified_kfold(y; k=3)
i = 1
for (idx_tr, idx_test) in skf
    X_tr = X[idx_tr, :]
    X_test = X[idx_test, :]
    y_tr = y[idx_tr]
    y_test = y[idx_test]
    if i == 1
        @test idx_tr == [1,2,3,4,5,9,10,11,12,13,14,15,19,20,21,22,23,24,25,26]
        @test idx_test == [6,7,8,16,17,18,27,28,29,30]
    elseif i == 2
        @test idx_tr == [4,5,6,7,8,12,13,14,15,16,17,18,23,24,25,26,27,28,29,30]
        @test idx_test == [1,2,3,9,10,11,19,20,21,22]
    elseif i == 3
        @test idx_tr == [1,2,3,7,8,9,10,11,15,16,17,18,19,20,21,22,27,28,29,30]
        @test idx_test == [4,5,6,12,13,14,23,24,25,26]
    end
    i += 1
end

####### cross_val_score
N = 50.0
X = [collect(1:N) collect(1:N)]
y = map(reshape(X[:, 1], size(X)[1])) do x
    if x < 0.3*N
        0
    elseif x > 0.6*N
        2
    else
        1
    end
end
scores = cross_val_score!(SVC(), X, y; cv=y->stratified_kfold(y; k=10))
@test_approx_eq scores["0"] 1/1.2
@test_approx_eq scores["1"] 1/1.2
@test_approx_eq scores["2"] 0.9266666666666667 

####### GridSearchCV
N = 50.0
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
# XXX Why is julia not inferring the parametric type here?
#gs = GridSearchCV(SVC(), params; scoring=f1_score, cv=y->stratified_kfold(y; k=10))
gs = GridSearchCV{SVC}(SVC(), params; scoring=f1_score, cv=y->stratified_kfold(y; k=10))
fit!(gs, X, y)
@test gs.best_score > 0.9

####### Preprocessing #######
N = 50.0
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

mms = MinMaxScaler()
X_mms = fit_transform(mms, X)
@test all(minimum(X_mms, 1) .== 0.0)
@test all(maximum(X_mms, 1) .== 1.0)
@test X != X_mms
mms = MinMaxScaler(10.0, 50.0)
X_mms = fit_transform(mms, X)
@test all(minimum(X_mms, 1) .== 10.0)
@test all(maximum(X_mms, 1) .== 50.0)
@test X != X_mms

####### Pipeline #######
N = 50.0
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

pipe = Pipeline([("mms", MinMaxScaler()), ("ss", StandardScaler())], ("svc", SVC()))
fit!(pipe, X, y)
y_pred = predict(pipe, X)
@test_approx_eq f1_score(y, y_pred)["1"] 1.0
@test f1_score(y, y_pred)["0"] == 1.0

params = Dict{ASCIIString, Vector}("mms__range_min"=>[0.0], "mms__range_max"=>[1.0], "svc__C"=>[0.01, 0.1, 1., 10., 100., 1000., 10000., 100000., 1000000.], "svc__kernel"=>["rbf", "linear", "polynomial", "sigmoid"])
# XXX Why doesn't type infernce work here?
#GridSearchCV(pipe, params)
gs = GridSearchCV{Pipeline}(pipe, params)
fit!(gs, X, y)
@test gs.best_score == 1.0
@test gs.best_params[:mms__range_min] == 0.0
@test gs.best_params[:mms__range_max] == 1.0
@test gs.best_params[:svc__kernel] == "linear"
@test gs.best_params[:svc__C] == 0.1
@test gs.best_estimator.preprocessors[1][1] == "mms"
@test gs.best_estimator.preprocessors[1][2].range_min == 0.0
@test gs.best_estimator.preprocessors[1][2].range_max == 1.0
@test gs.best_estimator.preprocessors[2][1] == "ss"
@test gs.best_estimator.estimator[1] == "svc"
@test gs.best_estimator.estimator[2].kernel == "linear"
@test gs.best_estimator.estimator[2].C == 0.1


