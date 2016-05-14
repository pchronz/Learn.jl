Cross-validation
================

Cross-validation in **Learn** provides various strategies for cross-validation, an integrated scoring functioncalled ``cross_val_score!``, and ``GridSearchCV``. The strategies comprise ``kfold``, ``stratified_kfold``, and ``shuffle_split``. The function ``cross_val_score!`` combines ``fit!``, ``predict``, and ``score`` with cross-validation to obtain a robust score a set of parameters. ``GridSearchCV`` finally allows to search for the parameter set, from a combination of given parameters, using cross-validation.

K-fold
---------------------

K-fold cross validation. The function ``kfold`` returns an Iterable that provides sets of training and test indicies.

.. function:: kfold(n_observations::Int; k::Integer=10)
    kfold(y::Vector; k::Integer=10) = kfold(length(y); k=k)
    kfold(X::Matrix; k::Integer=10) = kfold(size(X, 1); k=k)

    .. code-block:: julia
        
        kf = kfold(y; k=3)
        for (idx_tr, idx_test) in kf
            X_tr = X[idx_tr, :]
            y_tr = y[idx_tr, :]
            X_test = X[idx_test, :]
            y_test = y[idx_test, :]
            svc = SVC()
            fit!(svc, X_tr, y_tr)
            score(svc, X_test, y_test)
        end

Stratified K-fold
---------------------

Stratified version of K-fold cross validation. The function ``stratified_kfold`` returns an Iterable that provides sets of training and test indicies.

.. function:: stratified_kfold(y::Vector{Int}; k::Integer=2)

    .. code-block:: julia
        
        skf = stratified_kfold(y; k=3)
        for (idx_tr, idx_test) in skf
            X_tr = X[idx_tr, :]
            y_tr = y[idx_tr, :]
            X_test = X[idx_test, :]
            y_test = y[idx_test, :]
            svc = SVC()
            fit!(svc, X_tr, y_tr)
            score(svc, X_test, y_test)
        end

Shuffle-Split
---------------------

Choose values at random without repitition. The function ``shufflesplit`` returns an Iterable that provides sets of training and test indicies.

.. function:: shufflesplit(n_observations::Int; k::Integer=10)
    shufflesplit(y::Vector; k::Integer=10) = shufflesplit(length(y); k=k)
    shufflesplit(X::Matrix; k::Integer=10) = shufflesplit(size(X, 1); k=k)

    .. code-block:: julia
        
        ss = shuffle_split(X; k=3)
        for (idx_tr, idx_test) in ss
            X_tr = X[idx_tr, :]
            X_test = X[idx_test, :]
            km = Kmeans()
            fit!(km, X_tr)
            score(km, X_test)
        end

cross_val_score!
---------------------

Run fit, predict, and score on different folds, provided via a cross validation strategy. Then compute the mean of the scores across all folds. Works for regression, classification, clustering, and pipelines using those estimators.

.. function:: cross_val_score!{T<:Classifier}(estimator::Union{T, Pipeline{T}}, X::Matrix{Float64}, y::Vector; cv::Function=stratified_kfold, scoring::Union{Function, Void}=nothing)
    cross_val_score!{T<:AbstractFloat, S<:Regressor}(estimator::Union{S, Pipeline{S}}, X::Matrix{T}, y::Vector; cv::Function=kfold, scoring::Union{Function, Void}=nothing)
    cross_val_score!{T<:AbstractFloat, S<:Cluster}(estimator::Union{S, Pipeline{S}}, X::Matrix{T}; cv::Function=kfold, scoring::Union{Function, Void}=nothing)

    .. code-block: julia
        scores = cross_val_score!(SVC(), X, y; cv=y->stratified_kfold(y; k=10))

GridSearchCV
---------------------

Run ``cross_val_score!`` on different sets of parameters and return the best estimator, its parameters, and its score. ``GridSearchCV`` works on individual estimators as well as on ``Pipeline`` objects. Running a grid search with cross-validation on a pipeline is probably this package's most advanced features. 


.. function:: GridSearchCV(estimator::T, param_grid::Dict{ASCIIString, Vector}; scoring=nothing, cv=stratified_kfold)
    GridSearchCV{S<:Cluster}(estimator::Union{S, Pipeline{S}}, param_grid::Dict{ASCIIString, Vector}; scoring=nothing, cv=kfold)
    GridSearchCV{S<:Regressor}(estimator::Union{S, Pipeline{S}}, param_grid::Dict{ASCIIString, Vector}; scoring=nothing, cv=shufflesplit)
    
    Create a new instance for a grid search. For individual estimators the parameters are provided as a dictionary, with the parameter name as key and the paramter values as a list. For pipelines you need to combine the names of the pipeline stage and the parameter. See the examples below to understand how to prepare the parameters for individual estimators and for pipelines. 
    
    .. code-block:: julia

        params = Dict{ASCIIString, Vector}("C"=>[0.01, 0.1, 1., 10., 100., 1000., 10000., 100000., 1000000.], "kernel"=>["rbf", "linear", "polynomial", "sigmoid"])
        gs = GridSearchCV{SVC}(SVC(), params; scoring=f1_score, cv=y->stratified_kfold(y; k=10))
        fit!(gs, X, y)
        @show gs.best_score
        @show gs.best_estimator
        @show gs.best_params
        
    .. code-block:: julia
        
        pipe = Pipeline([("mms", MinMaxScaler()), ("ss", StandardScaler())], ("svc", SVC()))
        params = Dict{ASCIIString, Vector}("mms__range_min"=>[0.0], "mms__range_max"=>[1.0], "svc__C"=>[0.01, 0.1, 1., 10., 100., 1000., 10000., 100000., 1000000.], "svc__kernel"=>["rbf", "linear", "polynomial", "sigmoid"])
        gs = GridSearchCV{Pipeline}(pipe, params)
        fit!(gs, X, y)
        @show gs.best_score
        @show gs.best_params
        @show gs.best_estimator.preprocessors
        @show gs.best_estimator.estimator

