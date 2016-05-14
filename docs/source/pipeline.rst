Pipeline
========

``Pipeline`` allows to combine multiple preprocessors and one estimator, such as a classifier, regressor, or clustering algorithm into one estimator. ``Pipeline`` comes in especially handy in combination with cross validation, especially ``GridSearchCV``.

Creation
---------

A ``Pipeline`` is created by providing all of the necessary preprocessors and an estimator.

.. function:: Pipeline{T<:Estimator}(prepros::Vector{Tuple{ASCIIString, Preprocessor}}, est::Tuple{ASCIIString, T})
    
    Create a new pipeline object. The preprocessors are provided as a list of tuples. Each tuple contains the name of the preprocessor and the preprocessors object itself. 

    .. code-block:: julia

        pipe = Pipeline([("mms", MinMaxScaler()), ("ss", StandardScaler())], ("svc", SVC()))


Functions
---------

As the ``Pipeline`` is an ``Estimator`` the common functions for estimators all work on ``Pipeline`` as well.

.. function:: fit!(pipe::Pipeline, X::Matrix{Float64}, y::Vector)

    Fit the pipeline for regression or classification.

.. function:: fit!{T<:Cluster}(pipe::Pipeline{T}, X::Matrix{Float64})

    Fit the pipeline with an unsupervised estimator, such as a ``Kmeans``.

.. function:: predict(pipe::Pipeline, X::Matrix{Float64})

    Preprocess the data and predict values using the estimator's ``predict`` method.

.. function:: score{T<:AbstractFloat, S<:Union{Classifier, Regressor}}(pipe::Pipeline{S}, X::Matrix{T}, y_true::Vector; scoring::Union{Function, Void}=nothing)
    
    Scoring for classification and regression.

.. function:: score{T<:AbstractFloat, S<:Cluster}(pipe::Pipeline{S}, X::Matrix{T}; scoring::Union{Function, Void}=nothing)

    Scoring for unsupservised learning.

Example
-------

Create a pipeline with a ``MinMaxScaler``, followed by a ``StandardScaler``, and finally an ``SVC``.

    .. code-block:: julia

        pipe = Pipeline([("mms", MinMaxScaler()), ("ss", StandardScaler())], ("svc", SVC()))
        fit!(pipe, X, y)
        y_pred = predict(pipe, X)
        @test_approx_eq f1_score(y, y_pred)["1"] 1.0
        @test f1_score(y, y_pred)["0"] == 1.0

