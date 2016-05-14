Regression
==========

Learn.jl provides SVR, linear regression, decision tree regression, and random forest regression. The functions used are all the same, just the parameters change.

Functions
---------

.. function:: fit!{T<:AbstractFloat}(reg::Regressor, X::Matrix{T}, y::Vector{T})

    Fit the regressor reg to the inputs X and the targets y. 

    :param reg: The regressor object carrying parameters. This parameter will be modified by the function. After running ``fit!`` ``reg`` will contain all information required to make predictions.
    :param X: X assumes rows for observations and columns as features. 
    :param y: The targets.

.. function:: predict{T<:AbstractFloat}(reg::Regressor, X::Matrix{T})
    
    Predict values for ``X`` using the fitted estimator ``reg``.

    :param reg: The regressor after fitting with ``fit!``. 
    :param X: Input data with observations in rows and features in columns.

.. function:: function score{T<:AbstractFloat}(estimator::Regressor, X::Matrix{T}, y_true::Vector; scoring::Union{Function, Void}=r2_score)
    
    Return the score for a set of input observations ``X`` and targets ``y`` with scoring function ``scoring``.

    :param estimator: The estimator trained by ``fit``.
    :param X: The input values with observations as rows and features as columns.
    :param y_true: The true targets to compare predictions with.
    :param scoring: Optional scoring function. By default the R2 score is used for regression.

SVR
---

Implementation of the eps-SVR algorithm. This code uses ``LIBSVM.jl``, which in turn wraps LIBSVM.

.. function:: SVR(;kernel="rbf", degree=3, gamma=nothing, coef0=0.0, C=1.0, nu=0.5, p=0.1, cache_size=100.0, eps=0.001, shrinking=true, probability_estimates=false, weights=nothing, verbose=false)
    
    For detailed information about the parameters refer to the documentation of LIBSVM.jl_ and LIBSVM_ .

    .. _LIBSVM.jl: https://github.com/simonster/LIBSVM.jl 
    .. _LIBSVM: https://www.csie.ntu.edu.tw/~cjlin/libsvm/ 

    .. code-block:: julia
        
        reg = SVR(C=100.0)
        fit!(reg, X, y)
        y_pred = predict(reg, X)
        r2_score(y, y_pred)
        assert(r2_score(y, y_pred) == score(reg, X, y))

Linear regression
-----------------

Linear regression, using GLM.jl_. 

.. function:: LinearRegression(;fit_intercept::Bool=true, normalize::Bool=false)

    .. _GLM.jl: https://github.com/JuliaStats/GLM.jl 

    .. code-block:: julia
        
        reg = LinearRegression()
        fit!(reg, X, y)
        y_pred = predict(reg, X)
        assert(r2_score(y, y_pred) == score(reg, X, y))

Decision tree regression
------------------------

Decision tree regression, using DecisionTree.jl_. Currently, it seems that the algorithm breaks if the input data has only one feature. The issue is reported.

.. function:: DecisionTreeRegressor()

    .. _DecisionTree.jl: https://github.com/bensadeghi/DecisionTree.jl

    .. code-block:: julia

        reg = DecisionTreeRegressor()
        fit!(reg, X, y)
        y_pred = predict(reg, X)
        assert(r2_score(y, y_pred) == score(reg, X, y))

Random forest regression
------------------------

Random forest regression using DecisionTree.jl_. Currently, it seems that the algorithm breaks if the input data has only one feature. The issue is reported.

.. function:: RandomForestRegressor(;nsubfeatures::Integer=2, ntrees::Integer=5)

    .. code-block:: julia
        
        reg = RandomForestRegressor()
        fit!(reg, X, y)
        y_pred = predict(reg, X)
        assert(r2_score(y, y_pred) == score(reg, X, y))


