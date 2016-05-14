Preprocessing
=============

Learn.jl provides ``MinMaxScaler``, ``StandardScaler``, ``PCA``, and ``FastICA`` for preprocessing. Currently the only function for preprocessing is ``fit_transform!``. Its use is the same for all types of preprocessors.

Function
---------

.. function:: fit_transform!{T<:AbstractFloat}(pre::Preprocessor, X::Matrix{T})

    Fit the preprocessor ``pre`` to the inputs ``X`` and transform ``X``. 

    :param pre: The preprocessor object encapsulating parameters for the estimator. This parameter will be modified by the function. 
    :param X: X assumes rows for observations and columns as features. 

MinMaxScaler
------------

The min-max-scaler_ scales all features independently to a given range, which is [0, 1] by default.

.. _min-max-scaler: http://sebastianraschka.com/Articles/2014_about_feature_scaling.html#about-min-max-scaling 
.. function:: MinMaxScaler() = MinMaxScaler(0.0, 1.0)
    
    .. code-block:: julia

        mms = MinMaxScaler(10.0, 50.0)
        X_mms = fit_transform!(mms, X)
        @test all(minimum(X_mms, 1) .== 10.0)
        @test all(maximum(X_mms, 1) .== 50.0)
        @test X != X_mms

StandardScaler
--------------

The standard-scaler_, also called z-score normalization, scales all features independently to zero mean and unit variance. 

.. _standard-scaler: http://sebastianraschka.com/Articles/2014_about_feature_scaling.html#about-standardization 
    
    .. code-block:: julia

        ss = StandardScaler()
        X_ss = fit_transform!(ss, X)

PCA
---

PCA wraps the implementation in MultivariateStats_.

.. _MultivariateStats: https://github.com/JuliaStats/MultivariateStats.jl 

.. function:: PCA(;n_components::Union{Void, Int}=nothing)
    
    .. code-block:: julia

        pca = PCA()
        X_pca = fit_transform!(pca, X)

FastICA
-------

FastICA wraps the implementation in MultivariateStats_.

.. function:: FastICA(;n_components::Noint=nothing, whiten::Bool=false, max_iter::Noint=nothing, tol::Nofloat=nothing)
    
    .. code-block:: julia

        ica = FastICA(;n_components=2, whiten=true)
        X_ica = fit_transform!(ica, X)

