Clustering
==========

Learn.jl currently provides only kmeans clustering.

Functions
---------

.. function:: fit!{T<:AbstractFloat}(clust::Cluster, X::Matrix{T})

    Fit the estimator ``clust`` to the inputs ``X``. 

    :param clust: The estimator object encapsulating parameters for the estimator. This parameter will be modified by the function. After running ``fit!`` ``clust`` will contain all information required to make predictions.
    :param X: X assumes rows for observations and columns as features. 

.. function:: predict{T<:AbstractFloat}(clust::Cluster, X::Matrix{T})
    
    Predict values for ``X`` using the fitted estimator ``clust``.

    :param clust: The estimator after fitting with ``fit!``. 
    :param X: Input data with observations in rows and features in columns.

.. function:: score{T<:AbstractFloat}(clust::Cluster, X::Matrix{T}; scoring::Union{Function, Void}=nothing)
    
    Return the score for a set of input observations ``X`` with scoring function ``scoring``.

    :param clust: The estimator trained by ``fit``.
    :param X: The input values with observations as rows and features as columns.
    :param scoring: Optional scoring function. By default for K-means 1/totalcosts is used.

K-means
-------

Implementation of the K-means algorithm. This code uses Clustering.jl_, which in turn wraps LIBSVM.

.. _Clustering.jl: https://github.com/JuliaStats/Clustering.jl 

.. function:: Kmeans(;n_clusters::Int=8, max_iter::Int=300, n_init::Int=10, init::Symbol=:kmpp, tol::Float64=1e-4)
    
    For detailed information about the parameters refer to the documentation of Clustering.jl_.

    .. code-block:: julia

        clust = Kmeans(n_clusters=3)
        fit!(clust, X)
        predict(clust, X)
        score(clust, X_new)


