Scoring
=======

Learn.jl provides scoring functions for regression, classification, and clustering. For regression ``r2_score``, ``mean_squared_error``, and ``explained_variance_score`` are available. For classification ``f1_score``, ``precision_score``, and ``recall_score`` are available. For clustering 1/totalcosts is available for K-means, as the only integrated clustering algorithms. There is a ``score`` function for each type of estimator (Regressor, Classifier, and Cluster), which takes a set of observations ``X`` and  the true targets ``y`` for regression and classifcation. 

.. function:: score{T<:AbstractFloat}(clf::Estimator, X::Matrix{T}, y_true::Vector; scoring::Union{Function, Void})

.. function:: score{T<:AbstractFloat}(clust::Cluster, X::Matrix{T}; scoring::Union{Function, Void})

These functions delegate either to ``scoring`` or, if scoring is not provided, to a default scoring function. The default for regression is ``r2_score`` and for classification ``f1_score``. 

The scoring functions below take a set of predicted targets and the observed targets. Thus, ``score`` is a convenience function that combines ``predict`` with one of the scoring functions below.

Regression
----------

.. function:: r2_score(y_true::Vector{Float64}, y_pred::Vector{Float64})

    Compute the R2-score_ for observed targets ``y_true`` and predicted targets ``y_pred``.

.. _R2-score: https://en.wikipedia.org/wiki/Coefficient_of_determination


.. function:: mean_squared_error(y_true::Vector{Float64}, y_pred::Vector{Float64})

    Compute the mean squared error for observed targets ``y_true`` and predicted targets ``y_pred``.

.. function:: explained_variance_score(y_true::Vector{Float64}, y_pred::Vector{Float64})

    Compute the explained-variance-score_ for observed targets ``y_true`` and predicted targets ``y_pred``.

.. _explained_variance_score: https://en.wikipedia.org/wiki/Explained_variation 


Classification
--------------

The scoring functions for classification return a dictionary with the labels as keys and the scores as values.

.. function:: precision_score(y_observed, y_pred; ave_fun::Union{ASCIIString, Void}=nothing) 

    Compute the precision-score_ for observed targets ``y_true`` and predicted targets ``y_pred``.

.. _precision-score: https://en.wikipedia.org/wiki/Precision_and_recall 

.. function:: recall_score(y_observed, y_pred; ave_fun::Union{ASCIIString, Void}=nothing) 

    Compute the recall-score_ for observed targets ``y_true`` and predicted targets ``y_pred``.

.. _recall-score: https://en.wikipedia.org/wiki/Precision_and_recall 

.. function:: f1_score(y_observed, y_pred; ave_fun::Union{ASCIIString, Void}=nothing) 

    Compute the f1-score_ for observed targets ``y_true`` and predicted targets ``y_pred``.

.. _f1-score: https://en.wikipedia.org/wiki/F1_score 


