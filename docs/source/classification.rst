Classification
==============

Learn.jl provides SVC, logistic regression, random forest, decision tree, Gaussian naive Bayes, linear discriminant analysis for classification. The functions described below are the same for all classifiers. Examples with specific parameters can be found below.

Functions
---------

.. function:: fit!(clf::Classifier, X::Matrix{Float64}, y::Vector)

    Fit the classifier ``clf`` to the inputs ``X`` and the targets ``y``. 

    :param clf: The classifer object encapsulating parameters for the estimator. This parameter will be modified by the function. After running ``fit!`` ``clf`` will contain all information required to make predictions.
    :param X: X assumes rows for observations and columns as features. 
    :param y: The targets, which can be of any type.

.. function:: predict{T<:AbstractFloat}(clf::Classifier, X::Matrix{T})
    
    Predict values for ``X`` using the fitted estimator ``clf``.

    :param clf: The classifier after fitting with ``fit!``. 
    :param X: Input data with observations in rows and features in columns.

.. function:: score{T<:AbstractFloat}(clf::Classifier, X::Matrix{T}, y_true::Vector; scoring::Union{Function, Void}=f1_score)
    
    Return the score for a set of input observations ``X`` and targets ``y`` with scoring function ``scoring``.

    :param estimator: The estimator trained by ``fit``.
    :param X: The input values with observations as rows and features as columns.
    :param y_true: The true targets to compare predictions with.
    :param scoring: Optional scoring function. By default the F1 score is used for classification.

Multiclass Strategies
---------------------

Classifier that discriminate beween only two classes can still be used for multiclass classification. Typically one uses either the one-vs-one_ or the one-vs-all_ strategies. Both have their downsides. Learn.jl implements both. All classifiers accept a strategy parameter that allows to specify the multiclass strategy. The one-vs-one strategy will be used by default. To specify the strategy provide either ``OneVsOneStrategy`` or ``OneVsAllStrategy`` as keyword argument to the classifier's constructor like this:

    .. code-block:: julia
        
        clf = LogisticRegression(strategy=OneVsAllStrategy())

.. _one-vs-one: https://en.wikipedia.org/wiki/Multiclass_classification 
.. _one-vs-all: https://en.wikipedia.org/wiki/Multiclass_classification 

Logistic regression
-------------------

This code uses GLM.JL_.

.. _GLM.jl: https://github.com/JuliaStats/GLM.jl 

.. function:: LogisticRegression(; strategy::MulticlassStrategy=OneVsOneStrategy())
    
    .. code-block:: julia

        clf = LogisticRegression()
        fit!(clf, X, y)
        y_pred = predict(clf, X)
        score(y, X)

Random forest classification
----------------------------

Implementation of the random forest algorithm for classification. This code uses DecisionTree.jl_.

.. _DecisionTree.jl: https://github.com/bensadeghi/DecisionTree.jl 

.. function:: RandomForestClassifier(;nsubfeatures::Integer=2, ntrees::Integer=5, strategy::MulticlassStrategy=OneVsOneStrategy())
    
    For detailed information about the parameters refer to the documentation of DecisionTree.jl_.

    .. code-block:: julia

        clf = RandomForestClassifier()
        fit!(clf, X, y)
        y_pred = predict(clf, X)
        score(y, X)

Decision tree classification
----------------------------

Implementation of the decison tree algorithm for classification. This code uses DecisionTree.jl_.

.. function:: DecisionTreeClassifier(;strategy::MulticlassStrategy=OneVsOneStrategy())
    
    .. code-block:: julia

        clf = DecisionTreeClassifier()
        fit!(clf, X, y)
        y_pred = predict(clf, X)
        score(y, X)

Gaussian naive Bayes
--------------------

Implementation of the Gaussian naive Bayes algorithm. This code uses NaiveBayes.jl_.

.. _NaiveBayes.jl: https://github.com/johnmyleswhite/NaiveBayes.jl

.. function:: GaussianNB(;strategy::MulticlassStrategy=OneVsOneStrategy())
    
    .. code-block:: julia

        clf = GaussianNB()
        fit!(clf, X, y)
        y_pred = predict(clf, X)
        score(y, X)

Linear discriminant analysis
----------------------------

Implementation of the linear discrimant algorithm. This code uses MultivariateStats.jl_.

.. _MultivariateStats.jl: https://github.com/JuliaStats/MultivariateStats.jl 

.. function:: LinearDiscriminantAnalysis(;strategy::MulticlassStrategy=OneVsOneStrategy())
    
    .. code-block:: julia

        clf = LinearDiscriminantAnalysis()
        fit!(clf, X, y)
        y_pred = predict(clf, X)
        score(y, X)

