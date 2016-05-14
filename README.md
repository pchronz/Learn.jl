Learn.jl
========

[![Build Status](https://travis-ci.org/pchronz/Learn.jl.svg?branch=master)](https://travis-ci.org/pchronz/Learn.jl)

A unified package for machine learning in [Julia](http://julialang.org/).

This package is motivated by [sklearn](http://scikit-learn.org), available to python users. If you're familiar with sklearn, you can gauge Learn.jl's similarity to sklearn and its power from the following snippet.

```julia
pipe = Pipeline([("mms", MinMaxScaler()), ("ss", StandardScaler())], ("svc", SVC())) 
params = Dict{ASCIIString, Vector}("mms__range_min"=>[0.0], "mms__range_max"=>[1.0], "svc__C"=>[0.01, 0.1, 1., 10., 100., 1000., 10000., 100000., 1000000.], "svc__kernel"=>["rbf", "linear", "polynomial"    , "sigmoid"])
gs = GridSearchCV{Pipeline}(pipe, params)
fit!(gs, X, y)
@show gs.best_score
@show gs.best_params
@show gs.best_estimator.preprocessors
@show gs.best_estimator.estimator
```

# Docs
[Here](http://learnjl.readthedocs.io/en/latest/)

# Current features
## Regression
* SVR
* Linear regression
* Decision tree regression
* Random forest regression

## Classification
* Multiclass strategies
 * one-vs-one
 * one-vs-all
* Logistic regression
* Random forest classification
* Decision tree classification
* Gaussian naive Bayes
* Linear Discriminant analysis

## Clustering
* K-means

## Scoring
* Regression
 * R2 score
 * Mean squared error
 * Explained covariance score
* Classification
 * Precision
 * Recall
 * F1

## Preprocessing
* Min-Max scaler
* Standard scaler
* PCA
* FastICA
 
## Pipeline
* Regression
* Classification
* Clustering

## Cross-validation
* k-fold
* stratified k-fold
* shuffe-split
* GridSearchCV
 * Regression
 * Classification
 * Clustering
 * Pipelines with preprocesseors, and regression, classification or clustering

