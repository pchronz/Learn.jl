# TODO Document
# TODO run in parallel wherever possible or advantageous
# TODO keyword arguments instead of positional ones
module jlearn
################ SVM ###############
import LIBSVM
import MultivariateStats
import GLM
import DecisionTree
####### Aliases
typealias Noint Union{Void, Int}
typealias Nofloat Union{Void, Float64}

# TODO CSVC
# TODO NuSVC
# TODO OneClassSVM
# TODO NuSVR
abstract Estimator
abstract Classifier <: Estimator
abstract Regressor <: Estimator
type SVMWrapper
    kernel::ASCIIString
    degree::Integer
    gamma::Float64
    coef0::Float64
    C::Float64
    nu::Float64
    p::Float64
    cache_size::Float64
    eps::Float64
    shrinking::Bool
    probability_estimates::Bool
    weights
    verbose::Bool
    libsvm::LIBSVM.SVMModel
    SVMWrapper(kernel, degree, gamma, coef0, C, nu, p, cache_size, eps, shrinking, probability_estimates, weights, verbose) = new(kernel, degree, gamma, coef0, C, nu, p, cache_size, eps, shrinking, probability_estimates, weights, verbose)
end
################ SVC
type SVC <: Classifier
    svm::SVMWrapper
    SVC(;kernel="rbf", degree=3, gamma=0.0, coef0=0.0, C=1.0, nu=0.5, p=0.1, cache_size=100.0, eps=0.001, shrinking=true, probability_estimates=false, weights=nothing, verbose=false) = new(SVMWrapper(kernel, degree, gamma, coef0, C, nu, p, cache_size, eps, shrinking, probability_estimates, weights, verbose))
end
function fit!(svm::SVMWrapper, svm_type::Int32, X::Matrix{Float64}, y::Vector)
    function get_kernel(k_str::ASCIIString)
        if k_str == "rbf"
            LIBSVM.RBF
        elseif k_str == "linear"
            LIBSVM.Linear
        elseif k_str == "polynomial"
            LIBSVM.Polynomial
        elseif k_str == "sigmoid"
            LIBSVM.Sigmoid
        else
            error("Unknown kernel \"$(k_str)\"")
        end
    end
    kernel = get_kernel(svm.kernel)
    svm.gamma = 1/size(X, 1)
    # XXX Use dictionary to store the params in SVM and splice in here.
    libsvm = LIBSVM.svmtrain(y, X', 
        svm_type=svm_type,
        kernel_type=kernel,
        degree=svm.degree,
        gamma=svm.gamma,
        coef0=svm.coef0, 
        C=svm.C,
        nu=svm.nu,
        p=svm.p,
        cache_size=svm.cache_size,
        eps=svm.eps,
        shrinking=svm.shrinking,
        probability_estimates=svm.probability_estimates,
        weights=svm.weights,
        verbose=svm.verbose)
    svm.libsvm = libsvm
end
fit!(clf::SVC, X::Matrix{Float64}, y::Vector) = fit!(clf.svm, Int32(0), X, y)
predict(clf::SVC, X::Matrix{Float64}) = LIBSVM.svmpredict(clf.svm.libsvm, X')[1]

################ SVR
type SVR <: Regressor
    svm::SVMWrapper
    SVR(;kernel="rbf", degree=3, gamma=0.0, coef0=0.0, C=1.0, nu=0.5, p=0.1, cache_size=100.0, eps=0.001, shrinking=true, probability_estimates=false, weights=nothing, verbose=false) = new(SVMWrapper(kernel, degree, gamma, coef0, C, nu, p, cache_size, eps, shrinking, probability_estimates, weights, verbose))
end
fit!{T<:Number}(reg::SVR, X::Matrix{Float64}, y::Vector{T}) = fit!(reg.svm, Int32(3), X, y)
predict(reg, X::Matrix{Float64}) = LIBSVM.svmpredict(reg.svm.libsvm, X')[1]

####### Generalized linear models #######
type LinearRegression <: Regressor
    fit_intercept::Bool
    normalize::Bool
    coef_::Vector{Float64}
    intercept_::Vector{Float64}
    lm::GLM.LinearModel
    LinearRegression(;fit_intercept::Bool=true, normalize::Bool=false) = new(fit_intercept, normalize)
end
function fit!{T<:Number}(reg::LinearRegression, X::Matrix{Float64}, y::Vector{T})
    reg.lm = GLM.lm(X, y)
end
predict(reg::LinearRegression, X::Matrix{Float64}) = GLM.predict(reg.lm, X)
function score{T<:Number}(reg::LinearRegression, X::Matrix{Float64}, y::Vector{T})
    y_pred = predict(reg, X)
    r2_score(y, y_pred)
end

####### Ensemble methods #######
####### RandomForestRegressor
type RandomForestRegressor <: Regressor
    nsubfeatures::Integer
    ntrees::Integer
    forest::DecisionTree.Ensemble
    RandomForestRegressor(;nsubfeatures::Integer=2, ntrees::Integer=5) = new(nsubfeatures, ntrees)
end
function fit!{T<:Number}(reg::RandomForestRegressor, X::Matrix{Float64}, y::Vector{T}) 
    reg.forest = DecisionTree.build_forest(y, X, reg.nsubfeatures, reg.ntrees)
end
predict(reg::RandomForestRegressor, X::Matrix{Float64}) = DecisionTree.apply_forest(reg.forest, X)
function score{T<:Number}(reg::RandomForestRegressor, X::Matrix{Float64}, y::Vector{T})
    y_pred = predict(reg, X)
    r2_score(y, y_pred)
end
type RandomForestClassifier <: Classifier
    nsubfeatures::Integer
    ntrees::Integer
    forest::DecisionTree.Ensemble
    RandomForestClassifier(;nsubfeatures::Integer=2, ntrees::Integer=5) = new(nsubfeatures, ntrees)
end
function fit!{T}(reg::RandomForestClassifier, X::Matrix{Float64}, y::Vector{T}) 
    reg.forest = DecisionTree.build_forest(y, X, reg.nsubfeatures, reg.ntrees)
end
predict(reg::RandomForestClassifier, X::Matrix{Float64}) = DecisionTree.apply_forest(reg.forest, X)

####### DecisionTreeRegressor
type DecisionTreeRegressor <: Regressor
    tree::DecisionTree.Node
    DecisionTreeRegressor() = new()
end
function fit!{T<:Number}(reg::DecisionTreeRegressor, X::Matrix{Float64}, y::Vector{T}) 
    reg.tree = DecisionTree.build_tree(y, X)
end
predict(reg::DecisionTreeRegressor, X::Matrix{Float64}) = DecisionTree.apply_tree(reg.tree, X)
function score{T<:Number}(reg::DecisionTreeRegressor, X::Matrix{Float64}, y::Vector{T})
    y_pred = predict(reg, X)
    r2_score(y, y_pred)
end

################ Metrics ###############
####### Classifier scores
get_tp(y_observed, y_pred, label) = sum((y_observed .== label) & (y_pred .== label))
get_fp(y_observed, y_pred, label) = sum(!(y_observed .== label) & (y_pred .== label))
get_fn(y_observed, y_pred, label) = sum((y_observed .== label) & !(y_pred .== label))

# TODO "Micro" (prec, recall, f1)
# TODO "Weighted" (prec, recall, f1)
# TODO "Samples" (prec, recall, f1)
fun_score(fun::Function, y_observed::Vector, y_pred::Vector, pos_label::Any) = remove_nans(fun(y_observed, y_pred, pos_label))
function fun_score(fun::Function, y_observed::Vector, y_pred::Vector; ave_fun::Union{ASCIIString, Void}=nothing)
    labels = unique([unique(y_observed); unique(y_pred)])
    scors = Dict{ASCIIString, Float64}()
    for (i, label) in enumerate(labels)
        scors[string(label)] = fun(y_observed, y_pred, label)
    end
    if ave_fun == nothing
        return remove_nans(scors)
    elseif ave_fun == "macro"
        scor_mean = 0
        for (label, scor) in scors
            scor_mean += scor
        end
        return remove_nans(scor_mean/length(scors))
    end
end

precision_score(y_observed, y_pred; ave_fun::Union{ASCIIString, Void}=nothing) = fun_score(precision_score, y_observed, y_pred; ave_fun=ave_fun)
precision_score(y_observed, y_pred, pos_label) = fun_score(precision_score, y_observed, y_pred, pos_label)
recall_score(y_observed, y_pred; ave_fun::Union{ASCIIString, Void}=nothing) = fun_score(recall_score, y_observed, y_pred; ave_fun=ave_fun)
recall_score(y_observed, y_pred, pos_label) = fun_score(recall_score, y_observed, y_pred, pos_label)
f1_score(y_observed, y_pred; ave_fun::Union{ASCIIString, Void}=nothing) = fun_score(f1_score, y_observed, y_pred; ave_fun=ave_fun)
f1_score(y_observed, y_pred, pos_label) = fun_score(f1_score, y_observed, y_pred, pos_label)

remove_nans(scor::Float64) = isnan(scor) ? 0.0 : scor
function remove_nans(scors::Dict{ASCIIString, Float64}) 
    for scor in scors 
        if isnan(scor[2])
            scors[scor[1]] = 0.0
        end
    end
    scors
end

function precision_score(y_observed, y_pred, pos_label)
    tp = get_tp(y_observed, y_pred, pos_label)
    fp = get_fp(y_observed, y_pred, pos_label)
    if tp + fp == 0.0
        warn("TP + FP == 0 for precision for label $(pos_label). Setting precision to 0.0")
        return 0.0
    else
        tp/(tp + fp)
    end
end

function recall_score(y_observed, y_pred, pos_label)
    tp = get_tp(y_observed, y_pred, pos_label)
    fn = get_fn(y_observed, y_pred, pos_label)
    if tp + fn == 0.0
        warn("TP + FN == 0 for recall for label $(pos_label). Setting recall to 0.0")
        return 0.0
    else
        tp/(tp + fn)
    end
end

function f1_score(y_observed, y_pred, pos_label)
    prec = precision_score(y_observed, y_pred, pos_label)
    recall = recall_score(y_observed, y_pred, pos_label)
    if prec + recall == 0.0
        warn("Precision and recall == 0.0 for F1 for label $(pos_label). Setting F1 to 0.0")
    end
    2*prec*recall/(prec + recall)
end

####### Regression scores
r2_score(y_true::Vector{Float64}, y_pred::Vector{Float64}) = 1. - sum((y_true - y_pred).^2)/(sum((y_true .- mean(y_true)).^2))
mean_squared_error(y_true::Vector{Float64}, y_pred::Vector{Float64}) = mean((y_true - y_pred).^2)
explained_variance_score(y_true::Vector{Float64}, y_pred::Vector{Float64}) = (var(y_true) - var(y_true - y_pred))/var(y_true)

################ Pre-processing ###############
abstract Preprocessor

####### MinMaxScaler
type MinMaxScaler <: Preprocessor
    range_min::Float64
    range_max::Float64
end
MinMaxScaler() = MinMaxScaler(0.0, 1.0)
function fit_transform!(mms::MinMaxScaler, X::Matrix{Float64})
    X = copy(X)
    for col = 1:size(X, 2)
        mn, mx = extrema(X[:, col])
        X[:, col] -= mn
        X[:, col] /= (mx - mn)
        X[:, col] *= (mms.range_max - mms.range_min)
        X[:, col] += mms.range_min
    end
    X
end

####### StandardScaler
type StandardScaler <: Preprocessor
end
function fit_transform!(ss::StandardScaler, X::Matrix{Float64})
    X = copy(X)
    X = X .- mean(X, 1) 
    X = X ./ sqrt(var(X, 1))
    X
end

####### PCA
type PCA <: Preprocessor
    n_components::Union{Void, Int}
    M::MultivariateStats.PCA
    explained_variance_ratio_::Vector{Float64}
    PCA(n_components::Union{Void, Int}) = new(n_components)
end
PCA(;n_components::Union{Void, Int}=nothing) = PCA(n_components)
function fit_transform!(pca::PCA, X::Matrix{Float64})
    if pca.n_components == nothing
        M = MultivariateStats.fit(MultivariateStats.PCA, X')
    else
        M = MultivariateStats.fit(MultivariateStats.PCA, X'; maxoutdim=pca.n_components)
    end
    pca.M = M
    pca.explained_variance_ratio_ = MultivariateStats.principalvars(M)
    pca.explained_variance_ratio_ /= sum(pca.explained_variance_ratio_)
    MultivariateStats.transform(M, X')'
end

####### FastICA
type FastICA <: Preprocessor
    n_components::Noint
    params::Dict{Symbol, Any}
    M::MultivariateStats.ICA
    components_::Matrix{Float64}
    function FastICA(n_components::Noint, whiten::Bool, max_iter::Noint, tol::Nofloat)
        fica = new(n_components, Dict{Symbol, Any}())
        fica.params[:do_whiten] = whiten
        if max_iter != nothing fica.params[:maxiter] = max_iter end
        if tol != nothing fica.params[:tol] = tol end
        fica
    end
end
FastICA(;n_components::Noint=nothing, whiten::Bool=false, max_iter::Noint=nothing, tol::Nofloat=nothing) = FastICA(n_components, whiten, max_iter, tol)
function fit_transform!(ica::FastICA, X::Matrix{Float64})
    # Prepare the parameters
    n_components = ica.n_components == nothing ? size(X, 2) : ica.n_components
    M = MultivariateStats.fit(MultivariateStats.ICA, X', n_components; ica.params...)
    ica.M = M
    ica.components_ = M.W'
    MultivariateStats.transform(M, X')
end

################ Pipeline ###############
type Pipeline{T<:Estimator} <: Estimator
    preprocessors::Vector{Tuple{ASCIIString, Preprocessor}}
    estimator::Tuple{ASCIIString, T}
end
Pipeline{T<:Estimator}(prepro::Tuple{ASCIIString, Preprocessor}, est::Tuple{ASCIIString, T}) = Pipeline([prepro], est)
function fit_transform!(prepros::Vector{Preprocessor}, X::Matrix{Float64})
    for prepro = prepros
        X = fit_transform!(prepro, X)
    end
    X
end
function fit!(pipe::Pipeline, X::Matrix{Float64}, y::Vector)
    prepros::Vector{Preprocessor} = map(x->x[2], pipe.preprocessors)
    X = fit_transform!(prepros, X)
    fit!(pipe.estimator[2], X, y)
    return pipe
end
function predict(pipe::Pipeline, X::Matrix{Float64})
    prepros::Vector{Preprocessor} = map(x->x[2], pipe.preprocessors)
    X = fit_transform!(prepros, X)
    predict(pipe.estimator[2], X)
end

################ Cross Validation ###############
function kfold(y::Vector; k::Integer=10)
    n_items = length(y)
    if n_items < k
        error("n_items < k")
    end
    function kfold_producer()
        ratio = (k - 1)/k
        idx = collect(1:n_items)
        idx_test = idx .> round(Int, ratio*n_items)
        test_len = sum(idx_test)
        for f in 1:k
            idx_test_f = circshift(idx_test, test_len*(f - 1))
            produce(idx[!idx_test_f], idx[idx_test_f])
        end
    end
    Task(kfold_producer)
end

# TODO Extend accepted types of y to anything reasonable
function stratified_kfold(y::Vector{Int}; k::Integer=2)
    # XXX k needs to be leq than the num of least items with lower count of labels
    function stratified_kfold_producer()
        ratio = (k - 1)/k
        labels = unique(y)
        idxs = Dict{ASCIIString, Dict}()
        for lbl in labels
            idxs[string(lbl)] = Dict{ASCIIString, Any}()
            # Get the indices and their locations
            idx_lbl = collect(1:length(y))[y .== lbl]
            idx_lbl_boundary = round(Int, ratio * length(idx_lbl))
            idx_lbl_test_bool = [falses(idx_lbl_boundary); trues(length(idx_lbl) - idx_lbl_boundary)]
            test_len_lbl = sum(idx_lbl_test_bool)
            if length(idx_lbl) < k
                error("Not enough samples for label $(lbl) for $(k) folds.")
            end
            # Store for later use
            idxs[string(lbl)]["idx"] = idx_lbl
            idxs[string(lbl)]["idx_test_bool"] = idx_lbl_test_bool
            idxs[string(lbl)]["test_len"] = test_len_lbl
        end
        for f in 1:k
            idx_tr = []
            idx_test = []
            for lbl in labels
                # Retrieved stored values
                idx_lbl_test_bool = idxs[string(lbl)]["idx_test_bool"]
                test_len_lbl = idxs[string(lbl)]["test_len"]
                idx_lbl = idxs[string(lbl)]["idx"]
                idx_lbl_test_bool_f = circshift(idx_lbl_test_bool, (f - 1)*test_len_lbl)
                idx_lbl_tr = idx_lbl[!idx_lbl_test_bool_f]
                idx_lbl_test = idx_lbl[idx_lbl_test_bool_f]
                # Merge
                idx_tr = vcat(idx_tr, idx_lbl_tr)
                idx_test = vcat(idx_test, idx_lbl_test)
            end
            produce(idx_tr, idx_test)
        end
    end
    Task(stratified_kfold_producer)
end

# TODO extend to take a dict of scoring functions
# TODO Support for average function for scoring?
# TODO keyword arguments
function cross_val_score!(estimator::Estimator, X::Matrix{Float64}, y::Vector; cv::Function=stratified_kfold, scoring::Function=f1_score)
    scores = Dict{ASCIIString, Union(Vector{Float64}, Float64)}()
    for l in unique(y)
        scores[string(l)] = Float64[]
    end
    cv = cv(y)
    for (fold, (idx_tr, idx_test)) in enumerate(cv)
        X_tr = X[idx_tr, :]
        X_test = X[idx_test, :]
        y_tr = y[idx_tr]
        y_test = y[idx_test]

        fit!(estimator, X_tr, y_tr)
        y_pred = predict(estimator, X_test)
        scor_f = scoring(y_test, y_pred)
        for (label, scor) in scor_f
            push!(scores[label], scor)
        end
    end
    for (label, scors) in scores
        scores[label] = mean(scors)
    end
    scores
end

function Base.convert(::Type{Dict{Symbol, Vector}}, params_string::Dict{ASCIIString, Vector})
    params_sym = Dict{Symbol, Vector}()
    for (k, v) in params_string
        params_sym[symbol(k)] = v
    end
    params_sym
end
# TODO Verbosity
type GridSearchCV{T<:Estimator}
    estimator::T
    param_grid::Dict{Symbol, Vector}
    scoring::Function
    cv::Function
    best_score::Float64
    best_estimator::T
    best_params::Dict{Symbol, Any}
    scores::Dict{Dict{Symbol, Any}, Float64}
    GridSearchCV(estimator::T, param_grid::Dict{ASCIIString, Vector}; scoring=f1_score, cv=stratified_kfold) = new(estimator, convert(Dict{Symbol, Vector}, param_grid), scoring, cv, 0.0)
end

# Going through the tree recursively.
function next_params(params, established)
    params = copy(params)
    key = collect(keys(params))[1]
    vals = params[key]
    delete!(params, key)
    for v in vals
        params_cur = copy(established)
        params_cur[key] = v
        if length(keys(params)) == 0
            produce(params_cur)
        else
            next_params(params, params_cur)
        end
    end
end
# XXX How about some metaprogamming instead?
function set_params!(stage::Union{Estimator, Preprocessor}, params)
    for (k, v) in params
        setfield!(stage, symbol(k), v)
    end
    stage
end
function set_params!(stage::Union{SVC, SVR}, params)
    svm::SVMWrapper = stage.svm
    for (k, v) in params
        setfield!(svm, symbol(k), v)
    end
    stage
end
function set_params!(pipe::Pipeline, params)
    function stage_params(params, name::ASCIIString)
        rgx = Regex("^($(name))__(.+)\$")
        stg_params = map(params) do x
            mtch = match(rgx, string(x[1])) 
            mtch == nothing ? nothing : (mtch[2], x[2])
        end
        filter!(x->x != nothing, stg_params)
        Dict{ASCIIString, Any}(stg_params)
    end
    # Iterate over pipe stages
    for stage in pipe.preprocessors
        name = stage[1]
        prepro = stage[2]
        stg_params = stage_params(params, name)
        set_params!(prepro, stg_params)
    end
    # Get params for the estimator
    name = pipe.estimator[1]
    est = pipe.estimator[2]
    # Set the estimator params
    est_params = stage_params(params, name)
    set_params!(est, est_params)
    pipe
end
function fit!{T<:Estimator}(gridsearch::GridSearchCV{T}, X::Matrix, y::Vector)
    param_iterator() = next_params(gridsearch.param_grid, Dict{Symbol, Any}())
    gridsearch.scores = Dict{Dict{Symbol, Any}, Float64}()
    for param_set in Task(param_iterator)
        info("GridSearchCV params: $(param_set)")
        est = set_params!(gridsearch.estimator, param_set)
        scores_param = cross_val_score!(est, X, y; cv=gridsearch.cv, scoring=gridsearch.scoring)
        # Aggregate scores across labels
        score_param = reduce((x1, x2)->("sum", x1[2] + x2[2]), scores_param)[2]/length(scores_param)
        # Store results and their params
        gridsearch.scores[param_set] = score_param
        # Store the best result
        if score_param > gridsearch.best_score
            gridsearch.best_score = score_param
            gridsearch.best_estimator = deepcopy(est)
            gridsearch.best_params = param_set
        end
        info("Score: $(score_param)")
    end
    gridsearch
end

################ Exports ###############
export SVC, fit!, predict, precision_score, recall_score, f1_score, kfold, stratified_kfold, cross_val_score!, GridSearchCV, MinMaxScaler, fit_transform!, Pipeline, MetaPipeline, StandardScaler, PCA, FastICA, SVR, r2_score, mean_squared_error, explained_variance_score, LinearRegression, score, RandomForestRegressor, DecisionTreeRegressor, KNeighborsClassifier, RandomForestClassifier
end

