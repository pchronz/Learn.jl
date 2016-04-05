# TODO run in parallel wherever possible or advantageous
module jlearn
################ SVC ###############
using LIBSVM
type SVC
    kernel::AbstractString
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
    svc::LIBSVM.SVMModel
    SVC(;kernel="rbf", degree=3, gamma=0.0, coef0=0.0, C=1.0, nu=0.5, p=0.1, cache_size=100.0, eps=0.001, shrinking=true, probability_estimates=false, weights=nothing, verbose=false) = new(kernel, degree, gamma, coef0, C, nu, p, cache_size, eps, shrinking, probability_estimates, weights, verbose)
end

function fit!(clf::SVC, X::Matrix, y::Vector)
    clf.gamma = 1/size(X, 2)
    svc = svmtrain(y, X', 
        kernel_type=LIBSVM.RBF,
        degree=clf.degree,
        gamma=clf.gamma,
        coef0=clf.coef0, 
        C=clf.C,
        nu=clf.nu,
        p=clf.p,
        cache_size=clf.cache_size,
        eps=clf.eps,
        shrinking=clf.shrinking,
        probability_estimates=clf.probability_estimates,
        weights=clf.weights,
        verbose=clf.verbose)
    clf.svc = svc
    clf
end
function predict(clf::SVC, X::Matrix)
    svmpredict(clf.svc, X')[1]
end

################ Metrics ###############
get_tp(y_observed, y_pred, label) = sum((y_observed .== label) & (y_pred .== label))
get_fp(y_observed, y_pred, label) = sum(!(y_observed .== label) & (y_pred .== label))
get_fn(y_observed, y_pred, label) = sum((y_observed .== label) & !(y_pred .== label))

# TODO "Micro" (prec, recall, f1)
# TODO "Weighted" (prec, recall, f1)
# TODO "Samples" (prec, recall, f1)
function fun_score(fun, y_observed, y_pred, ave_fun, pos_label)
    labels = unique([unique(y_observed); unique(y_pred)])
    if length(labels) == 2
        if pos_label == nothing
            return remove_nans(fun(y_observed, y_pred, labels[2]))
        else
            return remove_nans(fun(y_observed, y_pred, pos_label))
        end
    else
        scors = Dict{AbstractString, Float64}()
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
end

precision_score(y_observed, y_pred; ave_fun=nothing, pos_label=nothing) = fun_score(precision_score, y_observed, y_pred, ave_fun, pos_label)
recall_score(y_observed, y_pred; ave_fun=nothing, pos_label=nothing) = fun_score(recall_score, y_observed, y_pred, ave_fun, pos_label)
f1_score(y_observed, y_pred; ave_fun=nothing, pos_label=nothing) = fun_score(f1_score, y_observed, y_pred, ave_fun, pos_label)

remove_nans(scor::Float64) = isnan(scor) ? 0.0 : scor
function remove_nans(scors::Dict{AbstractString, Float64}) 
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

################ Cross Validation ###############
function kfold(y::Vector, k::Integer=10)
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
function stratified_kfold(y::Vector{Int}, k::Integer=2)
    # XXX k needs to be leq than the num of least items with lower count of labels
    if length(y) < k
        error("n_items < k")
    end
    function stratified_kfold_producer()
        ratio = (k - 1)/k
        labels = unique(y)
        idxs = Dict{AbstractString, Dict}()
        for lbl in labels
            idxs[string(lbl)] = Dict{AbstractString, Any}()
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
function cross_val_score!(estimator::SVC, X::Matrix, y::Vector, scoring::Function=f1_score, cv::Function=stratified_kfold, n_folds=10)
    scores = Dict{AbstractString, Union(Vector{Float64}, Float64)}()
    for l in unique(y)
        scores[string(l)] = zeros(n_folds)
    end
    cv = cv(y, n_folds)
    for (fold, (idx_tr, idx_test)) in enumerate(cv)
        X_tr = X[idx_tr, :]
        X_test = X[idx_test, :]
        y_tr = y[idx_tr]
        y_test = y[idx_test]

        fit!(estimator, X_tr, y_tr)
        y_pred = predict(estimator, X_test)
        scor_f = scoring(y_test, y_pred)
        for (label, scor) in scor_f
            scores[label][fold] = scor
        end
    end
    # Compute the mean values
    for (label, scors) in scores
        scores[label] = mean(scors)
    end
    scores
end

################ Exports ###############
export SVC, fit!, predict, precision_score, recall_score, f1_score, kfold, stratified_kfold, cross_val_score!
end

