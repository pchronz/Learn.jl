include("jlearn.jl")
using Base.Test
using jlearn
using PyPlot

# TODO Whitening
# TODO PCA
# TODO ICA

N = 50.0
X = [collect(1:N) collect(1:N)]
y = map(reshape(X[:, 1], size(X)[1])) do x
    if x < 0.3*N
        0
    elseif x > 0.6*N
        1
    else
        1
    end
end

pipe = Pipeline([("mms_1", MinMaxScaler()), ("mms_2", MinMaxScaler())], ("svc", SVC()))
pipe = Pipeline(("mms_1", MinMaxScaler()), ("svc", SVC()))
fit!(pipe, X, y)
@show y
y_pred = predict(pipe, X)
@show y_pred

params = Dict{ASCIIString, Vector}("mms_1__range_min"=>[0.0], "mms_1__range_max"=>[1.0],"svc__C"=>[0.01, 0.1, 1., 10., 100., 1000., 10000., 100000., 1000000.], "svc__kernel"=>["rbf", "linear", "polynomial", "sigmoid"])
# XXX Why doesn't type infernce work here?
#GridSearchCV(pipe, params)
gs = GridSearchCV{Pipeline}(pipe, params)
fit!(gs, X, y)

