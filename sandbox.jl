include("jlearn.jl")
using Base.Test
using jlearn
using PyPlot

# TODO Whitening
# TODO PCA
# TODO ICA
# TODO Pipeline

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

# TODO GridSearchCV
params = Dict{ASCIIString, Vector}("C"=>[0.01, 0.1, 1., 10., 100., 1000., 10000., 100000., 1000000.], "kernel"=>["rbf", "linear", "polynomial", "sigmoid"])
# XXX Why doesn't type infernce work here?
#GridSearchCV(pipe, params)
gs = GridSearchCV{Pipeline}(pipe, params)
fit!(gs, X, y)
# TODO test w/ multiple different prepros

