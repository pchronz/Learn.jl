include("jlearn.jl")
using Base.Test
using jlearn

N = 50
X = [collect(1:N) collect(1:N)]
y = map(reshape(X[:, 1], size(X)[1])) do x
    if x < 0.3*N
        0
    elseif x > 0.6*N
        2
    else
        1
    end
end
cross_val_score!(SVC(), X, y)

