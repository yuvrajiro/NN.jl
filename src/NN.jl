module NN

include("backprop.jl")
include("errors.jl")
include("activation.jl")

using LinearAlgebra

struct perceptron
  activation::Function
end

struct layer
  layer::Array{perceptron,1}
end

struct NN
  NN::Array{layer,1}
end


perceptron_compute(x::perceptron,W::Array{Float64,1},I::Array{Float64,2}) = x.activation.(I * W)
layer_compute(l::layer,W::Array{Float64,2},I::Array{Float64,2}) = [pcom(i,W[:,j],I) for (i,j) in zip(l.layer,1:size(W,2)]

NeuralNetwork(x::Array{Tuple{Function,Int64},1}) = NN([layer([perceptron(x[i][1]) for k in 1:(x[i][2]+1)]) for i in 1:length(x)])

end
