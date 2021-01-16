module NN

include("backprop.jl")
include("errors.jl")
include("activation.jl")

using LinearAlgebra

struct NN
    layer::Array{Int64,1}
end

function compute(N::NN,W::Array{Array{Float64,2},1},Input::Array{Float64,1})


end
