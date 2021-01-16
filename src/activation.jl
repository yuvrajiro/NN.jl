# File for Activation Function


# Sigmoid Functiom

Sigmoid(x::Float64) = 1/(1+exp(-x))

# ReLU Function

ReLU(x::Float64) = max(x,0)

# Identity Function

identity(x::Float64) = x
