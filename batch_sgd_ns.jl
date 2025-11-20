# Source:
# https://github.com/KellerJordan/Muon/blob/master/muon.py


using LinearAlgebra, Random

function NewtonSchulz5(
    G::AbstractMatrix{T}, steps::Int=5
) where T <: AbstractFloat

    # "Cursed" coefficients from the paper
    a, b, c = T(3.4445), T(-4.7750), T(2.0315)
    # Alternative coefficients, less steep but sigma closer to one
    # a, b, c = T(2.9722), T(-3.4445), T(1.4722)
    
    X = Matrix{T}(G)

    rows, cols = size(G)
    transpose_f = rows > cols

    if transpose_f
        X = X'
    end

    # Spectral normalization
    eps = 1e-7
    X = X / (opnorm(X, 2) + eps)

    # Newton-Schulz iterations
    for _ in 1:steps
        A = X * X'
        B = b * A + c * A * A
        X = a * X + B * X
    end

    if transpose_f
        X = X'
    end

    return X
end

#=
    Compute update for weights using SGD with momentum
    and optional Newton-Schulz normalization.
    
    grad: 
        Gradient matrix.
    momentum: 
        Accumulated momentum matrix, same
        dimensions as grad.
    newton_schulz:
        Whether to apply Newton-Schulz
        normalization to the momentum.
    beta:
        Exponential moving average coefficient.
=#
function WeightUpdate(
    grad::AbstractMatrix{T}, momentum::AbstractMatrix{T}, newton_schulz::Bool=true, beta::T=T(0.95)
) where T <: AbstractFloat

    momentum = beta * momentum + (1 - beta) * grad
    update = newton_schulz ? NewtonSchulz5(momentum) : momentum

    rows, cols = size(grad)
    update *= sqrt(max(1., rows / cols))
    return update, momentum
end

#=
    Make a linear prediction given a weight matrix
    and an input: y = xW

    params:
        Weight matrix (input_dim, output_dim).
    input:
        Input vector (, input_dim) or matrix (batch_size, input_dim).
=#
function LinearPredict(
    params::AbstractMatrix{T}, input::AbstractArray{T}
) where T <: AbstractFloat

    if ndims(input) == 1
        input = reshape(input, 1, :)
    elseif ndims(input) != 2
        throw(ArgumentError("Input must be a vector or a matrix"))
    end

    return input * params
end

function L2Loss(
    pred::AbstractArray{T}, target::AbstractArray{T}
) where T <: AbstractFloat

    return sum((pred .- target) .^ 2) / (2 * size(pred, 1))
end

#=
    Compute the gradient of the L2 loss with respect to the
    parameters matrix underlying (the gradient is a matrix
    with the same shape as the parameters).

    input:
        Input vector (, input_dim) or matrix (batch_size, input_dim).
    pred:
        Prediction vector (, output_dim) or matrix (batch_size, output_dim).
    target:
        Target vector (, output_dim) or matrix (batch_size, output_dim).
=# 
function L2LossGradient(
    input::AbstractArray{T}, pred::AbstractArray{T}, target::AbstractArray{T}
) where T <: AbstractFloat

    if ndims(input) == 1
        input = reshape(input, 1, :)
    elseif ndims(input) != 2
        throw(ArgumentError("Input must be a vector or a matrix"))
    end

    batch_size = size(input, 1)
    grad_output = (pred .- target) / batch_size
    grad_params = input' * grad_output
    return grad_params
end

#=
    Perform a single training step using SGD with momentum
    and optional Newton-Schulz normalization.

    params:
        Weight matrix (input_dim, output_dim).
    input:
        Input vector (, input_dim) or matrix (batch_size, input_dim).
    target:
        Target vector (, output_dim) or matrix (batch_size, output_dim).
    momentum:
        Accumulated momentum matrix, same dimensions as params.
    newton_schulz:
        Whether to apply Newton-Schulz normalization to the momentum.
    learning_rate:
        Learning rate for the SGD update.
    beta:
        Exponential moving average coefficient.
=#
function TrainStep(
    params::AbstractMatrix{T},
    input::AbstractArray{T},
    target::AbstractArray{T},
    momentum::AbstractMatrix{T},
    newton_schulz::Bool=true,
    learning_rate::T=T(0.01),
    beta::T=T(0.95),
) where T <: AbstractFloat

    pred = LinearPredict(params, input)
    grad = L2LossGradient(input, pred, target)
    update, momentum = WeightUpdate(grad, momentum, newton_schulz, beta)
    params -= learning_rate * update

    return params, momentum
end

#=
    Train a linear model using Batch SGD with momentum
    and optional Newton-Schulz normalization.

    params:
        Weight matrix (input_dim, output_dim).
    input:
        Input vector (, input_dim) or matrix (batch_size, input_dim).
    target:
        Target vector (, output_dim) or matrix (batch_size, output_dim).
    epochs:
        Number of training epochs.
    batch_size:
        Size of each training batch.
    newton_schulz:
        Whether to apply Newton-Schulz normalization to the momentum.
    learning_rate:
        Learning rate for the SGD update.
    beta:
        Exponential moving average coefficient.
=# 
function Train(
    params::AbstractMatrix{T},
    input::AbstractArray{T},
    target::AbstractArray{T},
    epochs::Int=1000,
    batch_size::Int=size(input, 1),
    newton_schulz::Bool=true,
    learning_rate::T=T(0.01),
    beta::T=T(0.95)
) where T <: AbstractFloat

    momentum = zeros(T, size(params))
    losses = zeros(T, epochs)

    for epoch in 1:epochs
        indices = randperm(size(input, 1))
        input_shuffled = input[indices, :]
        target_shuffled = target[indices, :]
        epoch_loss = 0.0

        for batch in 1:batch_size:size(input, 1)
            batch_end = min(batch + batch_size - 1, size(input, 1))
            input_batch = input_shuffled[batch:batch_end, :]
            target_batch = target_shuffled[batch:batch_end, :]
            params, momentum = TrainStep(
                params,
                input_batch,
                target_batch,
                momentum,
                newton_schulz,
                learning_rate,
                beta
            )
            pred_batch = LinearPredict(params, input_batch)
            batch_loss = L2Loss(pred_batch, target_batch)
            epoch_loss += batch_loss * (batch_end - batch + 1) / size(input, 1)
        end
        losses[epoch] = epoch_loss
    end

    return params, losses
end