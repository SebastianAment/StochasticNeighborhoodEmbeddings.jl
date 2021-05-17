mutable struct OptimizationParameters
    d::Int # dimension of embedding
    # main optimization parameters
    max_iter::Int
    lr::Real
    beta_1::Real
    beta_2::Real
    weight_decay::Real
    eps::Real # adding to denominator of ADAM rescaling
    # early optimization parameters
    early_iter::Int
    early_lr::Real
    early_beta_1::Real
    early_beta_2::Real
    early_compression::Real
    early_exaggeration::Real
    # boolean flags
    dofast::Bool
    debias::Bool
    verbose::Bool
end

function OptimizationParameters(; d::Int = 2, max_iter::Int = 1024, lr::Real = 1,
            beta_1::Real = .9, beta_2::Real = .999, weight_decay::Real = 0,
            early_iter::Int = 64, early_lr::Real = lr, early_beta_1::Real = 0.5,
            early_beta_2::Real = 0.9, early_compression::Real = 1,
            early_exaggeration::Real = 12, eps::Real = 1e-8, verbose::Bool = false,
            debias::Bool = true, dofast::Bool = size(X, 2) ≥ fast_algorithm_min_size)
    early_exaggeration ≥ 1 || throw(DomainMismatch("early_exaggeration < 1"))
    OptimizationParameters(d, max_iter, lr, beta_1, beta_2, weight_decay, eps,
                early_iter, early_lr, early_beta_1, early_beta_2, early_compression,
                early_exaggeration, dofast, debias, verbose)
end
