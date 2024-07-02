module SpinShuttling

using LinearAlgebra
using Statistics
using SpecialFunctions
using QuadGK
using UnicodePlots: lineplot, lineplot!
using Base.Threads

include("integration.jl")
include("analytics.jl")
include("stochastics.jl")

export ShuttlingModel, OneSpinModel, TwoSpinModel, 
OneSpinForthBackModel, TwoSpinParallelModel, RandomFunction, CompositeRandomFunction,
OrnsteinUhlenbeckField, PinkBrownianField
export averagefidelity, fidelity, sampling, characteristicfunction, characteristicvalue
export dephasingmatrix, covariance, covariancematrix
export W

"""
Spin shuttling model defined by a stochastic field, the realization of the stochastic field is 
specified by the paths of the shuttled spins.

# Arguments
- `n::Int`: Number of spins
- `Ψ::Vector{<:Number}`: Initial state of the spin system, the length of the vector must be `2^n
- `T::Real`: Maximum time 
- `N::Int`: Time discretization 
- `B::GaussianRandomField`: Noise field
- `X::Vector{Function}`: Shuttling paths, the length of the vector must be `n
- `R::RandomFunction`: Random function of the sampled noises on paths
"""

struct ShuttlingModel
    n::Int # number of spins
    Ψ::Vector{<:Number}
    T::Real # time 
    N::Int # Time discretization 
    B::GaussianRandomField # Noise field
    X::Vector{Function}
    R::RandomFunction
end

function Base.show(io::IO, model::ShuttlingModel)
    println(io, "Model for spin shuttling")
    println(io, "Spin Number: n=$(model.n)")
    println(io, "Initial State: |Ψ₀⟩=$(round.(model.Ψ; digits=3))")
    println(io, "Noise Channel: $(model.B)")
    println(io, "Time Discretization: N=$(model.N)")
    println(io, "Process Time: T=$(model.T)")
    println(io, "Shuttling Paths:")
    t=range(0, model.T, model.N)
    fig=lineplot(t, model.X[1].(t); width=30, height=9,
    name="x1(t)")
    for i in 2:model.n
        lineplot!(fig, t, model.X[i].(t), name="x$i(t)")
    end
    display(fig)
end

"""
General one spin shuttling model initialized at initial state |Ψ₀⟩, 
with arbitrary shuttling path x(t). 

# Arguments
- `Ψ::Vector{<:Number}`: Initial state of the spin system, the length of the vector must be `2^n
- `T::Real`: Maximum time
- `N::Int`: Time discretization
- `B::GaussianRandomField`: Noise field
- `x::Function`: Shuttling path
"""
function OneSpinModel(Ψ::Vector{<:Number}, T::Real, N::Int,
    B::GaussianRandomField, x::Function)

    t = range(0, T, N)
    P = collect(zip(t, x.(t)))
    R = RandomFunction(P, B)
    model = ShuttlingModel(1, Ψ, T, N, B, [x], R)
    return model
end


"""
One spin shuttling model initialzied at |Ψ₀⟩=|+⟩.
The qubit is shuttled at constant velocity along the path `x(t)=L/T*t`, 
with total time `T` in `μs` and length `L` in `μm`.
"""
OneSpinModel(T::Real, L::Real, N::Int, B::GaussianRandomField) =
    OneSpinModel(1 / √2 * [1, 1], T, N, B, t::Real -> L / T * t)

"""
One spin shuttling model initialzied at |Ψ₀⟩=|+⟩.
The qubit is shuttled at constant velocity along a forth-back path 
`x(t, T, L) = t<T/2 ? 2L/T*t : 2L/T*(T-t)`, 
with total time `T` in `μs` and length `L` in `μm`.

# Arguments
- `T::Real`: Maximum time
- `L::Real`: Length of the path
- `N::Int`: Time discretization
- `B::GaussianRandomField`: Noise field
- `v::Real`: Velocity of the shuttling
"""
function OneSpinForthBackModel(t::Real, T::Real, L::Real, N::Int, B::GaussianRandomField) 
    x(t::Real, v::Real, L::Real)::Real = (t=t%(2L/v); v*t < L ? v*t : 2L-v*t)
    return OneSpinModel(1 / √2 * [1, 1], t, N, B, τ -> x(τ, 2L/T, L))
end

function OneSpinForthBackModel(T::Real, L::Real, N::Int, B::GaussianRandomField) 
    return OneSpinForthBackModel(T, T, L, N, B)
end


"""
General two spin shuttling model initialized at initial state |Ψ₀⟩,
with arbitrary shuttling paths x₁(t), x₂(t).

# Arguments
- `Ψ::Vector{<:Number}`: Initial state of the spin system, the length of the vector must be `2^n
- `T::Real`: Maximum time
- `N::Int`: Time discretization
- `B::GaussianRandomField`: Noise field
- `x₁::Function`: Shuttling path for the first spin
- `x₂::Function`: Shuttling path for the second spin
"""
function TwoSpinModel(Ψ::Vector{<:Number}, T::Real, N::Int,
    B::GaussianRandomField, x₁::Function, x₂::Function)

    X = [x₁, x₂]
    t = range(0, T, N)
    P = vcat(collect(zip(t, x₁.(t))), collect(zip(t, x₂.(t))))
    R = RandomFunction(P, B)
    model = ShuttlingModel(2, Ψ, T, N, B, X, R)
    return model
end

"""
Two spin shuttling model initialized at the singlet state `|Ψ₀⟩=1/√2(|↑↓⟩-|↓↑⟩)`.
The qubits are shuttled at constant velocity along the path `x₁(t)=L/T₁*t` and `x₂(t)=L/T₁*(t-T₀)`. 
The delay between the them is `T₀` and the total shuttling time is `T₁+T₀`.
It should be noticed that due to the exclusion of fermions, `x₁(t)` and `x₂(t)` cannot overlap.
"""
function TwoSpinModel(T₀::Real, T₁::Real, L::Real, N::Int, B::GaussianRandomField)
    function x₁(t::Real)::Real
        if t < 0
            return 0
        elseif 0 ≤ t < T₁
            return L / T₁ * t
        elseif T₁ ≤ t
            return L
        end
    end
    # δ small shift to avoid overlap
    δ = 1e-6
    function x₂(t::Real)::Real
        if t < T₀
            return δ
        elseif T₀ ≤ t < T₁ + T₀
            return L / T₁ * (t - T₀) + δ
        elseif t ≥ T₁ + T₀
            return L + δ
        end
    end
    Ψ = 1 / √2 .* [0, 1, -1, 0]
    T = T₀ + T₁
    return TwoSpinModel(Ψ, T, N, B, x₁, x₂)
end

"""
Two spin shuttling model initialized at the singlet state `|Ψ₀⟩=1/√2(|↑↓⟩-|↓↑⟩)`.
The qubits are shuttled at constant velocity along the 2D path 
`x₁(t)=L/T*t, y₁(t)=0` and `x₂(t)=L/T*t, y₂(t)=D`.
The total shuttling time is `T` and the length of the path is `L` in `μm`.
"""
function TwoSpinParallelModel(T::Real, D::Real, L::Real, N::Int,
    B::GaussianRandomField)
    @assert length(B.θ)>=3 
    x₁(t::Real)::Tuple{Real,Real} = (L / T * t, 0)
    x₂(t::Real)::Tuple{Real,Real} = (L / T * t, D)
    Ψ = 1 / √2 .* [0, 1, -1, 0]
    return TwoSpinModel(Ψ, T, N, B, x₁, x₂)
end

"""
Calculate the average fidelity of a spin shuttling model using numerical integration 
of the covariance matrix.

# Arguments
- `model::ShuttlingModel`: The spin shuttling model

"""
function averagefidelity(model::ShuttlingModel)::Real
    # model.R is immutable
    if model.n == 1
        R = model.R
    elseif model.n == 2
        R = CompositeRandomFunction(model.R, [1, -1])
    elseif model.n > 2
        error("The number of spins is not supported")
    end
    W = real(characteristicvalue(R))
    F = @. 1 / 2 * (1 + W)
    return F
end


"""
Monte-Carlo sampling of any objective function. 
The function must return Tuple{Real,Real} or Tuple{Vector{<:Real},Vector{<:Real}}

# Arguments
- `samplingfunction::Function`: The function to be sampled
- `M::Int`: Monte-Carlo sampling size
# Returns
- `Tuple{Real,Real}`: The mean and variance of the sampled function
- `Tuple{Vector{<:Real},Vector{<:Real}}`: The mean and variance of the sampled function
# Example
```julia
f(x) = x^2
sampling(f, 1000)
```

# Reference
https://en.wikipedia.org/wiki/Standard_deviation#Rapid_calculation_methods

"""
function sampling(samplingfunction::Function, M::Int)::Union{Tuple{Real,Real},Tuple{Vector{<:Real},Vector{<:Real}}}
    if nthreads() > 1
        return parallelsampling(samplingfunction, M)
    end
    N = length(samplingfunction(1))
    A = N > 1 ? zeros(N) : 0
    Q = copy(A)
    for k in 1:M
        x = samplingfunction(k)::Union{Real,Vector{<:Real}}
        Q = Q +(k-1)/k*(x-A).^ 2
        A = A + (x-A)/k
    end
    return A, Q / (M - 1)
end

function parallelsampling(samplingfunction::Function, M::Int)::Union{Tuple{Real,Real},Tuple{Vector{<:Real},Vector{<:Real}}}
    N = length(samplingfunction(1))
    if N > 1
        cache = zeros(N, M)
        @threads for i in 1:M
            cache[:, i] .= samplingfunction(i)
        end
        A = mean(cache, dims=2)
        Q = var(cache, dims=2)
        return A, Q 
    else
        cache = zeros(M)
        @threads for i in 1:M
            cache[i] = samplingfunction(i)
        end
        A = mean(cache)
        Q = var(cache)
        return A, Q 
    end
end


"""
Sampling an observable that defines on a specific spin shuttling model 

# Arguments
- `model::ShuttlingModel`: The spin shuttling model
- `objective::Function`: The objective function `objective(mode::ShuttlingModel; randseq)``
- `M::Int`: Monte-Carlo sampling size
"""
function sampling(model::ShuttlingModel, objective::Function, M::Int; vector::Bool=false)
    randpool = randn(model.n * model.N, M)
    samplingfunction = i::Int -> objective(model, randpool[:, i]; vector=vector)::Union{Real,Vector{<:Real}}
    return sampling(samplingfunction, M)
end

"""

# Arguments
- `i::Int`: index of qubits, range from (1,n)
- `p::Int`: index of spin state, range from (1,2^n)
- `n::Int`: number of spins
"""
function m(i::Int,p::Int,n::Int)
    1/2-digits(p, base=2, pad=n)[i]
end

"""
Calculate the dephasing matrix of a given spin shuttling model.
"""
function dephasingmatrix(model::ShuttlingModel)::Symmetric{<:Real}
    n=model.n
    W=zeros(2^n,2^n)
    for j in 1:2^n
        W[j,j]=1
        for k in 1:j-1
            c=[trunc(Int,m(i,j-1,n)-m(i,k-1,n)) for i in 1:n]
            R = CompositeRandomFunction(model.R, c)
            W[j,k] = characteristicvalue(R)
            W[k,j] = W[j,k]
        end
    end
    return Symmetric(W)
end

function dephasingcoeffs(n::Int)::Array{Real,3}
    M=zeros(2^n,2^n, n)
    for j in 1:2^n
        for k in 1:2^n
            c=[m(i,j-1,n)-m(i,k-1,n) for i in 1:n]
            M[j,k, :] = c
        end
    end
    return M
end

"""
Sample a phase integral of the process. 
The integrate of a random function should be obtained 
from directly summation without using high-order interpolation 
(Simpson or trapezoid). 
"""
function fidelity(model::ShuttlingModel, randseq::Vector{<:Real}; vector::Bool=false)::Union{Real,Vector{<:Real}}
    # model.R || error("covariance matrix is not initialized")
    N = model.N
    dt = model.T / N
    A = model.R(randseq)
    if model.n == 1
        Z = A
    elseif model.n == 2
        # only valid for two-spin EPR pair, ψ=1/√2(|↑↓⟩-|↓↑⟩)
        Z = A[1:N] - A[N+1:end] 
    else
        Z = missing
    end
    ϕ = vector ? cumsum(Z)* dt : sum(Z) * dt
    return (1 .+ cos.(ϕ)) / 2
end



"""
Analytical dephasing factor of a one-spin shuttling model.

# Arguments
- `T::Real`: Total time
- `L::Real`: Length of the path
- `B<:GaussianRandomField`: Noise field, Ornstein-Uhlenbeck or Pink-Brownian
- `path::Symbol`: Path of the shuttling model, `:straight` or `:forthback`
"""
function W(T::Real,L::Real,B::OrnsteinUhlenbeckField; path=:straight)::Real
    κₜ=B.θ[1]
    κₓ=B.θ[2]
    σ =B.σ
    β = κₜ*T
    γ = κₓ*L
    if path == :straight
        return exp(- σ^2/(4*κₜ*κₓ)/κₜ^2*P1(β, γ)/2)
    elseif path == :forthback
        β/=2
        return exp(- σ^2/(4*κₜ*κₓ)/κₜ^2*(P1(β, γ)+P4(β,γ)))
    else
        error("Path not recognized. Use :straight or :forthback for one-spin shuttling model.")
    end
end

function W(T::Real, L::Real, B::PinkBrownianField)::Real
    β= T.*B.γ
    γ= L*B.θ[1]
    return exp(-B.σ^2*T^2*F3(β,γ))
end


"""
Analytical dephasing factor of a sequenced two-spin EPR pair shuttling model.
"""
function W(T0::Real,T1::Real,L::Real,B::OrnsteinUhlenbeckField; path=:sequenced)::Real
    κₜ=B.θ[1]
    κₓ=B.θ[2]
    σ =B.σ
    τ = κₜ*T0
    β = κₜ*T1
    γ = κₓ*L
    if path == :sequenced
        return exp(-σ^2/(4*κₜ*κₓ)/κₜ^2*(F1(β, γ, τ)-F2(β, γ, τ)))
    elseif path == :parallel
        missing("Parallel path not implemented yet.")
    else
        error("Path not recognized. Use :sequenced or :parallel for two-spin EPR pair shuttling model.")
    end
end

end