module SpinShuttling

using LinearAlgebra
using Statistics
using SpecialFunctions
using QuadGK

include("integration.jl")
include("analytics.jl")
include("stochastics.jl")

export ShuttlingModel, OneSpinModel, TwoSpinModel, 
OneSpinForthBackModel, TwoSpinParallelModel, RandomFunction,
OrnsteinUhlenbeckField, PinkBrownianField
export averagefidelity, fidelity, sampling
export Χe

"""
Spin shuttling model defined by a stochastic field, the realization of the stochastic field is 
specified by the paths of the shuttled spins.

# Arguments
- `n::Int`: Number of spins
- `Ψ::Vector{<:Number}`: Initial state of the spin system, the length of the vector must be `2^n
- `T::Real`: Maximum time 
- `M::Int`:  Monte-Carlo sampling size
- `N::Int`: Time discretization 
- `B::GaussianRandomField`: Noise field
- `X::Vector{Function}`: Shuttling paths, the length of the vector must be `n
- `R::RandomFunction`: Random function of the sampled noises on paths
"""

struct ShuttlingModel
    n::Int # number of spins
    Ψ::Vector{<:Number}
    T::Real # time 
    M::Int # Monte-Carlo sampling size
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
    println(io, "Monte-Carlo Parameter: M=$(model.M), N=$(model.N)")
    println(io, "Process Time: T=$(model.T)")
    println(io, "Spin Trajectories: {X_i(t)}=$(model.X)")
end

"""
General one spin shuttling model initialized at initial state |Ψ₀⟩, 
with arbitrary shuttling path x(t). 
# Arguments
- `Ψ::Vector{<:Number}`: Initial state of the spin system, the length of the vector must be `2^n
- `T::Real`: Maximum time
- `M::Int`:  Monte-Carlo sampling size
- `N::Int`: Time discretization
- `B::GaussianRandomField`: Noise field
- `x::Function`: Shuttling path
"""
function OneSpinModel(Ψ::Vector{<:Number}, T::Real, M::Int, N::Int,
    B::GaussianRandomField, x::Function)

    t = range(0, T, N)
    P = hcat(t, x.(t))
    R = RandomFunction(P, B)
    model = ShuttlingModel(1, Ψ, T, M, N, B, [x], R)
    return model
end


"""
One spin shuttling model initialzied at |Ψ₀⟩=|+⟩.
The qubit is shuttled at constant velocity along the path `x(t)=L/T*t`, 
with total time `T` in `μs` and length `L` in `μm`.
"""
OneSpinModel(T::Real, L::Real, M::Int, N::Int, B::GaussianRandomField) =
    OneSpinModel(1 / √2 * [1, 1], T, M, N, B, t::Real -> L / T * t)

"""
One spin shuttling model initialzied at |Ψ₀⟩=|+⟩.
The qubit is shuttled at constant velocity along a forth-back path 
`x(t, T, L) = t<T/2 ? 2L/T*t : 2L/T*(T-t)`, 
with total time `T` in `μs` and length `L` in `μm`.
"""
function OneSpinForthBackModel(T::Real, L::Real, M::Int, N::Int, B::GaussianRandomField) 
    x(t::Real, T::Real, L::Real)::Real = t<T/2 ? 2L/T*t : 2L/T*(T-t)
    return OneSpinModel(1 / √2 * [1, 1], T, M, N, B, t::Real -> x(t, T, L))
end

"""
General two spin shuttling model initialized at initial state |Ψ₀⟩,
with arbitrary shuttling paths x₁(t), x₂(t).
# Arguments
- `Ψ::Vector{<:Number}`: Initial state of the spin system, the length of the vector must be `2^n
- `T::Real`: Maximum time
- `M::Int`:  Monte-Carlo sampling size
- `N::Int`: Time discretization
- `B::GaussianRandomField`: Noise field
- `x₁::Function`: Shuttling path for the first spin
- `x₂::Function`: Shuttling path for the second spin
"""
function TwoSpinModel(Ψ::Vector{<:Number}, T::Real, M::Int, N::Int,
    B::GaussianRandomField, x₁::Function, x₂::Function)

    X = [x₁, x₂]
    t = range(0, T, N)
    P = vcat(hcat(t, x₁.(t)), hcat(t, x₂.(t)))
    R = RandomFunction(P, B)
    model = ShuttlingModel(2, Ψ, T, M, N, B, X, R)
    return model
end

"""
Two spin shuttling model initialized at the singlet state `|Ψ₀⟩=1/√2(|↑↓⟩-|↓↑⟩)`.
The qubits are shuttled at constant velocity along the path `x₁(t)=L/T₁*t` and `x₂(t)=L/T₁*(t-T₀)`. 
The delay between the them is `T₀` and the total shuttling time is `T₁+T₀`.
It should be noticed that due to the exclusion of fermions, `x₁(t)` and `x₂(t)` cannot overlap.
"""
function TwoSpinModel(T₀::Real, T₁::Real, L::Real, M::Int, N::Int,
    B::GaussianRandomField)

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
    return TwoSpinModel(Ψ, T, M, N, B, x₁, x₂)
end

"""
Two spin shuttling model initialized at the singlet state `|Ψ₀⟩=1/√2(|↑↓⟩-|↓↑⟩)`.
The qubits are shuttled at constant velocity along the 2D path 
`x₁(t)=L/T*t, y₁(t)=0` and `x₂(t)=L/T*t, y₂(t)=D`.
The total shuttling time is `T` and the length of the path is `L` in `μm`.
"""
function TwoSpinParallelModel(T::Real, D::Real, L::Real, M::Int, N::Int,
    B::GaussianRandomField)
    @assert length(B.θ)>=3 
    x₁(t::Real)::Tuple{Real,Real} = (L / T * t, 0)
    x₂(t::Real)::Tuple{Real,Real} = (L / T * t, D)
    Ψ = 1 / √2 .* [0, 1, -1, 0]
    return TwoSpinModel(Ψ, T, M, N, B, x₁, x₂)
end

"""
Calculate the average fidelity of a spin shuttling model using numerical integration 
of the covariance matrix.
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
    χ = real(characteristicvalue(R))
    F = @. 1 / 2 * (1 + χ)
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

"""
Sampling an observable that defines on a specific spin shuttling model 
objective(mode, randseq, vector)
"""
function sampling(model::ShuttlingModel, objective::Function; vector::Bool=false)
    randpool = randn(model.n * model.N, model.M)
    samplingfunction = i::Int -> objective(model, randpool[:, i]; vector=vector)::Union{Real,Vector{<:Real}}
    return sampling(samplingfunction, model.M)
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
    phi = vector ? cumsum(Z)* dt : sum(Z) * dt
    return (1 .+ cos.(phi)) / 2
end


"""
Theoretical fidelity of a sequenced two-spin EPR pair shuttling model.
"""
function Χ(T0::Real, T1::Real, L::Real, B::OrnsteinUhlenbeckField)
    return Χ(T0, T1, L, B.θ[1], B.θ[2], B.σ)
end

"""
Theoretical fidelity of a one-spin shuttling model.
"""
function Χ(T::Real, L::Real, B::OrnsteinUhlenbeckField)::Real
    return Χ(T, L, B.θ[1], B.θ[2], B.σ)
end


"""
Theoretical fidelity of a one-spin shuttling model for a pink noise.
"""
function Χ(T::Real, B::PinkBrownianField)::Real
    return exp(-B.σ^2/2*ϕ(T, B.γ))
end

end