module SpinShuttling
using LinearAlgebra
using Statistics
using SpecialFunctions
using QuadGK

include("integration.jl")
include("analytics.jl")
include("stochastics.jl")

export ShuttlingModel, OneSpinModel, TwoSpinModel, fidelity, sampling
export OrnsteinUhlenbeckField, PinkBrownianField, GaussianRandomField, RandomFunction

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

mutable struct ShuttlingModel
    n::Int # number of spins
    Ψ::Vector{<:Number}
    T::Real # time 
    M::Int # Monte-Carlo sampling size
    N::Int # Time discretization 
    B::GaussianRandomField # Noise field
    X::Vector{Function}
    R::RandomFunction
    # incomplete initialization
    ShuttlingModel(n, Ψ, T, M, N, B, X)=new(n, Ψ, T, M, N, B, X)
end

function Base.show(io::IO, model::ShuttlingModel)
    println(io, "Model for spin shuttling")
    println(io, "T=$(model.T), L=$(model.L)")
    println(io, "n=$(model.n), |Ψ₀⟩=$(round.(model.Ψ; digits=3))")
    println(io, "Noise Channel: $(model.B)")
    println(io, "Monte-Carlo Parameter: M=$(model.M), N=$(model.N)")
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
- `instantiate::Bool`: Whether to instantiate the random function
"""
function OneSpinModel(Ψ::Vector{<:Number}, T::Real, M::Int, N::Int,
    B::GaussianRandomField, x::Function; instantiate::Bool=true)

    model = ShuttlingModel(1, Ψ, T, M, N, B, [x])
    if instantiate
        t = range(0, T, N)
        P = hcat(t, x.(t))
        model.R = RandomFunction(P, B)
    end
    return model
end

"""
One spin shuttling model initialzied at |Ψ₀⟩=|+⟩.
The qubit is shuttled at constant velocity along the path `x(t)=L/T*t`, 
with total time `T` in `μs` and length `L` in `μm`.
"""
OneSpinModel(T::Real, L::Real, M::Int, N::Int, B::GaussianRandomField; instantiate::Bool=true) =
    OneSpinModel(1 / √2 * [1, 1], T, M, N, B, t -> L / T * t, instantiate=instantiate)


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
- `instantiate::Bool`: Whether to instantiate the random function
"""
function TwoSpinModel(Ψ::Vector{<:Number}, T::Real, M::Int, N::Int,
    B::GaussianRandomField, x₁::Function, x₂::Function; instantiate::Bool=true)
    
    X = [x₁, x₂]
    model=ShuttlingModel(2, Ψ, T, M, N, B, X)
    if instantiate
        t = range(0, T, N)
        P=vcat(hcat(t, x₁.(t)), hcat(t, x₂.(t)))
        model.R=RandomFunction(P, B)
    end
    return model
end

"""
Two spin shuttling model initialized at the singlet state `|Ψ₀⟩=1/√2(|↑↓⟩-|↓↑⟩)`.
The qubits are shuttled at constant velocity along the path `x₁(t)=L/T₁*t` and `x₂(t)=L/T₁*(t-T₀)`. 
The delay between the them is `T₀` and the total shuttling time is `T₁+T₀`.
It should be noticed that due to the exclusion of fermions, `x₁(t)` and `x₂(t)` cannot overlap.
"""
function TwoSpinModel(T₀::Real, T₁::Real, L::Real, M::Int, N::Int, 
    B::GaussianRandomField; instantiate::Bool=true)

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
    return TwoSpinModel(Ψ, T, M, N, B, x₁, x₂, instantiate=instantiate)
end

"""
Calculate the fidelity of a spin shuttling model with respect to the initial state.
"""
function fidelity(model::ShuttlingModel, vector::Bool=false)::Union{Real,Vector{<:Real}}
    N=model.N 
    dt= model.T/model.N
    # model.R is immutable
    return vector ? characteristicfunction(model.R) : characteristicvalue(model.R)
end


"""
Monte-Carlo sampling of any objective function. 
The function must return Tuple{Real,Real} or Tuple{Vector{<:Real},Vector{<:Real}}
"""
function sampling(samplingfunction::Function, M::Int)::Union{Tuple{Real,Real},Tuple{Vector{<:Real},Vector{<:Real}}}
    N = length(samplingfunction(1))
    f_sum = N > 1 ? zeros(N) : 0
    f_var = copy(f_sum)
    for i in 1:M
        f_p = samplingfunction(i)::Union{Real,Vector{<:Real}}
        f_sum += f_p
        f_var += i > 1 ? abs.(i * f_p - f_sum) .^ 2 / (i * (i - 1)) : f_var
    end
    return f_sum / M, f_var / (M - 1)
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
        Z = A[1:N] - A[N+1:end]
    else
        Z = missing
    end
    phi = vector ? [cumsum((Z[1:N-1] + Z[2:N]) * dt / 2)] : sum(Z)*dt
    return cos.(phi)
end

function Χ(T0::Real, T1::Real, L::Real, B::OrnsteinUhlenbeckField)
    return Χ(T0, T1, L, B.θ[1], B.θ[2], B.σ)
end

function Χ(T::Real,L::Real,B::OrnsteinUhlenbeckField)::Real
    return Χ(T, L, B.θ[1], B.θ[2], B.σ)
end

end