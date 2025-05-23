module SpinShuttling

using LinearAlgebra
using Statistics
using SpecialFunctions
using QuadGK, HCubature
using Base.Threads

include("integration.jl")
include("analytics.jl")
include("stochastics.jl")
include("geometry.jl")
include("sampling.jl")
include("measures.jl")

export ShuttlingModel, OneSpinModel, TwoSpinModel,
    OneSpinForthBackModel, 
    TwoSpinSequentialModel, TwoSpinParallelModel, 
    RandomFunction, CompositeRandomFunction,
    OrnsteinUhlenbeckField, PinkLorentzianField, PinkPiField
export sampling, restriction, initialize!, characteristicfunction, characteristicvalue
export statefidelity, dephasingmatrix, dephasingfactor
export covariance, covariancematrix, concurrence
export W

"""
Spin shuttling model defined by a stochastic field, the realization of the stochastic field is 
specified by the paths of the shuttled spins.

# Arguments
- `n::Int`: Number of spins
- `Ψ::Vector{<:Complex}`: Initial state of the spin system, the length of the vector must be `2^n
- `T::Real`: Maximum time 
- `N::Int`: Time discretization 
- `B::GaussianRandomField`: Noise field
- `X::Vector{<:Function}`: Shuttling paths, the length of the vector must be `n
- `initialize::Bool`: Initialize the random function

# Fields
- `R::RandomFunction`: Random function defined by the noise field and the paths

# Example
```julia
model = ShuttlingModel(1, [1+0im, 0+0im], 1.0, 100, OrnsteinUhlenbeckField([1.0, 1.0, 1.0]), [t->t])
```
"""
struct ShuttlingModel
    n::Int # number of spins
    Ψ::Vector{<:Complex}
    T::Real # time 
    N::Int # Time discretization 
    B::GaussianRandomField # Noise field
    X::Vector{<:Function}
    R::RandomFunction
    function ShuttlingModel(n::Int, Ψ::Vector{<:Complex}, T::Real, N::Int, B::GaussianRandomField, X::Vector{<:Function}; initialize::Bool=true)
        R = restriction(X, range(0, T, N), B; initialize=initialize)
        new(n, Ψ, T, N, B, X, R)
    end
end

function Base.show(io::IO, model::ShuttlingModel)
    println(io, "Model for spin shuttling")
    println(io, "Spin Number: n=$(model.n)")
    println(io, "Initial State: |Ψ₀⟩=$(round.(model.Ψ; digits=3))")
    println(io, "Noise Channel: $(model.B)")
    println(io, "Time Discretization: N=$(model.N)")
    println(io, "Process Time: T=$(model.T)")
end

"""
General one spin shuttling model initialized at initial state |Ψ₀⟩, 
with arbitrary shuttling path x(t). 

# Arguments
- `Ψ::Vector{<:Complex}`: Initial state of the spin system, the length of the vector must be `2^n
- `T::Real`: Maximum time
- `N::Int`: Time discretization
- `B::GaussianRandomField`: Noise field
- `x::Function`: Shuttling path
- `initialize::Bool`: Initialize the random function

# Example
```julia
model = OneSpinModel(1 / √2 * [1+0im, 1+0im], 1.0, 100, OrnsteinUhlenbeckField([1.0, 1.0, 1.0]), t->t)
```
"""
function OneSpinModel(Ψ::Vector{<:Complex}, T::Real, N::Int,
    B::GaussianRandomField, x::Function; initialize::Bool=true)

    model = ShuttlingModel(1, Ψ, T, N, B, [x], initialize=initialize)
    return model
end


"""
One spin shuttling model initialzied at |Ψ₀⟩=|+⟩.
The qubit is shuttled at constant velocity along the path `x(t)=L/T*t`, 
with total time `T` in `μs` and length `L` in `μm`.

# Arguments
- `T::Real`: Maximum time
- `L::Real`: Length of the path
- `N::Int`: Time discretization
- `B::GaussianRandomField`: Noise field
- `v::Real`: Velocity of the shuttling

# Example
```julia
model = OneSpinModel(1.0, 1.0, 100, OrnsteinUhlenbeckField([1.0, 1.0, 1.0]), 1.0)
```
"""
OneSpinModel(T::Real, L::Real, N::Int, B::GaussianRandomField; initialize::Bool=true) =
    OneSpinModel(1 / √2 * [1+0im, 1+0im], T, N, B, t::Real -> L / T * t, initialize=initialize)

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

# Example
```julia
model = OneSpinForthBackModel(1.0, 1.0, 100, OrnsteinUhlenbeckField([1.0, 1.0, 1.0]))
```
"""
function OneSpinForthBackModel(t::Real, T::Real, L::Real, N::Int, B::GaussianRandomField; initialize::Bool=true)   
    x(t::Real, v::Real, L::Real)::Real = (t = t % (2L / v); v * t < L ? v * t : 2L - v * t)
    return OneSpinModel(1 / √2 * [1+0im, 1+0im], t, N, B, τ -> x(τ, 2L / T, L), initialize=initialize)
end

function OneSpinForthBackModel(T::Real, L::Real, N::Int, B::GaussianRandomField; initialize::Bool=true)
    return OneSpinForthBackModel(T, T, L, N, B, initialize=initialize)
end


"""
General two spin shuttling model initialized at initial state |Ψ₀⟩,
with arbitrary shuttling paths x₁(t), x₂(t).

# Arguments
- `Ψ::Vector{<:Complex}`: Initial state of the spin system, the length of the vector must be `2^n
- `T::Real`: Maximum time
- `N::Int`: Time discretization
- `B::GaussianRandomField`: Noise field
- `x₁::Function`: Shuttling path for the first spin
- `x₂::Function`: Shuttling path for the second spin

# Example
```julia
model = TwoSpinModel(1 / √2 * [1+0im, 1+0im, 1+0im, 1+0im], 1.0, 100, OrnsteinUhlenbeckField([1.0, 1.0, 1.0]), t->t, t->t+1)
```
"""
function TwoSpinModel(Ψ::Vector{<:Complex}, T::Real, N::Int,
    B::GaussianRandomField, x₁::Function, x₂::Function; initialize::Bool=true)

    X = [x₁, x₂]
    model = ShuttlingModel(2, Ψ, T, N, B, X; initialize=initialize)
    return model
end

"""
Two spin shuttling model initialized at the singlet state `|Ψ₀⟩=1/√2(|↑↓⟩-|↓↑⟩)`.
The qubits are shuttled at constant velocity along the path `x₁(t)=L/T₁*t` and `x₂(t)=L/T₁*(t-T₀)`. 
The delay between the them is `T₀` and the total shuttling time is `T₁+T₀`.
It should be noticed that due to the exclusion of fermions, `x₁(t)` and `x₂(t)` cannot overlap.

# Arguments
- `Ψ::Vector{<:Complex}`: Initial state of the spin system, the length of the vector must be `4`, default is `1/√2(|↑↓⟩-|↓↑⟩)`
- `T₀::Real`: Delay time
- `T₁::Real`: Shuttling time
- `L::Real`: Length of the path
- `N::Int`: Time discretization
- `B::GaussianRandomField`: Noise field

# Example
```julia
model = TwoSpinSequentialModel(1.0, 1.0, 1.0, 100, OrnsteinUhlenbeckField([1.0, 1.0, 1.0]))
```
"""
function TwoSpinSequentialModel(Ψ::Vector{<:Complex}, T₀::Real, T₁::Real, L::Real, N::Int, B::GaussianRandomField; initialize::Bool=true)
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
    T = T₀ + T₁
    return TwoSpinModel(Ψ, T, N, B, x₁, x₂; initialize=initialize)
end

function TwoSpinSequentialModel(T₀::Real, T₁::Real, L::Real, N::Int, B::GaussianRandomField; initialize::Bool=true)
    return TwoSpinSequentialModel( 1 / √2 .* [0, 1+0im, -1+0im, 0], T₀, T₁, L, N, B; initialize=initialize)
end

"""
Two spin shuttling model initialized at the singlet state `|Ψ₀⟩=1/√2(|↑↓⟩-|↓↑⟩)`.
The qubits are shuttled at constant velocity along the 2D path 
`x₁(t)=L/T*t, y₁(t)=0` and `x₂(t)=L/T*t, y₂(t)=D`.
The total shuttling time is `T` and the length of the path is `L` in `μm`.

# Arguments
- `Ψ::Vector{<:Complex}`: Initial state of the spin system, the length of the vector must be `4`, default is `1/√2(|↑↓⟩-|↓↑⟩)`
- `T::Real`: Total time
- `D::Real`: Distance between the two qubits
- `L::Real`: Length of the path
- `N::Int`: Time discretization
- `B::GaussianRandomField`: Noise field

# Example
```julia
model = TwoSpinParallelModel(1.0, 1.0, 1.0, 100, OrnsteinUhlenbeckField([1.0, 1.0, 1.0]))
```
"""
function TwoSpinParallelModel(Ψ::Vector{<:Complex}, T::Real, D::Real, L::Real, N::Int,
    B::GaussianRandomField; initialize::Bool=true)
    x₁(t::Real)::Tuple{Real,Real} = (L / T * t, 0)
    x₂(t::Real)::Tuple{Real,Real} = (L / T * t, D)
    return TwoSpinModel(Ψ, T, N, B, x₁, x₂, initialize=initialize)
end

function TwoSpinParallelModel(T::Real, D::Real, L::Real, N::Int, B::GaussianRandomField; initialize::Bool=true)
    return TwoSpinParallelModel(1 / √2 .* [0, 1+0im, -1+0im, 0], T, D, L, N, B; initialize=initialize)
end

"""

# Arguments
- `i::Int`: index of qubits, range from (1,n)
- `p::Int`: index of spin state, range from (1,2^n)
- `n::Int`: number of spins
"""
function m(i::Int, p::Int, n::Int)
    1 / 2 - digits(p, base=2, pad=n)[i]
end

"""
Calculate the dephasing matrix of a given spin shuttling model.

# Arguments
- `model::ShuttlingModel`: The spin shuttling model
- `method::Symbol`: The method to calculate the dephasing factor, `:trapezoid`, `:simpson`, `:adaptive`

# Returns
- `W::Matrix{<:Real}`: The dephasing matrix

# Example
```julia
model = OneSpinModel(1.0, 1.0, 100, OrnsteinUhlenbeckField([1.0, 1.0, 1.0]), t->t)
W = dephasingmatrix(model)
```
"""
function dephasingmatrix(model::ShuttlingModel; method::Symbol=:simpson)::Matrix{<:Real}
    n = model.n
    W = zeros(2^n, 2^n)
    for j in 1:2^n
        W[j, j] = 1
        for k in 1:j-1
            c = [trunc(Int, m(i, j - 1, n) - m(i, k - 1, n)) for i in 1:n]
            W[j, k] = dephasingfactor(model, c, method=method)
            W[k, j] = W[j, k]
        end
    end
    return W
end

"""
Calculate the dephasingfactor according to a special combinator of the noise sequence.

# Arguments
- `model::ShuttlingModel`: The spin shuttling model
- `c::Vector{Int}`: The combinator of the noise sequence, which should have the same length as the number of spins.
- `method::Symbol`: The method to calculate the characteristic value, `:trapezoid`, `:simpson`, `:adaptive`

# Returns
- `Real`: The dephasing factor

# Example
```julia
model = OneSpinModel(1.0, 1.0, 100, OrnsteinUhlenbeckField([1.0, 1.0, 1.0]), t->t)
c = [1, 1, 1]
dephasingfactor(model, c)
```
"""
function dephasingfactor(model::ShuttlingModel, c::Vector{Int}; method::Symbol=:simpson)::Real
    # rewrite the function to use the hcubature methods
    @assert method in [:trapezoid, :simpson, :adaptive]
    if method==:adaptive
        M = zeros(model.n, model.n)
        @threads for i in 1:model.n
            for j in 1:model.n
                K(t::AbstractVector)= covariance((t[1], model.X[i](t[1])...), (t[2], model.X[j](t[2])...), model.B)
                M[i, j] = hcubature(K, [0.0, 0.0], [model.T, model.T], rtol=1e-5)[1]
            end
        end
        return exp.(-sum((c * c').*M)/2)
    else
        R = CompositeRandomFunction(model.R, c, initialize=false)
        return characteristicvalue(R, method=method)
    end
end

function dephasingcoeffs(n::Int)::Array{Real,3}
    M = zeros(2^n, 2^n, n)
    for j in 1:2^n
        for k in 1:2^n
            c = [m(i, j - 1, n) - m(i, k - 1, n) for i in 1:n]
            M[j, k, :] = c
        end
    end
    return M
end


"""
Calculate the state fidelity of a spin shuttling model using numerical integration
of the covariance matrix.

# Arguments
- `model::ShuttlingModel`: The spin shuttling model
- `method::Symbol`: The method to calculate the dephasing factor, `:trapezoid`, `:simpson`, `:adaptive`

# Returns
- `Real`: The state fidelity
"""
function statefidelity(model::ShuttlingModel; method::Symbol=:simpson)::Real
    Ψ= model.Ψ
    w=dephasingmatrix(model, method=method)
    ρt=w.*(Ψ*Ψ')
    return Ψ'*ρt*Ψ
end


"""
Sampling an observable that defines on a specific spin shuttling model 

# Arguments
- `model::ShuttlingModel`: The spin shuttling model
- `objective::Function`: The objective function `objective(mode::ShuttlingModel; randseq)``
- `M::Int`: Monte-Carlo sampling size
"""
function sampling(model::ShuttlingModel, objective::Function, M::Int; isarray::Bool=false, isparallel::Bool=true)
    N=model.n * model.N
    samplingfunction()::Union{Number,VecOrMat{<:Number}} = objective(model, randn(N); isarray=isarray)
    if isparallel
        return parallelsampling(samplingfunction, M)
    else
        return serialsampling(samplingfunction, M)
    end
end


"""
Sample the state fidelity of a spin shuttling model using Monte-Carlo sampling.

# Arguments
- `model::ShuttlingModel`: The spin shuttling model
- `randseq::Vector{<:Real}`: The random sequence
- `isarray::Bool`: Return the dephasing matrix array for each time step
"""
function statefidelity(model::ShuttlingModel, randseq::Vector{<:Real}; isarray=false)::Union{Real,Vector{<:Real}}
    w=dephasingmatrix(model, randseq; isarray=isarray)
    ψ = model.Ψ
    f= w->real(ψ'*(w.*(ψ*ψ'))*ψ)
    return isarray ? vec(mapslices(f, w, dims=[1,2])) : f(w)
end


"""
Sample the dephasing matrix array for a given normal random vector.

# Arguments
- `model::ShuttlingModel`: The spin shuttling model
- `randseq::Vector{<:Real}`: The random sequence
- `isarray::Bool`: Return the dephasing matrix array for each time step
"""
function dephasingmatrix(model::ShuttlingModel, randseq::Vector{<:Real}; isarray=false)::Array{<:Complex}
    # model.R || error("covariance matrix is not initialized")
    N = model.N
    n=model.n
    dt = model.T / N
    noises = model.R(randseq)
    B=[noises[(i-1)*N+1:i*N] for i in 1:n]
    c=dephasingcoeffs(n)
    if isarray
        W=ones(Complex, 2^n,2^n, N)
        for j in 1:2^n
            for k in 1:j-1
                B_eff=sum(c[j,k,:].*B) 
                W[j, k, :] = exp.(im*cumsum(B_eff)*dt)
                W[k, j, :] = W[j, k, :]'
            end
        end
    else
        W=ones(Complex, 2^n,2^n)
        for j in 1:2^n
            for k in 1:j-1
                B_eff=sum(c[j,k,:].*B) 
                W[j, k] = exp(im*sum(B_eff)*dt)
                W[k, j] = W[j, k]'
            end
        end
    end
    return W
end


"""
Restrict the random field along a parameteized curve. 
# Arguments
- `X::Vector{<:Function}`: a vector of functions [x₁(t), x₂(t), ...]
- `t::AbstractArray`: time array
- `B::GaussianRandomField`: a Gaussian random field
- `initialize::Bool`: initialize the random function

# Returns
- `RandomFunction`: a new random function restricted along the curve

"""
function restriction(X::Vector{<:Function},t::AbstractArray,B::GaussianRandomField; initialize::Bool=true)::RandomFunction 
    f(x::Function, t::Real) = (t, x(t)...)
    P = vcat([f.(x, t) for x in X]...)
    return RandomFunction(P, B, initialize=initialize)
end

"""
Analytical dephasing factor of a one-spin shuttling model.

# Arguments
- `T::Real`: Total time
- `L::Real`: Length of the path
- `B<:GaussianRandomField`: Noise field, Ornstein-Uhlenbeck or Pink-Lorentzian
- `path::Symbol`: Path of the shuttling model, `:straight` or `:forthback`
"""
function W(T::Real, L::Real, B::OrnsteinUhlenbeckField; path=:straight)::Real
    κₜ = B.θ[1]
    κₓ = B.θ[2]
    σ = B.σ
    β = κₜ * T
    γ = κₓ * L
    if path == :straight
        return exp(-σ^2 / (4 * κₜ * κₓ) / κₜ^2 * P1(β, γ) / 2)
    elseif path == :forthback
        β /= 2
        return exp(-σ^2 / (4 * κₜ * κₓ) / κₜ^2 * (P1(β, γ) + P4(β, γ)))
    else
        error("Path not recognized. Use :straight or :forthback for one-spin shuttling model.")
    end
end

function W(T::Real, L::Real, B::PinkLorentzianField; path=:straight)::Real
    γ = L * B.κ
    if path == :straight
        β = T .* B.γ
        return exp(-B.σ^2 * T^2 * F3(β, γ))
    elseif path == :forthback
        T/=2
        β = T .* B.γ 
        return exp(-B.σ^2 * T^2*(2*F3(β, γ)+F4(β, γ)))
    end
end


"""
Analytical dephasing factor of a sequenced two-spin EPR pair shuttling model.
"""
function W(T0::Real, T1::Real, L::Real, B::OrnsteinUhlenbeckField; path=:sequenced)::Real
    κₜ = B.θ[1]
    κₓ = B.θ[2]
    σ = B.σ
    τ = κₜ * T0
    β = κₜ * T1
    γ = κₓ * L
    if path == :sequenced
        return exp(-σ^2 / (4 * κₜ * κₓ) / κₜ^2 * (F1(β, γ, τ) - F2(β, γ, τ)))
    elseif path == :parallel
        return exp(-σ^2 / (8 *κₜ*κₓ*κₓ) / κₜ^2 *(1-exp(-κₓ*T1)) * P1(κₜ*T0, κₓ*L))
    else
        error("Path not recognized. Use :sequenced or :parallel for two-spin EPR pair shuttling model.")
    end
end

end