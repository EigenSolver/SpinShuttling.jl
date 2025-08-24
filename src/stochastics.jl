abstract type RandomField end
abstract type GaussianRandomField <: RandomField end 
abstract type PoissonRandomField <: RandomField end 

abstract type RandomVectorField end
abstract type GaussianRandomVectorField <: RandomVectorField end


Point{N} = Tuple{Vararg{Real,N}}

"""
Ornstein-Uhlenbeck field, the correlation function of which is 
`σ^2 * exp(-|t₁ - t₂|/θₜ) * exp(-|x₁-x₂|/θₓ)` 
where `t` is time and `x` is position.
"""
struct OrnsteinUhlenbeckField <: GaussianRandomField
    μ::Union{<:Real,Function} # mean
    θ::Vector{<:Real}
    σ::Real # covariance
end


"""
Pink-Lorentzian Field, the correlation function of which is
`σ^2 * (expinti(-γ[2]abs(t₁ - t₂)) - expinti(-γ[1]abs(t₁ - t₂)))/log(γ[2]/γ[1]) * exp(-κ|x₁-x₂|)`
where `expinti` is the exponential integral function.
"""
struct PinkLorentzianField <: GaussianRandomField
    μ::Union{<:Real,Function}  # mean
    κ::Real
    σ::Real
    γ::Tuple{<:Real,<:Real} # cutoffs of 1/f 
end


"""
Pink-HeavisidePi Field, the correlation function of which is
`σ^2 * (expinti(-γ[2]abs(t₁ - t₂)) - expinti(-γ[1]abs(t₁ - t₂)))/log(γ[2]/γ[1]) * sinc(-κ|x₁-x₂|)`
where `expinti` is the exponential integral function.
"""
struct PinkPiField <: GaussianRandomField
    μ::Union{<:Real,Function}  # mean
    κ::Real
    σ::Real
    γ::Tuple{<:Real,<:Real} # cutoffs of 1/f 
end

mutable struct GaussianRandomFunction 
    μ::Vector{<:Real}
    P::Vector{<:Point} # sample trace
    Σ::Symmetric{<:Real} # covariance matrices
    L::Union{Matrix{<:Real},Nothing} # Lower triangle matrix of Cholesky decomposition
    """
    Create a new `GaussianRandomFunction` instance with precomputed values.

    # Arguments
    - `μ::Vector{<:Real}`: Mean vector.
    - `P::Vector{<:Point}`: Sample trace.
    - `Σ::Symmetric{<:Real}`: Covariance matrix.
    - `L::Matrix{<:Real}`: Lower triangle matrix of Cholesky decomposition.

    # Returns
    A new `GaussianRandomFunction` instance.
    """
    function GaussianRandomFunction(μ::Vector{<:Real}, P::Vector{<:Point}, Σ::Symmetric{<:Real}, L::Union{Matrix{<:Real},Nothing})
        new(μ, P, Σ, L)
    end

    """
    Create a new `GaussianRandomFunction` instance.

    # Arguments
    - `P::Vector{<:Point}`: Sample trace.
    - `process::GaussianRandomField`: The Gaussian random field process.
    - `initialize::Bool=true`: Whether to initialize the Cholesky decomposition of the covariance matrix.

    # Returns
    A new `GaussianRandomFunction` instance.
    """
    function GaussianRandomFunction(P::Vector{<:Point}, process::GaussianRandomField; initialize::Bool=true)
        μ = process.μ isa Function ? process.μ(P) : repeat([process.μ], length(P))
        Σ = covariancematrix(P, process)
        R = GaussianRandomFunction(μ, P, Σ, nothing)
        initialize ? initialize!(R) : missing
        return R
    end

end

"""
Initialize the Cholesky decomposition of the covariance matrix of a random function.
"""
function initialize!(R::GaussianRandomFunction)
    try
        R.L = collect(cholesky(R.Σ).L)
    catch
        R.L = collect(cholesky(R.Σ, RowMaximum(), check=false).L)
    end
end

"""
Divide the covariance matrix of a direct summed random function into partitions. 

# Arguments
- `R::GaussianRandomFunction`: a direct sum of random processes R₁⊕ R₂⊕ ... ⊕ Rₙ
- `n::Int`: number of partitions or spins

# Returns
- `Matrix{Matrix{<:Real}}`: a matrix of covariance matrices
"""
function covariancepartition(R::GaussianRandomFunction, n::Int)::Matrix{Matrix{<:Real}}
    Λ = Matrix{Matrix{<:Real}}(undef, n, n)
    N = length(R.P) ÷ n
    Σ(i::Int, j::Int) = R.Σ[(i-1)*N+1:i*N, (j-1)*N+1:j*N]
    for i in 1:n
        for j in 1:n
            Λ[i, j] = Σ(i, j)
        end
    end
    return Λ
end

"""

# Arguments
- `R::GaussianRandomFunction`: a direct sum of random processes R₁⊕ R₂⊕ ... ⊕ Rₙ
- `n::Int`: number of partitions or spins

# Returns
- `Vector{Vector{<:Real}}`: a vector of mean vectors
"""
function meanpartition(R::GaussianRandomFunction, n::Int)::Vector{Vector{<:Real}}
    N = length(R.P) ÷ n
    return [R.μ[(i-1)*N+1:i*N] for i in 1:n]
end

"""
Create a new random function composed by a linear combination of random processes.
The input random function represents the direct sum of these processes. 
The output random function is a tensor contraction from the input.

# Arguments
- `R::GaussianRandomFunction`: a direct sum of random processes R₁⊕ R₂⊕ ... ⊕ Rₙ
- `c::Vector{Int}`: a vector of coefficients
- `initialize::Bool=true`: whether to initialize the Cholesky decomposition of the covariance matrix

# Returns
- `GaussianRandomFunction`: a new random function composed by a linear combination of random processes
"""
function CompositeGaussianRandomFunction(R::GaussianRandomFunction, c::Vector{Int}; initialize::Bool=false)::GaussianRandomFunction
    n = length(c)
    N = size(R.Σ, 1)
    μ = sum(c .* meanpartition(R, n))
    Σ = Symmetric(sum((c * c') .* covariancepartition(R, n)))
    t = [(p[1],) for p in R.P[1:(N÷n)]]
    R = GaussianRandomFunction(μ, t, Σ, nothing)
    initialize ? initialize!(R) : missing
    return R
end


function CompositeGaussianRandomFunction(P::Vector{<:Point}, process::GaussianRandomField, c::Vector{Int})::GaussianRandomFunction
    return CompositeGaussianRandomFunction(GaussianRandomFunction(P, process), c)
end

"""
Generate a random time series from a Gaussian random field.

`R()` generates a random time series from a Gaussian random field `R`
`R(randseq)` generates a random time series from a Gaussian random field `R` with a given random sequence `randseq`.
"""
function (R::GaussianRandomFunction)(randseq::Vector{<:Real})
    return R.μ .+ R.L * randseq
end

(R::GaussianRandomFunction)() = R(randn(size(R.Σ, 1)))


"""
Covariance function of Gaussian random field.

# Arguments
- `p₁::Point`: time-position array
- `p₂::Point`: time-position array
- `process<:GaussianRandomField`: a Gaussian random field, e.g. `OrnsteinUhlenbeckField` or `PinkLorentzianField`
"""
function covariance(p₁::Point, p₂::Point, process::OrnsteinUhlenbeckField)::Real
    process.σ^2 / prod(2 * process.θ) * exp(-dot(process.θ, abs.(p₁ .- p₂)))
end

function covariance(p₁::Point, p₂::Point, process::PinkLorentzianField)::Real
    t₁ = p₁[1]
    t₂ = p₂[1]
    x₁ = p₁[2:end]
    x₂ = p₂[2:end]
    γ = process.γ
    cov_log = t₁ != t₂ ? (expinti(-γ[2]abs(t₁ - t₂)) - expinti(-γ[1]abs(t₁ - t₂))) / log(γ[2] / γ[1]) : 1
    cov_exp = exp(-process.κ * norm(x₁ .- x₂))
    return process.σ^2 * cov_log * cov_exp
end

function covariance(p₁::Point, p₂::Point, process::PinkPiField)::Real
    t₁ = p₁[1]
    t₂ = p₂[1]
    x₁ = p₁[2:end]
    x₂ = p₂[2:end]
    γ = process.γ
    cov_log = t₁ != t₂ ? (expinti(-γ[2]abs(t₁ - t₂)) - expinti(-γ[1]abs(t₁ - t₂))) / log(γ[2] / γ[1]) : 1
    cov_sinc = sinc(-process.κ * norm(x₁ .- x₂))
    return process.σ^2 * cov_log * cov_sinc
end

"""
Covariance matrix of a Gaussian random field. 
When `P₁=P₂`, it is the auto-covariance matrix of a Gaussian random process. 
When `P₁!=P₂`, it is the cross-covariance matrix between two Gaussian random processes.
# Arguments
- `P₁::Vector{<:Point}`: time-position array
- `P₂::Vector{<:Point}`: time-position array
- `process::GaussianRandomField`: a Gaussian random field
"""
function covariancematrix(P₁::Vector{<:Point}, P₂::Vector{<:Point}, process::GaussianRandomField)::Matrix{Real}
    @assert length(P₁) == length(P₂)
    N = length(P₁)
    A = Matrix{Real}(undef, N, N)
    @threads for i in 1:N
        for j in 1:N
            A[i, j] = covariance(P₁[i], P₂[j], process)
        end
    end
    return A
end

"""
Auto-Covariance matrix of a Gaussian random process.
# Arguments
- `P::Vector{<:Point}`: time-position array
- `process::GaussianRandomField`: a Gaussian random field

# Returns
- `Symmetric{Real}`: auto-covariance matrix

"""
function covariancematrix(P::Vector{<:Point}, process::GaussianRandomField)::Symmetric{<:Real}
    N = length(P)
    A = Matrix{Real}(undef, N, N)
    @threads for i in 1:N
        for j in i:N
            A[i, j] = covariance(P[i], P[j], process)
        end
    end
    return Symmetric(A)
end

"""
Compute the characteristic functional of the process from the 
numerical quadrature of the covariance matrix.
Using Simpson's rule by default.
"""
function characteristicfunction(R::GaussianRandomFunction; method::Symbol=:simpson)::Tuple{Vector{<:Real},Vector{<:Number}}
    # need further optimization
    dt = R.P[2][1] - R.P[1][1]
    N = size(R.Σ, 1)
    @assert N % 2 == 1
    χ(j::Int) = exp.(1im * integrate(view(R.μ, 1:j), dt, method=method)) * exp.(-integrate(view(R.Σ, 1:j, 1:j), dt, dt, method=method) / 2)
    t = [p[1] for p in R.P[2:2:N-1]]
    f = [χ(j) for j in 3:2:N] # only for simpson's rule
    return (t, f)
end

"""
Compute the final phase of the characteristic functional of the process from the 
numerical quadrature of the covariance matrix.
Using Simpson's rule by default.
"""
function characteristicvalue(R::GaussianRandomFunction; method::Symbol=:simpson)::Number
    dt = R.P[2][1] - R.P[1][1]
    f1 = exp.(1im * integrate(R.μ, dt, method=method))
    f2 = exp.(-integrate((@view R.Σ[:, :]), dt, dt, method=method) / 2)
    return f1 * f2
end