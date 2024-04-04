abstract type RandomField end
abstract type GaussianRandomField <: RandomField end # n-dimensional

"""
Ornstein-Uhlenbeck field, the correlation function of which is 
`σ^2 * exp(-|t₁ - t₂|/θ_t) * exp(-|x₁-x₂|/θ_x)` 
where `t` is time and `x` is position.
"""
struct OrnsteinUhlenbeckField <: GaussianRandomField
    μ::Union{<:Real,Function} # mean
    θ::Vector{<:Real}
    σ::Real # covariance
end

"""
Pink-Brownian Field, the correlation function of which is
`σ^2 * (expinti(-γ[2]abs(t₁ - t₂)) - expinti(-γ[1]abs(t₁ - t₂)))/log(γ[2]/γ[1]) * exp(-|x₁-x₂|/θ)`
where `expinti` is the exponential integral function.
"""
struct PinkBrownianField <: GaussianRandomField
    μ::Union{<:Real,Function}  # mean
    θ::Vector{<:Real}
    σ::Real
    γ::Tuple{<:Real,<:Real} # cutoffs of 1/f 
end

"""
Similar type of `RandomFunction` in Mathematica.
Can be used to generate a time series on a given time array subject to 
a Gaussian random process traced from a Gaussian random field.

# Arguments
- `μ::Vector{<:Real}`: mean of the process
- `P::Array{<:Real}`: time-position array
- `Σ::Symmetric{<:Real}`: covariance matrices
- `C::Cholesky`: cholesky decomposition of the covariance matrices
"""
struct RandomFunction
    μ::Vector{<:Real}
    P::Array{<:Real} # sample trace
    Σ::Symmetric{<:Real} # covariance matrices
    C::Cholesky # decomposition
    function RandomFunction(P::Matrix{<:Real}, process::GaussianRandomField)
        μ=process.μ isa Function ? process.μ(P) : repeat([process.μ], size(P, 1))
        Σ=covariancematrix(P, process)
        return new(μ, P, Σ, cholesky(Σ))
    end

    function RandomFunction(μ::Vector{<:Real}, P::Array{<:Real}, Σ::Symmetric{<:Real}, C::Cholesky)
        new(μ, P, Σ, C)
    end
end


"""
Divide the covariance matrix of a direct summed random function into partitions. 
"""
function covariancepartition(R::RandomFunction, n::Int)::Matrix{Matrix{<:Real}}
    Λ=Matrix{Matrix{<:Real}}(undef, n, n)
    N=size(R.P,1)÷n
    Σ(i::Int,j::Int) = R.Σ[(i-1)*N+1: i*N , (j-1)*N+1: j*N]
    for i in 1:n
        for j in 1:n
            Λ[i,j]=Σ(i,j)
        end
    end
    return Λ
end

"""

"""
function meanpartition(R::RandomFunction, n::Int)::Vector{Vector{<:Real}}
    N=size(R.P,1)÷n
    return [R.μ[(i-1)*N+1: i*N] for i in 1:n]
end

"""
Create a new random function composed by a linear combination of random processes.
The input random function represents the direct sum of these processes. 
The output random function is a tensor contraction from the input.
"""
function CompositeRandomFunction(R::RandomFunction, c::Vector{Int})::RandomFunction
    n=length(c)
    N=size(R.Σ,1)
    μ = sum(c .*meanpartition(R, n))
    Σ = Symmetric(sum((c*c') .* covariancepartition(R, n)))
    return RandomFunction(μ, R.P[1:(N÷n),1], Σ, cholesky(Σ))
end

function CompositeRandomFunction(P::Vector{Matrix{Real}}, process::GaussianRandomField, c::Vector{Int})::RandomFunction
    return CompositeRandomFunction(RandomFunction(P, process), c)
end


function (R::RandomFunction)(randseq::Vector{<:Real})
    return R.μ .+ R.C.L*randseq
end

(R::RandomFunction)()=R(randn(size(R.Σ, 1)))


"""
Covariance function of Ornstein-Uhlenbeck process.
"""
function covariance(p₁::Vector{<:Real}, p₂::Vector{<:Real}, process::OrnsteinUhlenbeckField)::Real
    process.σ^2 / prod(2 * process.θ) * exp(-dot(process.θ, abs.(p₁ - p₂)))
end

"""
Covariance function of Pink-Brownian process.
"""
function covariance(p₁::Vector{<:Real}, p₂::Vector{<:Real}, process::PinkBrownianField)::Real
    t₁ = p₁[1]
    t₂ = p₂[1]
    x₁ = p₁[2:end]
    x₂ = p₂[2:end]
    γ = process.γ
    cov_pink = t₁ != t₂ ? (expinti(-γ[2]abs(t₁ - t₂)) - expinti(-γ[1]abs(t₁ - t₂)))/log(γ[2]/γ[1]) : 1
    cov_brown = exp(-dot(process.θ, abs.(x₁ - x₂)))
    return process.σ^2 * cov_pink * cov_brown
end

"""
Covariance matrix of a Gaussian random field. 
When `P₁=P₂`, it is the auto-covariance matrix of a Gaussian random process. 
When `P₁!=P₂`, it is the cross-covariance matrix between two Gaussian random processes.
# Arguments
- `P₁::Matrix{<:Real}`: time-position array
- `P₂::Matrix{<:Real}`: time-position array
- `process::GaussianRandomField`: a Gaussian random field
"""
function covariancematrix(P₁::Matrix{<:Real}, P₂::Matrix{<:Real}, process::GaussianRandomField)::Matrix{Real}
    @assert size(P₁) == size(P₂)
    N = size(P₁, 1)
    P₁ = P₁'; P₂ = P₂';
    A = Matrix{Real}(undef, N, N)
    Threads.@threads for i in 1:N
        for j in 1:N
            A[i, j] = covariance(P₁[:, i], P₂[:, j], process)
        end
    end
    return A
end

"""
Auto-Covariance matrix of a Gaussian random process.
"""
function covariancematrix(P::Matrix{<:Real}, process::GaussianRandomField)::Symmetric
    N = size(P, 1)
    P = P'
    A = Matrix{Real}(undef, N, N)
    Threads.@threads for i in 1:N
        for j in i:N
            A[i, j] = covariance(P[:, i], P[:, j], process)
        end
    end
    return Symmetric(A)
end

"""
Compute the characteristic functional of the process from the 
numerical quadrature of the covariance matrix.
Using Simpson's rule by default.
"""
function characteristicfunction(R::RandomFunction)::Tuple{Vector{<:Real},Vector{<:Number}}
    # need further optimization
    dt=R.P[2,1]-R.P[1,1]
    N=size(R.Σ,1)
    @assert N%2==1
    χ(j::Int)=exp.(1im*integrate(view(R.μ, 1:j), dt))*exp.(-integrate(view(R.Σ, 1:j,1:j), dt, dt)/2)
    t=R.P[2:2:N-1,1]
    f=[χ(j) for j in 3:2:N] # only for simpson's rule
    return (t,f)
end

"""
Compute the final phase of the characteristic functional of the process from the 
numerical quadrature of the covariance matrix.
Using Simpson's rule by default.
"""
function characteristicvalue(R::RandomFunction)::Number
    dt=R.P[2,1]-R.P[1,1]
    return exp.(1im*integrate(R.μ, dt))*exp.(-integrate((@view R.Σ[:,:]), dt, dt)/2)
end
