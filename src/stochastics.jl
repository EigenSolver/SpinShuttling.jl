abstract type GaussianRandomField end # n-dimensional

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
Generate a time series on a given time array subject to 
a Gaussian random process traced from a Gaussian random field.
"""
struct RandomFunction
    μ::Union{Real, Vector{<:Real}}
    P::Matrix{<:Real} # sample trace
    Σ::Symmetric{<:Real} # covariance matrix
    C::Cholesky # decomposition
    function RandomFunction(P::Matrix{<:Real}, field::GaussianRandomField)
        μ=field.μ isa Function ? field.μ(P) : field.μ
        Σ=covariancematrix(P, field)
        return new(μ, P, Σ, cholesky(Σ))
    end
end

"""
Generate a time series subject to a Gaussian random process using a given random seed.
"""
function (R::RandomFunction)(randseq::Vector{<:Real})
    return R.μ .+ R.C.L*randseq
end

(R::RandomFunction)()=R(randn(size(R.Σ, 1)))

"""
Covariance function between two points of a Gaussian random field.
# Arguments
- `p₁::Vector{<:Real}`: time-position vector
- `p₂::Vector{<:Real}`: time-position vector
- `field<:GaussianRandomField`: a Gaussian random field
"""
function covariance(p₁::Vector{<:Real}, p₂::Vector{<:Real}, field::OrnsteinUhlenbeckField)::Real
    field.σ^2 / prod(2 * field.θ) * exp(-dot(field.θ, abs.(p₁ - p₂)))
end

function covariance(p₁::Vector{<:Real}, p₂::Vector{<:Real}, field::PinkBrownianField)::Real
    t₁ = p₁[1]
    t₂ = p₂[1]
    x₁ = p₁[2:end]
    x₂ = p₂[2:end]
    cov_pink = (expinti(-γ[2]abs(t₁ - t₂)) - expinti(-γ[1]abs(t₁ - t₂)))/log(γ[2]/γ[1])
    cov_brown = exp(-dot(field.θ, abs.(x₁ - x₂)))
    return field.σ^2 * cov_pink * cov_brown
end

"""
Covariance matrix of a Gaussian random field. 
When `P₁=P₂`, it is the auto-covariance matrix of a Gaussian random process. 
When `P₁!=P₂`, it is the cross-covariance matrix between two Gaussian random processes.
# Arguments
- `P₁::Matrix{<:Real}`: time-position array
- `P₂::Matrix{<:Real}`: time-position array
- `field::GaussianRandomField`: a Gaussian random field
"""
function covariancematrix(P₁::Matrix{<:Real}, P₂::Matrix{<:Real}, field::GaussianRandomField)::Matrix{Real}
    @assert size(P₁) == size(P₂)
    N = size(P₁, 1)
    P₁ = P₁'; P₂ = P₂';
    A = Matrix{Real}(undef, N, N)
    for i in 1:N
        for j in 1:N
            A[i, j] = covariance(P₁[:, i], P₂[:, j], field)
        end
    end
    return A
end

"""
Auto-Covariance matrix of a Gaussian random process.
"""
function covariancematrix(P::Matrix{<:Real}, field::GaussianRandomField)::Symmetric
    N = size(P, 1)
    P = P'
    A = Matrix{Real}(undef, N, N)
    for i in 1:N
        for j in i:N
            A[i, j] = covariance(P[:, i], P[:, j], field)
        end
    end
    return Symmetric(A)
end

"""
Compute the characteristic functional of the process from the 
numerical quadrature of the covariance matrix.
Using Simpson's rule by default.
"""
function characteristicfunction(R::RandomFunction)::Tuple{Vector{<:Real},Vector{<:Real}}
    dt=R.P[2,1]-R.P[1,1]
    N=size(R.Σ,1)
    @assert N%2==1
    χ(j::Int)=exp.(-integrate((@view R.Σ[1:j,1:j]), dt, dt)/2)
    t=R.P[2:2:N-1,1]
    f=[χ(j) for j in 3:2:N]
    return (t,f)
end

"""
Compute the final phase of the characteristic functional of the process from the 
numerical quadrature of the covariance matrix.
Using Simpson's rule by default.
"""
function characteristicvalue(R::RandomFunction)::Real
    dt=R.P[2,1]-R.P[1,1]
    return exp.(- integrate(R.Σ, dt, dt)/2)
end
