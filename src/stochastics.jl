abstract type GaussianRandomField end # n-dimensional

struct OrnsteinUhlenbeckField <: GaussianRandomField
    μ::Union{<:Real,Function} # mean
    θ::Vector{<:Real}
    σ::Real # covariance
end

struct PinkBrownianField <: GaussianRandomField
    μ::Union{<:Real,Function}  # mean
    θ::Vector{<:Real}
    σ::Real
    γ::Tuple{<:Real,<:Real} # cutoffs of 1/f 
end

"""
Similar type of `RandomFunction` in Mathematica.
Generate a time series on a given time array subject to a Gaussian random process.
"""
struct RandomFunction
    μ::Union{Real, Vector{<:Real}}
    P::Matrix{<:Real} # sample trace
    Σ::Symmetric{<:Real} # covariance matrix
    C::Cholesky # decomposition
    function RandomFunction(P::Matrix{<:Real}, process::GaussianRandomField)
        μ=process.μ isa Function ? process.μ(P) : process.μ
        Σ=covariancematrix(P, process)
        return new(μ, P, Σ, cholesky(Σ))
    end
end


"""

"""
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
    cov_pink = (expinti(-γ[2]abs(t₁ - t₂)) - expinti(-γ[1]abs(t₁ - t₂)))/log(γ[2]/γ[1])
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
    for i in 1:N
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
    for i in 1:N
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


struct LaTeXEquation
    content::String
end

function Base.show(io::IO, ::MIME"text/latex", x::LaTeXEquation)
    # Wrap in $$ for display math printing
    return print(io, "\$\$ " * x.content * " \$\$")
end

LaTeXEquation(raw"""
    \left[\begin{array}{c}
        x \\
        y
    \end{array}\right]
""")