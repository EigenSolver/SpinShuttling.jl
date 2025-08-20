abstract type QuantumProcess end

"""

MixingUnitaryChannel <: QuantumProcess
A quantum process that applies a set of unitary operations with given probabilities.
# Fields
- `Us::Vector{Matrix{Complex{Float64}}}`: Vector of unitary
    matrices, each of size `2^n × 2^n` for `n` qubits.
- `Ps::Vector{Float64}`: Vector of mixing probabilities, which sum to 1

"""
struct MixingUnitaryChannel <: QuantumProcess
    Us::Vector{Matrix{Complex{Float64}}} # Unitary matrix of size `2^n × 2^n`
    Ps::Vector{Float64} # Mixing probabilities, sum to 1
end

"""
KrausOps(channel::MixingUnitaryChannel)::Vector{Matrix{Complex{Float64}}}
Convert a `MixingUnitaryChannel` to its Kraus operators.
# Arguments
- `channel::MixingUnitaryChannel`: The mixing unitary channel.
# Returns
- `Vector{Matrix{Complex{Float64}}}`: Vector of Kraus operators,
"""
function KrausOps(channel::MixingUnitaryChannel)
    M = length(channel.Us)
    @assert length(channel.Ps) == M
    KrausOps = Vector{Matrix{Complex{Float64}}}(undef, M)
    for i in 1:M
        KrausOps[i] = sqrt(channel.Ps[i]) * channel.Us[i]
    end
    return KrausOps
end

"""
E::MixingUnitaryChannel(ρ::AbstractMatrix{<:Complex})

# Arguments
- `E::MixingUnitaryChannel`: A mixing unitary channel.
- `ρ::AbstractMatrix{<:Complex}`: The density matrix to apply the channel
# Returns
- `AbstractMatrix{<:Complex}`: The resulting density matrix after applying the channel.
"""
function (E::MixingUnitaryChannel)(ρ::AbstractMatrix{<:Complex})
    M = length(E.Us)
    @assert size(ρ) == size(E.Us[1]) "ρ must match the size of the unitary matrices"
    result = zeros(Complex{Float64}, size(ρ))
    for i in 1:M
        result += E.Ps[i] * (E.Us[i] * ρ * E.Us[i]')
    end
    return result
end

"""

processfidelity(E::MixingUnitaryChannel, S::Matrix{<:Number}; d::Int=0)
Compute the process fidelity of a quantum channel defined by a `MixingUnitaryChannel` and
a target process `S`.
# Arguments
- `E::MixingUnitaryChannel`: The mixing unitary channel.
- `S::Matrix{<:Number}`: The target process matrix.
- `d::Int=0`: Dimension of the Hilbert space. If `d` is `0`, it is inferred from the size of `E`.
# Returns
- `Float64`: The process fidelity, a real number in the closed interval **[0
"""
function processfidelity(E::MixingUnitaryChannel, S::Matrix{<:Number}; d::Int=0)
    KrausOps = KrausOps(E)
    Λ = paulitransfermatrix(KrausOps; normalized=true)
    return processfidelity(Λ, S; d=d)
end
