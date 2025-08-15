using Base.Iterators
σx = [0 1; 1 0]
σy = [0 -im; im 0]
σz = [1 0; 0 -1]
σi = [1 0; 0 1]

"""
paulitransfermatrix(KrausOps::Vector{<:Matrix{<:Complex}}; normalized::Bool=false)
Compute the **Pauli transfer matrix** for a set of Kraus operators.
# Arguments
- `KrausOps::Vector{<:Matrix{<:Complex}}`: Vector
    of Kraus operators, each of size `2^n × 2^n` for `n` qubits.
- `normalized::Bool=false`: If `true`, the transfer matrix is normalized by the number of Kraus operators.
# Returns
- `Matrix{Float64}`: The Pauli transfer matrix of size `4^n × 4^n`.
"""
function paulitransfermatrix(KrausOps::Vector{<:Matrix{<:Complex}}; normalized::Bool=false)
    M=length(KrausOps)
    @assert log(4, N) % 1 ==0
    d = size(KrausOps[1])[1] # dimension of the basis
    n = Int(log(2,d))
    pauli_basis=[σI, σx, σy, σz]

    n_pauli_basis = Vector{Matrix{Complex{Float64}}}(undef, 4^n)
    for (j,tpl) in enumerate(product(ntuple(_ -> pauli_basis, n)...))
        n_pauli_basis[j]=kron(tpl...)
    end

    return processtomography(KrausOps, n_pauli_basis; normalized=normalized)
end

"""
processtomography(KrausOps::Vector{<:Matrix{<:Complex}}, basis::Vector{<:Matrix{<:Complex}}; normalized::Bool=false)
Compute the **transfer matrix** for a quantum channel defined by a set of Kraus operators.
# Arguments
- `KrausOps::Vector{<:Matrix{<:Complex}}`: Vector
    of Kraus operators, each of size `2^n × 2^n` for `n` qubits.
- `basis::Vector{<:Matrix{<:Complex}}`: Vector
    of basis matrices, each of size `2^n × 2^n` for `n` qubits.
- `normalized::Bool=false`: If `true`, the transfer matrix is normalized by the number of Kraus operators.
# Returns
- `Matrix{Float64}`: The process tomography transfer matrix of size `N × N
"""
function processtomography(KrausOps::Vector{<:Matrix{<:Complex}}, basis::Vector{<:Matrix{<:Complex}}; normalized::Bool=false)
    N=length(basis)
    M=length(KrausOps)
    @assert log(4, N) % 1 ==0
    d = tr(basis[1]) # dimension of the basis

    TM=zeros(Real, N, N) # transfer matrix
    for j in 1:N # 4^n
        for k in 1:N # 4^n
            ρ=zeros(Complex, size(basis[j]))
            for E in KrausOps # M samples
                ρ+=E*basis[j]*E'
            end
            TM[j,k]=real(tr(basis[k]*ρ)/d)
        end
    end
    if !normalized
        TM = TM ./ M
    end
    return TM 
end
