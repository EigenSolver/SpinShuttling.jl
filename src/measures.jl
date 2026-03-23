"""
    concurrence(ρ) -> Float64

Compute the **Wootters concurrence** of a two‑qubit (4×4) density matrix `ρ`.

# Arguments
- `ρ::AbstractMatrix{<:Number}`: Hermitian, positive‑semidefinite matrix with `size == (4,4)` and `tr(ρ) ≈ 1`.

# Returns
- `Float64` in the closed interval **[0,1]**.

# Algorithm
1. Build the spin‑flip operator `Y ⊗ Y` (Kronecker product of Pauli‑Y).
2. Form spin-flip matrix ρ̃.
3. Compute the eigenvalues `λ_i` of `ρ * ρ̃`, take their square‑roots, sort descending.
4. Return `max(0, λ₁ − λ₂ − λ₃ − λ₄)`.

# Reference
W.K. Wootters, *Phys. Rev. Lett.* **80**, 2245 (1998).
"""
function concurrence(ρ::AbstractMatrix{<:Number})::Float64
    @assert size(ρ) == (4,4) "ρ must be a 4×4 matrix (two qubits)"

    # Pauli σ_y
    σy = [0 -im; im 0]
    YtY = kron(σy, σy)          # Y ⊗ Y

    ρ̃ = YtY * conj.(ρ) * YtY   # spin‑flipped state
    R  = ρ * ρ̃                 # intermediate product

    # Numerical eigenvalues can pick up tiny imaginary parts → take real part
    λ = real.(eigvals(R))
    λ = clamp.(λ, 0, Inf)       # clip negative numerical noise

    λ_sqrt = sort(sqrt.(λ); rev = true)
    C = max(0.0, λ_sqrt[1] - λ_sqrt[2] - λ_sqrt[3] - λ_sqrt[4])
    return C
end

# -------------------------------------------------------------
# Quick demo (uncomment to run):

# ψ = [1,0,0im,1] ./ sqrt(2)      # Bell state |Φ⁺⟩
# ρ = ψ * ψ'                      # density matrix
# concurrence(ρ)  # → 1.0
# -------------------------------------------------------------

"""
    vonneumannentropy(ρ::AbstractMatrix{<:Number})::Float64
    
Compute the **von Neumann entropy** of a quantum state represented by a density matrix `ρ`.
# Arguments
- `ρ::AbstractMatrix{<:Number}`: Hermitian, positive-semidefinite
    matrix representing the quantum state.
# Returns
- `Float64`: The von Neumann entropy, a non-negative real number.

"""
function vonneumannentropy(ρ::AbstractMatrix{<:Number})::Float64
    @assert ishermitian(ρ) "ρ must be Hermitian"
    λ = real.(eigvals(ρ))
    λ = clamp.(λ, 0, Inf)  # clip negative numerical noise
    return -sum(λ .* log.(λ .+ eps()))  # add eps to avoid log(0)
end




"""
returns the process fidelity of a quantum channel defined by a transfer matrix `Λ` and a target process `S`.

# Arguments
- `Λ::Matrix{<:Number}`: Transfer matrix of the quantum channel.
- `S::Matrix{<:Number}`: Target process matrix.
- `d::Int=0`: Dimension of the Hilbert space. If `d`
is `0`, it is inferred from the size of `Λ`.
# Returns
- `Float64`: The process fidelity, a real number in the closed interval **[0
"""
function processfidelity(Λ::Matrix{<:Number}, S::Matrix{<:Number})
    @assert size(Λ) == size(S)
    d2 = size(Λ, 1) 
    return tr(Λ'*S)/d2
end

"""

# Arguments
- `krausops::Vector{Matrix{<:Number}}`: Vector of Kraus operators, each of size `2^n × 2^n` for `n` qubits.
- `U::Matrix{<:Number}`: The target unitary matrix.
- `d::Int=0`: Dimension of the Hilbert space. If `d`
is `0`, it is inferred from the size of `U`.
# Returns
- `Float64`: The process fidelity, a real number in the closed interval **[0
"""
function processfidelity(krausops::Vector{Matrix{<:Number}}, U::Matrix{<:Number})
    Λ = paulitransfermatrix(krausops; normalized=true)
    S = paulitransfermatrix(U; normalized=true)
    return processfidelity(Λ, S)
end


"""

[1] L. H. Pedersen, K. Molmer, and N. M. Moller, Fidelity of quantum operations, Physics Letters A 367, 47 (2007).

Calculate the average gate fidelity between two unitary operations.
# Arguments
- `U0::Matrix{<:Number}`: The first unitary operation,
    represented as a matrix of size `2^n × 2^n` for `n` qubits.
- `U1::Matrix{<:Number}`: The second unitary operation,
    represented as a matrix of size `2^n × 2^n` for `n` qubits.
# Returns
- `Real`: The average gate fidelity between the two unitary operations.
"""
function averagegatefidelity(U0::Matrix{<:Number}, U1::Matrix{<:Number})::Real
    n=size(U0, 1)
    M=U0'*U1
    return (tr(M*M')+abs(tr(M))^2) / (n*(n+1))
end

"""
Calculate the average gate fidelity between a quantum channel defined by Kraus operators and a unitary operation.
# Arguments
- `krausops::Vector{<:Matrix{<:Number}}`: Vector
    of Kraus operators, each of size `2^n × 2^n` for `n` qubits.
- `U::Matrix{<:Number}`: The unitary operation, represented as a matrix of size `2^n × 2^n` for `n` qubits.
# Returns
- `Real`: The average gate fidelity between the quantum channel and the unitary operation.
"""
function averagegatefidelity(krausops::Vector{<:Matrix{<:Number}}, U::Matrix{<:Number})::Real
    n=size(U, 1)
    f=0.0
    for V in krausops
        M=U'*V
        f+=tr(M*M')+abs(tr(M))^2
    end
    return  f / (n*(n+1))
end

