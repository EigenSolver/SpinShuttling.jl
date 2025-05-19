# -------------------------------------------------------------
# A lightweight Julia implementation of the Wootters concurrence
# for an arbitrary two‑qubit (4×4) density matrix.
# -------------------------------------------------------------


"""
    concurrence(ρ) -> Float64

Compute the **Wootters concurrence** of a two‑qubit (4×4) density matrix `ρ`.

# Arguments
- `ρ::AbstractMatrix{<:Complex}`: Hermitian, positive‑semidefinite matrix with `size == (4,4)` and `tr(ρ) ≈ 1`.

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
function concurrence(ρ::AbstractMatrix{<:Complex})::Float64
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
