using Symbolics
using Base

σ_z = [1 0; 0 -1];
σ_i = [1 0; 0 1];

function noise_hamiltonian(n::Int)::AbstractMatrix{Num}
    @variables B[1:n]
    S=repeat([σ_i],n)
    S_z(i::Int)=(S[i]=σ_z;kron(S...))
    H=sum([B[i] * S_z(i) for i in 1:n])
    return H
end

function diagonal_coeffs(n::Int)::AbstractArray{Vector}
    @variables B[1:n]
    H = noise_hamiltonian(n)
    coeffs=Vector{Vector}(undef, 2^n)
    for i in 1:2^n
        coeffs[i]=[Symbolics.coeff(H[i,i], B[j]) for j in 1:n]
    end
    return coeffs
end

diagonal_coeffs(n::Int)=collect(Iterators.product(collect(repeat([[1,-1]],n))...))
