@testset "test symbolics" begin
    for n in 2:4
        @variables B[1:n]
        H=noise_hamiltonian(n) 
        @test H isa AbstractMatrix{Num}
        @test all([simplify(-H[i,i]==H[i+2^(n-1),i+2^(n-1)]) == true for i in 1:2^(n-1)])
    end
end

@testset "test average unitary" begin
    ψ=1/√2*[1,0,0,-1]
    N=40000
    A(n) = Diagonal(1 .+ randn(n))
    f1=mean([ψ'*A(4)*ψ for i in 1:N])
    f2=ψ'*mean([A(4) for i in 1:N])*ψ
    
    @test isapprox(f1, 1, rtol=0.01)
    @test isapprox(f2, 1, rtol=0.01)
end