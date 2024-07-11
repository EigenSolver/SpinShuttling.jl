using Test
using SpinShuttling

@testset "test multithreading" begin
    T=4; L=10; σ = sqrt(2) / 20; N=501; κₜ=1;κₓ=1;
    
    B=OrnsteinUhlenbeckField(0,[κₜ,κₓ],σ)
    model=OneSpinModel(T,L,N,B)

    M = 100000; 
    F=sampling(model, statefidelity, M)
    println(F)
end
