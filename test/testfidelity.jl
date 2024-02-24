@testset begin "test fidelity"
    T=400; L=10; σ = sqrt(2) / 20; M = 10000; N=301; κₜ=1/20;κₓ=1/0.1;
    v=L/T;
    
    B=OrnsteinUhlenbeckField(0,[κₜ,κₓ],σ)
    
    model=OneSpinModel(T, L, M, N, B)
    
    F_num_mc, F_num_mc_var=sampling(model, fidelity, vector=true)
    F_num_ni = [fidelity(OneSpinModel(t, t*v, M, N, B, instantiate=false)) 
    for t in range(0,T,N)] 
    @test F_num_mc isa Vector{<:Real}
end