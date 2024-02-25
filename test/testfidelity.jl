using Plots
##
figsize=(400, 300)
visualize=false

@testset begin "test fidelity"
    T=400; L=10; σ = sqrt(2) / 20; M = 10000; N=301; κₜ=1/20;κₓ=1/0.1;
    v=L/T;
    
    B=OrnsteinUhlenbeckField(0,[κₜ,κₓ],σ)
    
    model=OneSpinModel(T, L, M, N, B)
    
    F_num_mc, F_num_mc_var=sampling(model, fidelity, vector=true)
    @test F_num_mc isa Vector{<:Real}
    F_num_ni = [averagefidelity(OneSpinModel(t, t*v, M, N, B)) 
    for t in range(0,T,10)] 
    @test F_num_mc isa Vector{<:Real}
end

##
@testset begin "test single spin characteristics"
    T=400; L=10; σ = sqrt(2) / 20; M = 10000; N=601; κₜ=1/20;κₓ=1/0.1;
    v=L/T;
    t=range(0, T, N)
    P=hcat(t, v.*t)
    B=OrnsteinUhlenbeckField(0,[κₜ,κₓ],σ)
    # R=RandomFunction(P , B)
    model=OneSpinModel(T,L,M,N,B)
    f1=averagefidelity(model)
    f2, f2_err=sampling(model, fidelity)
    f3=1/2*(1+Χ(T,L,B))
    @test isapprox(f1, f3,rtol=1e-2)
    @test isapprox(f2, f3, rtol=1e-2) 
    println("NI:", f1)
    println("MC:", f2)
    println("TH:", f3)
end

##
@testset begin "test two spin characteristics"
    L=10; σ =sqrt(2)/20; M=10000; N=501; T1=200; T0=25*0.05; κₜ=1/20; κₓ=1/0.1;
    B=OrnsteinUhlenbeckField(0,[κₜ,κₓ],σ)
    model=TwoSpinModel(T0, T1, L, M, N, B)
    if visualize
        display(heatmap(collect(model.R.Σ), title="cross covariance matrix, test fig 4", size=figsize))
    end
    f1=averagefidelity(model)
    f2, f2_err=sampling(model, fidelity)
    f3=1/2*(1+Χ(T0, T1, L,B))
    @test isapprox(f1, f3,rtol=1e-2)
    @test isapprox(f2, f3, rtol=1e-2) 
    println("NI:", f1)
    println("MC:", f2)
    println("TH:", f3)
end

