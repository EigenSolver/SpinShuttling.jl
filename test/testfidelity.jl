##
visualize=true

#
@testset begin "test single spin shuttling fidelity"
    T=400; L=10; σ = sqrt(2) / 20; M = 20000; N=601; κₜ=1/20;κₓ=1/0.1;
    v=L/T;
    t=range(0, T, N)
    P=collect(zip(t, v.*t))
    B=OrnsteinUhlenbeckField(0,[κₜ,κₓ],σ)

    model=OneSpinModel(T,L,N,B)
    # test customize println
    println(model)

    f1=averagefidelity(model)
    f2, f2_err=sampling(model, fidelity, M)
    f3=1/2*(1+W(T,L,B))
    @test isapprox(f1, f3,rtol=1e-2)
    @test isapprox(f2, f3, rtol=1e-2) 
    println("NI:", f1)
    println("MC:", f2)
    println("TH:", f3)
end

#
@testset begin "test single spin forth-back shuttling fidelity"
    T=200; L=10; σ = sqrt(2) / 20; M = 5000; N=501; κₜ=1/20;κₓ=10; 
    # exponential should be smaller than 100
    B=OrnsteinUhlenbeckField(0,[κₜ,κₓ],σ)

    if visualize
        t=range(1e-2*T,T, 10)
        f_mc=[sampling(OneSpinForthBackModel(T,L,N,B), fidelity, M)[1] for T in t]
        f_ni=[averagefidelity(OneSpinForthBackModel(T,L,N,B)) for T in t]
        f_th=[(1+W(T,L,B,path=:forthback))/2 for T in t]
        fig=lineplot(t, f_mc,
            xlabel="t", ylabel="F", name="monte-carlo sampling",
            # ribbon=@. sqrt(f_mc_err/M)
            )
            lineplot!(fig, t, f_ni, name="numerical integration")
            lineplot!(fig, t, f_th, name="theoretical fidelity")
        display(fig)
    end

    model=OneSpinForthBackModel(T,L,N,B)
    # test customize println
    println(model)

    f1=averagefidelity(model)
    f2, f2_err=sampling(model, fidelity, M)
    f3=1/2*(1+W(T, L, B, path=:forthback))
    @test isapprox(f1, f3,rtol=1e-2)
    @test isapprox(f2, f3, rtol=1e-2) 
    println("NI:", f1)
    println("MC:", f2)
    println("TH:", f3)
end

##
@testset begin "test two spin sequenced shuttling fidelity"
    L=10; σ =sqrt(2)/20; M=20000; N=501; T1=200; T0=25*0.05; κₜ=1/20; κₓ=1/0.1;
    B=OrnsteinUhlenbeckField(0,[κₜ,κₓ],σ)
    model=TwoSpinModel(T0, T1, L, N, B)
    if visualize
        display(heatmap(collect(model.R.Σ), title="cross covariance matrix, test fig 4"))
    end
    f1=averagefidelity(model)
    f2, f2_err=sampling(model, fidelity, M)
    f3=1/2*(1+W(T0, T1, L,B))
    @test isapprox(f1, f3,rtol=1e-2)
    @test isapprox(f2, f3, rtol=1e-2) 
    println("NI:", f1)
    println("MC:", f2)
    println("TH:", f3)
end

#
@testset "1/f noise chacacteristics" begin
    σ = sqrt(2)/20; M = 400; N=501; L=10; γ=(1e-2,1e2); # MHz
    # 0.01 ~ 100 μs
    # v = 0.1 ~ 1000 m/s
    v=1; T=L/v; κₓ=10;
    # T=10 
    # 1/T=0.1 N/T=20 
    B=PinkBrownianField(0,[κₓ],σ, γ)
    model=OneSpinModel(T,L,N,B)

    f1=averagefidelity(model)
    f2, f2_err=sampling(model, fidelity, M)
    f3=1/2*(1+W(T,L,B))
    @test isapprox(f1, f3,rtol=1e-2)
    @test isapprox(f2, f3, rtol=1e-2)
    println("NI:", f1)
    println("MC:", f2)
    println("TH:", f3)

    k=10
    t=range(0,T,k)
    f_mc=zeros(k);  f_mc_err=zeros(k); f_ni=zeros(k);
    @showprogress for i in eachindex(t)
        model=OneSpinModel(t[i],L,N,B)
        f_mc[i], f_mc_err[i]=sampling(model, fidelity, M)
        f_ni[i]=averagefidelity(model)
    end
    f_th=map(T->(1+W(T,L,B))/2, t)|>collect
    if visualize
        fig=lineplot(t,f_mc, 
            xlabel="t", ylabel="F", name="monte-carlo sampling",
            # ribbon=sqrt.(f_mc_err/M)
            )
            lineplot!(fig, t, f_ni, name="numerical integration")
            lineplot!(fig, t, f_th, name="theoretical fidelity")
        display(fig)
    end

    @test all([abs(f_mc[i]-f_th[i]) < sqrt(f_mc_err[i]) for i in eachindex(t)])
end