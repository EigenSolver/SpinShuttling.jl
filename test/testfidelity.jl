# ##
visualize=true

#
@testset begin "test single spin shuttling fidelity"
    T=100; L=10; σ = sqrt(2) / 20; M = 5000; N=601; κₜ=1/20;κₓ=1/0.1;
    γ = (1e-9, 1e3) # MHz
    B1 = OrnsteinUhlenbeckField(0, [κₜ, κₓ], σ)
    B2 = PinkLorentzianField(0, κₓ, σ, γ)
    for B in (B1,B2)
        model=OneSpinModel(T,L,N,B)
        # test customize println
        println(model)
        
        f1=statefidelity(model)
        f2, f2_err=sampling(model, statefidelity, M)
        f3=1/2*(1+W(T,L,B))
        @test isapprox(f1, f3,rtol=3e-2)
        @test isapprox(f2, f3, rtol=3e-2) 
        println("NI:", f1)
        println("MC:", f2)
        println("TH:", f3)
    end
end

#
@testset begin "test single spin forth-back shuttling fidelity"
    T=20; σ = sqrt(2) / 20; M = 5000; N=801; κₜ=1/20;κₓ=10;
    L=0.1; #α=1 
    γ = (1e-9, 1e3) # MHz 
    # exponential should be smaller than 100
    B1 = OrnsteinUhlenbeckField(0, [κₜ, κₓ], σ)
    B2 = PinkLorentzianField(0, κₓ, σ, γ)

    for B in (B1,B2)
        if visualize
            t=range(1e-2*T,T, 10)
            f_mc=[sampling(OneSpinForthBackModel(T,L,N,B), statefidelity, M)[1] for T in t]
            f_ni=[statefidelity(OneSpinForthBackModel(T,L,N,B)) for T in t]
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

        f1=statefidelity(model)
        f2, f2_err=sampling(model, statefidelity, M)
        f3=1/2*(1+W(T, L, B, path=:forthback))
        @test isapprox(f1, f3,rtol=3e-2)
        @test isapprox(f2, f3, rtol=3e-2) 
        println("NI:", f1)
        println("MC:", f2)
        println("TH:", f3)
    end
end


#
@testset "1/f noise chacacteristics" begin
    σ = sqrt(2)/20; M = 5000; N=501; L=10; γ=(1e-2,1e2); # MHz
    # 0.01 ~ 100 μs
    # v = 0.1 ~ 1000 m/s
    v=1; T=L/v; κ=10;
    # T=10 
    # 1/T=0.1 N/T=20 
    B=PinkLorentzianField(0,κ,σ, γ)
    model=OneSpinModel(T,L,N,B)

    f1=statefidelity(model)
    f2, f2_err=sampling(model, statefidelity, M)
    f3=1/2*(1+W(T,L,B))
    @test isapprox(f1, f3,rtol=3e-2)
    @test isapprox(f2, f3, rtol=3e-2)
    println("NI:", f1)
    println("MC:", f2)
    println("TH:", f3)

    k=10
    t=range(0.1,T,k)
    f_mc=zeros(k);  f_mc_err=zeros(k); f_ni=zeros(k);
    @showprogress for i in eachindex(t)
        model=OneSpinModel(t[i],L,N,B)
        f_mc[i], f_mc_err[i]=sampling(model, statefidelity, M)
        f_ni[i]=statefidelity(model)
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
end

##
@testset begin "test two spin sequenced shuttling fidelity"
    L=10; σ =sqrt(2)/20; M=5000; N=501; T1=200; T0=25*0.05; κₜ=1/20; κₓ=1/0.1;
    B=OrnsteinUhlenbeckField(0,[κₜ,κₓ],σ)
    model=TwoSpinSequentialModel(T0, T1, L, N, B)
    if visualize
        display(heatmap(collect(model.R.Σ), title="cross covariance matrix, two spin EPR"))
    end
    f1=statefidelity(model, method=:adaptive)
    f2, f2_err=sampling(model, statefidelity, M)
    f3=1/2*(1+W(T0, T1, L,B))
    @test isapprox(f1, f3,rtol=3e-2)
    @test isapprox(f2, f3, rtol=3e-2) 
    println("NI:", f1)
    println("MC:", f2)
    println("TH:", f3)
end


@testset begin "test two spin parallel shuttling fidelity"
    L=10; σ =sqrt(2)/20; M=5000; N=501; T=200; κₜ=1/20; κₓ=1/0.1;
    D=0.3;
    B=OrnsteinUhlenbeckField(0,[κₜ,κₓ,κₓ],σ)
    model=TwoSpinParallelModel(T, D, L, N, B)
    if visualize
        display(heatmap(collect(model.R.Σ), title="cross covariance matrix, two spin EPR"))
    end
    f1=statefidelity(model)
    f2, f2_err=sampling(model, statefidelity, M)
    w=exp(-σ^2 / (8 *κₜ*κₓ*κₓ) / κₜ^2 *(1-exp(-κₓ*D)) * SpinShuttling.P1(κₜ*T, κₓ*L))
    f3=1/2*(1+w)
    @test isapprox(f1, f3,rtol=3e-2)
    @test isapprox(f2, f3, rtol=3e-2) 
    println("NI:", f1)
    println("MC:", f2)
    println("TH:", f3)
end
