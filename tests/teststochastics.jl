using Plots
##
figsize=(400, 300)
visualize=false

@testset begin "test of random function"
    T=400; L=10; σ = sqrt(2) / 20; M = 10000; N=301; κₜ=1/20;κₓ=1/0.1;
    v=L/T;
    t=range(0, T, N)
    P=hcat(t, v.*t)
    B=OrnsteinUhlenbeckField(0,[κₜ,κₓ],σ)
    R=RandomFunction(P , B)
    @test R() isa Vector{<:Real}
    @test R.Σ isa Symmetric
    t₀=T/5
    P1=hcat(t, v.*t); P2=hcat(t, v.*(t.-t₀))
    crosscov=covariancematrix(P1, P2, B)

    if visualize
        display(heatmap(collect(R.Σ), title="covariance matrix, test fig 1", size=figsize)) 
        display(plot([R(),R(),R()],title="random function, test fig 2", size=figsize))
        display(heatmap(crosscov, title="cross covariance matrix, test fig 3", size=figsize))
    end

    @test transpose(crosscov) == covariancematrix(P2, P1, B)
end

## 
@testset "trapezoid vs simpson for covariance matrix" begin
    T=400; L=10; σ = sqrt(2) / 20; N=301; κₜ=1/20;κₓ=1/0.1;
    v=L/T;
    t=range(0, T, N)
    P=hcat(t, v.*t)
    M=20
    err1=zeros(M)
    err2=zeros(M)
    i=1
    for T in T/2 .*(1 .+rand(M))
        B=OrnsteinUhlenbeckField(0,[κₜ,κₓ],σ)
        R=RandomFunction(P , B)
        dt=T/N
        f1=exp(-integrate(R.Σ[:,:], dt, dt, method=:trapezoid)/2) 
        f2=exp(-integrate(R.Σ[:,:], dt, dt, method=:simpson)/2)
        f3=Χ(T,L,B)
        err1[i]=abs(f1-f3)
        err2[i]=abs(f2-f3)
        i+=1
    end
    println("std 1st order:", std(err1))
    println("std 2st order:", std(err2))
    fig=plot(err1, xlabel="sample", ylabel="error", label="trapezoid")
    plot!(err2, label="simpson")
    display(fig)
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
    f1=fidelity(model)
    f2, f2_err=sampling(model, fidelity)
    f3=Χ(T,L,B)
    @test isapprox(f1, f3,rtol=1e-2)
    @test isapprox(f2, f3, rtol=1e-2) 
    println("NI:", f1)
    println("MC:", f2)
    println("TH:", f3)
end

##
@testset begin "test two spin characteristics"
    L=10; σ =sqrt(2)/20; M=1; N=201; T1=1000; T0=25*0.05; κₜ=1/20; κₓ=1/0.1;
    B=OrnsteinUhlenbeckField(0,[κₜ,κₓ],σ)
    model=TwoSpinModel(T0, T1, L, M, N, B)
    if visualize
        display(heatmap(collect(model.R.Σ), title="cross covariance matrix, test fig 4", size=figsize))
    end
    f1=fidelity(model)
    f2, f2_err=sampling(model, fidelity)
    f3=Χ(T,L,B)
    @test isapprox(f1, f3,rtol=1e-2)
    @test isapprox(f2, f3, rtol=1e-2) 
    println("NI:", f1)
    println("MC:", f2)
    println("TH:", f3)
end


# ## 
# @testset "numerical stability at margin" begin
#     let L=10; σ =sqrt(2)/20; M=300; N=201; T1=1000; T0=25*0.05; κₜ=1/20; κₓ=1/0.1;
#         B=OrnsteinUhlenbeckField(0,[κₜ,κₓ],σ)
#         model=TwoSpinModel(T0, T1, L, M, N, B)
#         f1=fidelity(model)
#         f2, f2_err=sampling(model, fidelity)
#         f3=Χ(T0 ,T1, L, B)
#         println(f1,"  ", f3)
#         println(f2,"  ", f3)
#         # @test isapprox(f1, f3,rtol=1e-5)
#         # @test isapprox(f2, f3, rtol=1e-5) 
#     end
# end 