using SpinShuttling: covariancematrix, CompositeRandomFunction, 
Symmetric, covariancepartition, ishermitian, issymmetric
using Plots
##
figsize=(400, 300)
visualize=false

@testset begin "test of random function"
    T=400; L=10; σ = sqrt(2) / 20; M = 10000; N=11; κₜ=1/20;κₓ=1/0.1;
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

    P=vcat(P1,P2)
    R=RandomFunction(P,B)
    c=[1,1]
    RΣ=sum(c'*c .* covariancepartition(R, 2))
    @test size(RΣ) == (N,N)

    @test issymmetric(RΣ)
    @test ishermitian(RΣ)

    RC=CompositeRandomFunction(R, c)
    if visualize
        display(heatmap(sqrt.(RΣ)))
    end
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
        f3=1/2(1+Χ(T,L,B))
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

