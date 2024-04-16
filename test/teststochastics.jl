using SpinShuttling: covariancepartition, Symmetric, Cholesky, ishermitian, issymmetric, integrate
using LsqFit
using Statistics: std, mean

visualize=false

#
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
        display(heatmap(collect(R.Σ), title="covariance matrix, test fig 1")) 
        display(lineplot([R(),R(),R()],title="random function, test fig 2"))
        display(heatmap(crosscov, title="cross covariance matrix, test fig 3"))
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

#
@testset "trapezoid vs simpson for covariance matrix" begin
    L=10; σ = sqrt(2) / 20; N=501; κₜ=1/20;κₓ=1/0.1;
    v=20;
    B=OrnsteinUhlenbeckField(0,[κₜ,κₓ],σ)

    M=30
    err1=zeros(M)
    err2=zeros(M)
    i=1
    for T in range(1, 50, length=M)
        t=range(0, T, N)
        P=hcat(t, v.*t)
        R=RandomFunction(P , B)
        dt=T/N
        f1=exp(-integrate(R.Σ[:,:], dt, dt, method=:trapezoid)/2) 
        f2=exp(-integrate(R.Σ[:,:], dt, dt, method=:simpson)/2)
        @test isapprox(f1,f2,rtol=1e-2) 
        f3=W(T,L,B)
        err1[i]=abs(f1-f3)
        err2[i]=abs(f2-f3)
        i+=1
    end
    println("mean 1st order:", mean(err1))
    println("mean 2nd order:", mean(err2))
    if visualize
        fig=plot(err1, xlabel="T", ylabel="error", label="trapezoid")
        plot!(err2, label="simpson")
        display(fig)
    end
end

## 
@testset "symmetric integration for covariance matrix" begin
    σ = sqrt(2) / 20; N=201; κₜ=1;κₓ=10;
    γ=(1e-5,1e5) # MHz
    # 0.01 ~ 100 μs
    # v = 0.1 ~ 1000 m/s
    v=2;
    B=PinkBrownianField(0,[κₓ],σ, γ)
    # B=OrnsteinUhlenbeckField(0,[κₜ,κₓ],σ)

    M=5
    err1=zeros(M)
    err1_sym=zeros(M)
    err2=zeros(M)
    i=1
    T_range=range(10, 20, length=M)
    for T in T_range
        t=range(0, T, N)
        P=hcat(t, v.*t)
        R=RandomFunction(P , B)
        dt=T/N
        f1=exp(-integrate(R.Σ[:,:], dt, dt, method=:trapezoid)/2) 
        f1_sym=exp(-integrate(R.Σ, dt)/2)
        f2=exp(-integrate(R.Σ[:,:], dt, dt, method=:simpson)/2)
        f3=W(T,v*T,B)
        err1[i]=abs(f1-f3)/f3
        err1_sym[i]=abs(f1_sym-f3)/f3
        err2[i]=abs(f2-f3)/f3
        i+=1
    end
    println("mean 1st order:", mean(err1))
    println("mean 1st order symmetric:", mean(err1_sym))
    println("mean 2nd order:", mean(err2))
    @test mean(err1) < 1e-2
    @test mean(err1_sym) < 1e-2
    @test mean(err2) < 1e-2
end
