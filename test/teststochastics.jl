using SpinShuttling: covariancepartition, Symmetric, Cholesky, ishermitian, issymmetric, integrate
using SpinShuttling: X_seq_shuttle_delay
using LsqFit
using Statistics: std, mean

visualize=true

#
@testset begin "test of random function"
    T=400; L=10; σ = sqrt(2) / 20; M = 10000; N=21; κₜ=1/20;κₓ=1/0.1;
    v=L/T;
    t=range(0, T, N)
    P=collect(zip(t, v.*t))
    B=OrnsteinUhlenbeckField(0,[κₜ,κₓ],σ)
    R=GaussianRandomFunction(P , B, initialize=false)
    initialize!(R)
    @test R() isa Vector{<:Real}
    @test R.Σ isa Symmetric

    t₀=T/5


    P1=P; P2=collect(zip(t, v.*(t.-t₀)))
    crosscov=covariancematrix(P1, P2, B)

    if visualize
        display(heatmap(collect(R.Σ), title="covariance matrix, test fig 1")) 
        display(lineplot(R(),title="random function, test fig 2"))
        display(heatmap(crosscov, title="cross covariance matrix, test fig 3"))
    end

    @test transpose(crosscov) == covariancematrix(P2, P1, B)

    P_comp=collect(zip(t, v.* t, v.*(t.-t₀)))
    R=GaussianRandomFunction(P_comp,B)

    c=[1,1]
    RΣ=sum(c'*c .* covariancepartition(R, 2))

    @test issymmetric(RΣ)
    @test ishermitian(RΣ)

    c=[1, -1]
    RC=CompositeGaussianRandomFunction(R, c)
    RCΣ=sum(c'*c .* covariancepartition(R, 2))

    @test issymmetric(RCΣ)
    @test ishermitian(RCΣ)

    @test sum(RC.Σ .-reduce(vcat, [reduce(hcat, row) for row in eachrow(RCΣ)]))<1e-6


    if visualize
        display(heatmap(sqrt.(RΣ)))
        display(heatmap(sqrt.(RCΣ)))
    end
end

@testset "semi-definite positive covariance matrix" begin
    σ = 1; N = 201; 
    M=10000;
    κₓ = 6.0;  # approximately 150nm
    T = 1;
    δt = T / N;
    γ = (1e-9, 1e3) # MHz
    # 0.0001 ~ 10000 μs
    # v = 1e-4 ~ 1e5 m/s
    B = PinkLorentzianField(0, κₓ, σ, γ)

    v=1; l=5
    ψ0=1/√2*[0im,0,1,0,-1,0,0,0];
    ψ1=1/√6*[0im,2,-1,0,-1,0,0,0];
    lc=1/κₓ;

    realistic_seq_shuttling(v::Real, τ::Real, l::Real, d::Real) = ShuttlingModel(3, ψ1, 2τ+l/v, N, B, X_seq_shuttle_delay(3, v, τ, l, d))
    model=realistic_seq_shuttling(v, lc/v, l, 0.000)
    
    @test issymmetric(model.R.Σ)
    @test ishermitian(model.R.Σ)
    @test isposdef(model.R.Σ) == false
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
        P=collect(zip(t, v.*t))
        R=GaussianRandomFunction(P , B)
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
        fig=lineplot(err1, xlabel="T", ylabel="error", name="trapezoid")
        lineplot!(fig, err2, name="simpson")
        display(fig)
    end
end

## 
@testset "symmetric integration for covariance matrix" begin
    σ = sqrt(2) / 20; κₜ=1;κₓ=10;
    γ=(1e-5,1e5) # MHz
    # 0.01 ~ 100 μs
    # v = 0.1 ~ 1000 m/s
    v=2;
    B=PinkLorentzianField(0,κₓ, σ, γ)
    # B=OrnsteinUhlenbeckField(0,[κₜ,κₓ],σ)

    M=5
    err1=zeros(M)
    err1_sym=zeros(M)
    err2=zeros(M)
    i=1

    N=201;
    T_range=range(10, 20, length=M)
    for T in T_range
        t=range(0, T, N)
        P=collect(zip(t, v.*t))
        R=GaussianRandomFunction(P , B)
        @test size(R.Σ) == (N,N)
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
