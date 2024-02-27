using SpinShuttling: covariancematrix, CompositeRandomFunction, 
Symmetric, Cholesky, covariancepartition, ishermitian, issymmetric, integrate, std,
characteristicfunction
using Plots
using FFTW
using LsqFit
using Statistics: std, mean

##
figsize=(400, 300)
visualize=true

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
    T=400; L=10; σ = sqrt(2) / 20; N=1001; κₜ=1/20;κₓ=1/0.1;
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
    if visualize
        fig=plot(err1, xlabel="sample", ylabel="error", label="trapezoid")
        plot!(err2, label="simpson")
        display(fig)
    end
end

##
@testset "test 1/f noise" begin
    σ = sqrt(2) / 20; M = 1000; N=601; κₜ=1/20;κₓ=0;
    L=10;
    γ=(1e-8,1e8) # MHz
    # 0.01 ~ 100 μs
    # v = 0.1 ~ 1000 m/s
    v=2; T=L/v;
    B=PinkBrownianField(0,[κₓ],σ, γ)
    model=OneSpinModel(T,L,M,N,B)
    @test model.R.Σ isa Symmetric
    @test model.R.C isa Cholesky
    println(model)

    random_trace=model.R()
    if visualize
        display(heatmap(sqrt.(model.R.Σ)))
    end
    display(model.R.Σ[1:5,1:5])
    @test random_trace isa Vector{<:Real}
    
    if visualize
        fig=scatter(random_trace, title="1/f noise", size=figsize)
        display(fig)
    end

    
    freq=fftfreq(N,1/(2*T))
    psd_sheet=zeros(N, M)

    for i in 1:M
        trace=model.R()
        psd_sheet[:,i]= abs.(fft(trace)).^2
    end

    psd=mean(psd_sheet,dims=2);
    psd_std=std(psd_sheet,dims=2);
    freq=freq[2:N÷2];
    psd=psd[2:N÷2];

    # make a linear fit to the log-log plot
    cutoff1=40
    cutoff2=50
    freq=freq[cutoff1:end-cutoff2]
    psd=psd[cutoff1:end-cutoff2]

    fit = curve_fit((x,p) -> p[1] .+ p[2]*x, log.(freq), log.(psd), [0.0, 0.0])
    println("fit: ",fit.param)
    # @test isapprox(fit.param[2], -1, atol=4e-2)
    if visualize
        println("cutoff freq: ",(freq[cutoff1], freq[cutoff2]))

        fig=plot(freq,psd,
        label="PSD", 
        scale=:log10, 
        xlabel="Frequency (MHz)", ylabel="Power",
        lw=2, marker=:vline
        )
        # plot the fit
        plot!(freq,exp.(fit.param[1] .+ fit.param[2]*log.(freq)), 
        label="Fitting", lw=2)
        display(fig)
    end
end