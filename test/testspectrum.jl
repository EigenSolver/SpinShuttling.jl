using FFTW
using Statistics: std, mean
using LsqFit: curve_fit

visualize=true

##
@testset "test 1/f noise" begin
    σ = 1; M = 1000; N=1001;
    L=10;
    γ=(1e-5,1e3) # MHz
    # 0.01 ~ 100 μs
    # v = 0.1 ~ 1000 m/s
    v=1; T=L/v;
    B=PinkLorentzianField(0.0,0.0,σ, γ)
    model=OneSpinModel(T,L,N,B)
    println(model)

    random_trace=model.R()
    if visualize
        display(heatmap(sqrt.(model.R.Σ), title="covariance matrix"))
    end
    display(model.R.Σ[1:5,1:5])
    @test random_trace isa Vector{<:Real}
    
    if visualize
        fig=scatterplot(random_trace, title="1/f noise", xlabel="t", ylabel="B(t)")
        display(fig)
    end

    freq=rfftfreq(N,N/T)
    psd_sheet=zeros(N÷2+1, M)

    for i in 1:M
        trace=model.R()
        psd_sheet[:,i]= abs.(rfft(trace)).^2
    end

    psd=mean(psd_sheet,dims=2);
    psd_std=std(psd_sheet,dims=2);
    freq=freq[2:end];
    psd=psd[2:end];
    # make a linear fit to the log-log plot
    
    cutoff1=1
    cutoff2=0
    
    freq=freq[cutoff1:end-cutoff2]
    psd=psd[cutoff1:end-cutoff2]

    fit = curve_fit((x,p) -> p[1] .+ p[2]*x, log.(freq), log.(psd), [0.0, 0.0])
    println("fit: ",fit.param)
    @test isapprox(fit.param[2], -1, atol=3e-1)
    if visualize
        println("cutoff freq: ",(freq[1], freq[end]))

        fig=lineplot(freq,psd,
        name="PSD", 
        xscale=:log10, yscale=:log10,
        xlabel="Frequency (MHz)", ylabel="Power",
        )
        # plot the fit
        lineplot!(fig, freq,exp.(fit.param[1] .+ fit.param[2]*log.(freq)), 
        name="Fitting")
        # lineplot!(fig, freq, collect(map(f->analytical_psd(f, γ, σ), freq)), name="Analytical")
        display(fig)
    end

end

@testset "test white noise" begin
    σ = 1; M = 1000; N=1001;
    L=10;
    γ=(1e-5,1e3) # MHz
    # 0.01 ~ 100 μs
    # v = 0.1 ~ 1000 m/s
    v=1; T=L/v;
    B=PinkPiField(0.0, 1 ,σ, γ)
    model=OneSpinModel(T,L,N,B)
    println(model)

    random_trace=model.R()
    if visualize
        display(heatmap(sqrt.(model.R.Σ), title="covariance matrix"))
    end
    display(model.R.Σ[1:5,1:5])
    @test random_trace isa Vector{<:Real}
    
    if visualize
        fig=scatterplot(random_trace, title="1/f noise", xlabel="t", ylabel="B(t)")
        display(fig)
    end

    freq=rfftfreq(N,N/T)
    psd_sheet=zeros(N÷2+1, M)

    for i in 1:M
        trace=model.R()
        psd_sheet[:,i]= abs.(rfft(trace)).^2
    end

    psd=mean(psd_sheet,dims=2);
    psd_std=std(psd_sheet,dims=2);
    freq=freq[2:end];
    psd=psd[2:end];
    # make a linear fit to the log-log plot
    
    cutoff1=1
    cutoff2=0
    
    freq=freq[cutoff1:end-cutoff2]
    psd=psd[cutoff1:end-cutoff2]

    fit = curve_fit((x,p) -> p[1] .+ p[2]*x, log.(freq), log.(psd), [0.0, 0.0])
    println("fit: ",fit.param)
    @test isapprox(fit.param[2], -1, atol=3e-1)
    if visualize
        println("cutoff freq: ",(freq[1], freq[end]))

        fig=lineplot(freq,psd,
        name="PSD", 
        xscale=:log10, yscale=:log10,
        xlabel="Frequency (MHz)", ylabel="Power",
        )
        # plot the fit
        lineplot!(fig, freq,exp.(fit.param[1] .+ fit.param[2]*log.(freq)), 
        name="Fitting")
        # lineplot!(fig, freq, collect(map(f->analytical_psd(f, γ, σ), freq)), name="Analytical")
        display(fig)
    end

end