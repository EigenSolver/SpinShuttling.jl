using Plots
using FFTW
using Statistics: std, mean
using LsqFit: curve_fit

figsize=(400, 300)
visualize=false

##
@testset "test 1/f noise" begin
    σ = 1; M = 1000; N=1001; κₜ=1;κₓ=0;
    L=10;
    γ=(1e-5,1e3) # MHz
    # 0.01 ~ 100 μs
    # v = 0.1 ~ 1000 m/s
    v=1; T=L/v;
    B=PinkBrownianField(0,[κₓ],σ, γ)
    model=OneSpinModel(T,L,N,B)
    println(model)

    random_trace=model.R()
    if visualize
        display(heatmap(sqrt.(model.R.Σ), title="covariance matrix", size=figsize))
    end
    display(model.R.Σ[1:5,1:5])
    @test random_trace isa Vector{<:Real}
    
    if visualize
        fig=scatter(random_trace, title="1/f noise", xlabel="t", ylabel="B(t)", size=figsize)
        display(fig)
    end


    analytical_psd(ω::Real, γ::Tuple, σ::Real)=sqrt(2/π)* σ^2*(atan(γ[2]/ω)-atan(γ[1]/ω))/ω/log(γ[2]/γ[1])
    
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
    @test isapprox(fit.param[2], -1, atol=2e-1)
    if visualize
        println("cutoff freq: ",(freq[1], freq[end]))

        fig=plot(freq,psd,
        label="PSD", 
        scale=:log10, 
        xlabel="Frequency (MHz)", ylabel="Power",
        lw=2, marker=:vline
        )
        # plot the fit
        plot!(freq,exp.(fit.param[1] .+ fit.param[2]*log.(freq)), 
        label="Fitting", lw=2)
        plot!(freq, collect(map(f->analytical_psd.(f, γ, σ), freq)), label="Analytical", lw=2)
        display(fig)
    end

end