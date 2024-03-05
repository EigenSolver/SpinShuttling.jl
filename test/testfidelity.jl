using Plots
using SpinShuttling: characteristicfunction
##
figsize=(400, 300)
visualize=false

##
# @testset begin "test single spin shuttling fidelity"
#     T=400; L=10; σ = sqrt(2) / 20; M = 20000; N=601; κₜ=1/20;κₓ=1/0.1;
#     v=L/T;
#     t=range(0, T, N)
#     P=hcat(t, v.*t)
#     B=OrnsteinUhlenbeckField(0,[κₜ,κₓ],σ)
# 
#     model=OneSpinModel(T,L,M,N,B)
#     # test customize println
#     println(model)

#     f1=averagefidelity(model)
#     f2, f2_err=sampling(model, fidelity)
#     f3=1/2*(1+φ(T,L,B))
#     @test isapprox(f1, f3,rtol=1e-2)
#     @test isapprox(f2, f3, rtol=1e-2) 
#     println("NI:", f1)
#     println("MC:", f2)
#     println("TH:", f3)
# end

#
@testset begin "test single spin forth-back shuttling fidelity"
    T=200; L=10; σ = sqrt(2) / 20; M = 5000; N=501; κₜ=1/20;κₓ=1/0.1;
    v=L/T;
    t=range(0, T, N)
    P=hcat(t, v.*t)
    B=OrnsteinUhlenbeckField(0,[κₜ,κₓ],σ)

    if visualize
        t=range(1e-2*T,T, 10)
        f_mc=[sampling(OneSpinForthBackModel(T,L,M,N,B), fidelity)[1] for T in t]
        f_ni=[averagefidelity(OneSpinForthBackModel(T,L,M,N,B)) for T in t]
        f_th=map(T->(1+φ(T,L,B,path=:forthback))/2, t)|>collect
        fig=plot(t, f_mc, size=figsize, 
            xlabel="t", ylabel="F", label="monte-carlo sampling",
            # ribbon=@. sqrt(f_mc_err/M)
            )
            plot!(t, f_ni, label="numerical integration")
            plot!(t, f_th, label="theoretical fidelity")
        display(fig)
    end

    model=OneSpinForthBackModel(T,L,M,N,B)
    # test customize println
    println(model)

    f1=averagefidelity(model)
    f2, f2_err=sampling(model, fidelity)
    f3=1/2*(1+φ(T, L, B, path=:forthback))
    @test isapprox(f1, f3,rtol=1e-2)
    @test isapprox(f2, f3, rtol=1e-2) 
    println("NI:", f1)
    println("MC:", f2)
    println("TH:", f3)
end

# ##
# @testset begin "test two spin sequenced shuttling fidelity"
#     L=10; σ =sqrt(2)/20; M=20000; N=501; T1=200; T0=25*0.05; κₜ=1/20; κₓ=1/0.1;
#     B=OrnsteinUhlenbeckField(0,[κₜ,κₓ],σ)
#     model=TwoSpinModel(T0, T1, L, M, N, B)
#     if visualize
#         display(heatmap(collect(model.R.Σ), title="cross covariance matrix, test fig 4", size=figsize))
#     end
#     f1=averagefidelity(model)
#     f2, f2_err=sampling(model, fidelity)
#     f3=1/2*(1+φ(T0, T1, L,B))
#     @test isapprox(f1, f3,rtol=1e-2)
#     @test isapprox(f2, f3, rtol=1e-2) 
#     println("NI:", f1)
#     println("MC:", f2)
#     println("TH:", f3)
# end

##
# @testset "1/f noise chacacteristics" begin
#     σ = sqrt(2)/20; M = 500; N=501; L=10; γ=(1e-7,1e7); # MHz
#     # 0.01 ~ 100 μs
#     # v = 0.1 ~ 1000 m/s
#     v=0.1; T=L/v; κₓ=20;
#     B=PinkBrownianField(0,[κₓ],σ, γ)
#     model=OneSpinModel(T,L,M,N,B)

#     f1=averagefidelity(model)
#     f2, f2_err=sampling(model, fidelity)
#     f3=1/2*(1+φ(T,L,B))
#     @test isapprox(f1, f3,rtol=1e-2)
#     @test isapprox(f2, f3, rtol=1e-2)
#     println("NI:", f1)
#     println("MC:", f2)
#     println("TH:", f3)

#     t=range(1e-4*T,T,10)
#     f_mc, f_mc_err=sampling(model, fidelity, vector=true)
#     t_ni, chi_ni=characteristicfunction(model.R)
#     f_ni= @. (1+real(chi_ni))/2
#     f_th=map(T->(1+φ(T,L,B))/2, t)|>collect
#     if visualize
#         fig=plot( size=figsize, 
#             xlabel="t", ylabel="F", label="monte-carlo sampling",
#             ribbon=sqrt.(f_mc_err/M)
#             )
#             plot!(t_ni, f_ni, label="numerical integration")
#             plot!(t,f_th, label="theoretical fidelity")
#         savefig(fig, "1fnoise.png")
#     end

#     @test all([abs(f_mc[i]-f_th[i]) < sqrt(f_mc_err[i]) for i in 10:N ])
# end

