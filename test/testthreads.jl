using Test
using SpinShuttling
using DelimitedFiles
# using UnicodePlots
# using ProgressMeter

@testset "test multithreading" begin
    T=4; L=10; σ = sqrt(2) / 20; N=501; κₜ=1;κₓ=1;
    
    B=OrnsteinUhlenbeckField(0,[κₜ,κₓ],σ)
    model=OneSpinModel(T,L,N,B)

    # save the original Σ to text file
    if Threads.nthreads() > 1
        writedlm("test/cache.csv", model.R.Σ)
    else
        # read the original Σ from text file
        Σ = readdlm("test/cache.csv")
        # test the original Σ is the same as the one in the model
        @test isapprox(Σ, model.R.Σ, rtol=1e-10)
        # delete the cache file
        rm("test/cache.csv", force=true)
    end

    M = 100000; 
    F=sampling(model, statefidelity, M)
    println(F)
end
