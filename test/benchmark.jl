using SpinShuttling
using Statistics
import SpinShuttling.sampling

T=4; L=10; σ = sqrt(2) / 20; M = Int(1e6); N=999; κₜ=1;κₓ=1;
B=OrnsteinUhlenbeckField(0,[κₜ,κₓ],σ)
model=OneSpinModel(T,L,N,B)

function f(model::ShuttlingModel, randseq::Vector{<:Real}; isarray::Bool=false)::Complex
    return exp(im*sum(model.R(randseq)))
end

println("Benchmark parallel sampling:")
@time sampling(model, f, M, isparallel=true)

println("Benchmark fidelity:")
@time sampling(model, statefidelity, M, isparallel=true)


println("Benchmark sequential sampling:")
@time sampling(model, f, M, isparallel=false)

function f(model::ShuttlingModel)::Complex
    return exp(im*sum(model.R()))
end

println(typeof(model.R.L))
A=model.R.L|>collect
sample=zeros(Complex{Float64},M)
randpool=randn(M,N)
println("Benchmark standard sampling:")
@time for i in 1:M
    sample[i]=cos(sum(A*randpool[i,:]))
end

mean(sample),var(sample)
