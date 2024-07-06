
"""
Monte-Carlo sampling of any objective function. 
The function must return Tuple{Number,Number} or Tuple{Vector{<:Number},Vector{<:Number}}

# Arguments
- `samplingfunction::Function`: The function to be sampled
- `M::Int`: Monte-Carlo sampling size
# Returns
- `Tuple{Number,Number}`: The mean and variance of the sampled function
- `Tuple{Vector{<:Number},Vector{<:Number}}`: The mean and variance of the sampled function
# Example
```julia
f(x) = x^2
sampling(f, 1000)
```

# Reference
https://en.wikipedia.org/wiki/Standard_deviation#Rapid_calculation_methods
"""
function sampling(samplingfunction::Function, M::Int)::Union{Tuple{Number,Number},Tuple{Vector{<:Number},Vector{<:Number}}}
    if nthreads() > 1
        return parallelsampling(samplingfunction, M)
    end
    N = length(samplingfunction(1))
    A = N > 1 ? zeros(N) : 0
    Q = copy(A)
    for k in 1:M
        x = samplingfunction(k)::Union{Number,Vector{<:Number}}
        Q = Q + (k - 1) / k * abs.(x - A) .^ 2
        A = A + (x - A) / k
    end
    return A, Q / (M - 1)
end

function parallelsampling(samplingfunction::Function, M::Int)::Union{Tuple{Number,Number},Tuple{Vector{<:Number},Vector{<:Number}}}
    N = length(samplingfunction(1))
    if N > 1
        cache = zeros(N, M)
        @threads for i in 1:M
            cache[:, i] .= samplingfunction(i)
        end
        A = mean(cache, dims=2)
        Q = var(cache, dims=2)
        return A, Q
    else
        cache = zeros(M)
        @threads for i in 1:M
            cache[i] = samplingfunction(i)
        end
        A = mean(cache)
        Q = var(cache)
        return A, Q
    end
end
