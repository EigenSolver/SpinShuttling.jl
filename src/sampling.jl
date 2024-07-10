"""
Monte-Carlo sampling of any objective function without storage in memory. 
The function must return Tuple{Number,Number} or Tuple{VecOrMat{<:Number},VecOrMat{<:Number}}

# Arguments
- `samplingfunction::Function`: The function to be sampled
- `M::Int`: Monte-Carlo sampling size
# Returns
- `Tuple{Number,Number}`: The mean and variance of the sampled function
- `Tuple{VecOrMat{<:Number},VecOrMat{<:Number}}`: The mean and variance of the sampled function
# Example
```julia
f(x) = x^2
sampling(f, 1000)
```

# Reference
https://en.wikipedia.org/wiki/Standard_deviation#Rapid_calculation_methods
"""
function serialsampling(samplingfunction::Function, M::Int)::Union{Tuple{Number,Number},Tuple{VecOrMat{<:Number},VecOrMat{<:Number}}}
    A = samplingfunction()
    A isa Array ? A .= 0 : A = 0
    Q = copy(A)
    for k in 1:M
        x = samplingfunction()::Union{Number,VecOrMat{<:Number}}
        Q = Q + (k - 1) / k * abs.(x - A) .^ 2
        A = A + (x - A) / k
    end
    return A, Q / (M - 1)
end

"""
Multi-threaded Monte-Carlo sampling of any objective function.
The function must return Tuple{Number,Number} or Tuple{VecOrMat{<:Number},VecOrMat{<:Number}}
"""
function parallelsampling(samplingfunction::Function, M::Int)::Union{Tuple{Number,Number},Tuple{VecOrMat{<:Number},VecOrMat{<:Number}}}
    obj = samplingfunction()
    if obj isa Number
        cache = zeros(typeof(obj), M)
        @threads for i in 1:M
            cache[i] = samplingfunction()
        end
        A = mean(cache)
        Q = var(cache)
        return A, Q
    elseif obj isa Vector
        cache = zeros(typeof(obj).parameters[1], length(obj), M)
        @threads for i in 1:M
            cache[:, i] .= samplingfunction()
        end
        A = dropdims(mean(cache, dims=2),dims=2)
        Q = dropdims(var(cache, dims=2),dims=2)
        return A, Q
    elseif obj isa Matrix
        cache = zeros(typeof(obj).parameters[1], size(obj)..., M)
        @threads for i in 1:M
            cache[:, :, i] .= samplingfunction()
        end
        A = dropdims(mean(cache, dims=3),dims=3)
        Q = dropdims(var(cache, dims=3),dims=3)
        return A, Q 
    else
        error("The objective function must return `VecOrMat{<:Number}` or `Number`")
    end
end
