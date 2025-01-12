```

```
function compositedephasing(model::ShuttlingModel, c::Vector{Int})::Real
    # @assert length(c) == model.n
    R = CompositeRandomFunction(model.R, c)
    return characteristicvalue(R, method=:trapezoid)
end


```
Periodic Dynamical Decoupling (PDD) mask for shuttling model.
The mask in one period is defined as $(1,-1)$.
# Arguments
- `model::ShuttlingModel`: The shuttling model`
- `n::Int`: The number of periods in the sequence
```
function W_pdd(model::ShuttlingModel, n::Int)
    @assert model.N%(2*n)==0
    compositedephasing(model, repeat([1,-1],n))
end

```
Carr-Purcell (CP) mask for shuttling model.
The mask in one period is defined as $(1,-1,-1,1)$.
# Arguments
- `model::ShuttlingModel`: The shuttling model`
- `n::Int`: The number of periods in the sequence
```
function W_cp(model::ShuttlingModel, n::Int)
    @assert model.N%(4*n)==0
    compositedephasing(model, repeat([1,-1,-1,1],n))
end