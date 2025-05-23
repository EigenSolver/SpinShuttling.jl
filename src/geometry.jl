
"""
One spin model with a right angle turn trajectory

# Arguments
- `T1::Real`: time to reach the corner
- `T2::Real`: time to reach the final position
- `v::Real`: velocity
- `N::Int`: number of time steps
- `B::GaussianRandomField`: noise process

# Returns
- `OneSpinModel`: a one spin model

"""
function OneSpinTurnModel(T1::Real, T2::Real, v::Real, N::Int, B::GaussianRandomField)
    L1=v*T1; L2=v*T2;
    function x(t::Real)::Tuple{Real,Real}
        if t<0
            return (0.0,0.0)
        elseif 0 ≤ t < T1
            return (v * t, 0.0)
        elseif T1 ≤ t ≤ T1+T2
            return (L1, v*(t-T1))
        else
            return (L1,L2)
        end
    end

    Ψ = 1 / √2 .* [1+0im,1+0im]
    return OneSpinModel(Ψ, T1+T2, N, B, x)
end

"""
One spin shuttling with a square trajectory

# Arguments
- `t::Real`: total time
- `T::Real`: time to complete a square
- `L::Real`: side length
- `N::Int`: number of time steps
- `B::GaussianRandomField`: noise process

# Returns
- `OneSpinModel`: a one spin model
"""
function OneSpinRectangleModel(t::Real, T::Real, a::Real, b::Real, N::Int, B::GaussianRandomField)

    v = 2(a+b)/T

    function x(t::Real)::Tuple{Real,Real}
        t=mod(t, T)
        T1=a/v
        if 0 ≤ t < T1
            return (v * t, 0.0)
        elseif T1 ≤ t < T/2
            return (a, v*(t-T1))
        elseif T/2 ≤ t < T/2+T1
            return (a-v*(t-T/2), b)
        elseif T/2+T1 ≤ t ≤ T
            return (0.0, b-v*(t-T/2-T1))
        else
            return (0.0,0.0)
        end
    end

    Ψ = 1 / √2 .* [1+0im,1+0im]

    return OneSpinModel(Ψ, t, N, B, x)
end

function OneSpinRectangleModel(T::Real, a::Real, b::Real, N::Int, B::GaussianRandomField)
    return OneSpinRectangleModel(T, T, a, b, N, B)
end

function OneSpinSquareModel(t::Real, T::Real, L::Real, N::Int, B::GaussianRandomField)
    return OneSpinRectangleModel(t, T, L, L, N, B)
end

function OneSpinSquareModel(T::Real, L::Real, N::Int, B::GaussianRandomField)
    return OneSpinRectangleModel(T, T, L, N, B)
end


function OneSpinTriangleModel(t::Real, T::Real, a::Real, N::Int, B::GaussianRandomField)
    v = 3a/T
    function x(t::Real)::Tuple{Real,Real}
        t=mod(t, T)
        if 0 ≤ t < T/3
            return (v * t, 0.0)
        elseif T/3 ≤ t < 2T/3
            return (v*(T-t)/2, v*(t-T/3)*√3/2)
        elseif 2T/3 ≤ t ≤ T
            return (v*(T-t)/2, (v*(T-t))*√3/2)
        else
            return (0.0,0.0)
        end
    end

    Ψ = 1 / √2 .* [1+0im,1+0im]

    return OneSpinModel(Ψ, t, N, B, x)
end

"""

# Arguments
- `n::Int`: number of spins
- `v::Real`: velocity
- `d::Real`: distance between spins in the same channel

# Returns
- `Vector{Function}`: a vector of functions
"""
X_seq_shuttle(n::Int, v::Real, d::Real) = [t->(v*t + (k-1)*d,0.0) for k in 1:n]


"""

# Arguments
- `n::Int`: number of spins
- `v::Real`: velocity
- `d::Real`: distance between two parallel chennels 

# Returns
- `Vector{Function}`: a vector of functions
"""
X_prl_shuttle(n::Int, v::Real, d::Real) = [t->(v*t, (k-1)*d) for k in 1:n]


"""

# Arguments
- `t::Real`: time
- `v::Real`: velocity
- `t1::Real`: time to start moving
- `t2::Real`: time to stop moving
"""
function X_padding(t::Real, v::Real, t1::Real, t2::Real)::Real
    if t<t1
        return 0
    elseif t<t2
        return v*(t-t1)
    else
        return v*(t2-t1)
    end
end


"""

# Arguments
- `n::Int`: number of spins
- `v::Real`: velocity
- `τ::Real`: time delay between two spins
- `l::Real`: length of the channel
- `d::Real`: initial distance between spins in the same channel

# Returns
- `Vector{Function}`: a vector of functions
"""
function X_seq_shuttle(n::Int, v::Real, τ::Real, l::Real, d::Real=0.0) 
    return [t->X_padding(t-(k-1)*τ,v,0,l/v)+(n-k)*d for k in 1:n]
end


"""

# Arguments
- `v::Real`: velocity
- `d1::Real`: distance between the spins in the first channel
- `d2::Real`: distance between the parallel channel

# Returns
- `Vector{Function}`: a vector of functions, only for 3 spins
"""
function X_tri_shuttle(v::Real,d1::Real,d2::Real)
    return [t->(v*t,0.0),t->(d1+v*t,0.0),t->(v*t,d2)]
end


"""

# Arguments
- `v::Real`: velocity
- `τ::Real`: time delay between two spins
- `l::Real`: length of the channel
- `d::Real`: distance between parallel channels

# Returns
- `Vector{Function}`: a vector of functions, only for 3 spins
"""
function X_tri_shuttle_delay(v::Real, τ::Real, l::Real, d::Real)
    x1=t->(X_padding(t, v, 0, l/v),0.0) 
    x2=t->(X_padding(t, v, τ, l/v+τ),0.0)
    x3=t->(X_padding(t, v, 0, l/v), d)
    return [x1,x2,x3]
end

