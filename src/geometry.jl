
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
function OneSpinTurnModel(T1::Real, T2::Real, v::Real, N::Int, B::GaussianRandomField;
    initialize::Bool=false)
    L1 = v * T1
    L2 = v * T2
    function x(t::Real)::Tuple{Real,Real}
        if t < 0
            return (0.0, 0.0)
        elseif 0 ≤ t < T1
            return (v * t, 0.0)
        elseif T1 ≤ t ≤ T1 + T2
            return (L1, v * (t - T1))
        else
            return (L1, L2)
        end
    end

    Ψ = 1 / √2 .* [1 + 0im, 1 + 0im]
    return OneSpinModel(Ψ, T1 + T2, N, B, x; initialize=initialize)
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
function OneSpinRectangleModel(t::Real, T::Real, a::Real, b::Real, N::Int, B::GaussianRandomField;
    initialize::Bool=false)

    v = 2(a + b) / T

    function x(t::Real)::Tuple{Real,Real}
        t = mod(t, T)
        T1 = a / v
        if 0 ≤ t < T1
            return (v * t, 0.0)
        elseif T1 ≤ t < T / 2
            return (a, v * (t - T1))
        elseif T / 2 ≤ t < T / 2 + T1
            return (a - v * (t - T / 2), b)
        elseif T / 2 + T1 ≤ t ≤ T
            return (0.0, b - v * (t - T / 2 - T1))
        else
            return (0.0, 0.0)
        end
    end

    Ψ = 1 / √2 .* [1 + 0im, 1 + 0im]

    return OneSpinModel(Ψ, t, N, B, x; initialize=initialize)
end

function OneSpinRectangleModel(T::Real, a::Real, b::Real, N::Int, B::GaussianRandomField;
    initialize::Bool=false)
    return OneSpinRectangleModel(T, T, a, b, N, B; initialize=initialize)
end

function OneSpinSquareModel(t::Real, T::Real, L::Real, N::Int, B::GaussianRandomField;
    initialize::Bool=false)
    return OneSpinRectangleModel(t, T, L, L, N, B; initialize=initialize)
end

function OneSpinSquareModel(T::Real, L::Real, N::Int, B::GaussianRandomField;
    initialize::Bool=false)
    return OneSpinRectangleModel(T, T, L, N, B; initialize=initialize)
end


function OneSpinTriangleModel(t::Real, T::Real, a::Real, N::Int, B::GaussianRandomField;
    initialize::Bool=false)
    v = 3a / T
    function x(t::Real)::Tuple{Real,Real}
        t = mod(t, T)
        if 0 ≤ t < T / 3
            return (v * t, 0.0)
        elseif T / 3 ≤ t < 2T / 3
            return (v * (T - t) / 2, v * (t - T / 3) * √3 / 2)
        elseif 2T / 3 ≤ t ≤ T
            return (v * (T - t) / 2, (v * (T - t)) * √3 / 2)
        else
            return (0.0, 0.0)
        end
    end

    Ψ = 1 / √2 .* [1 + 0im, 1 + 0im]

    return OneSpinModel(Ψ, t, N, B, x; initialize=initialize)
end

"""

# Arguments
- `t::Real`: total time
- `T::Real`: time to complete a circle
- `R::Real`: radius
- `N::Int`: number of time steps
- `B::GaussianRandomField`: noise process
"""
function OneSpinCircleModel(t::Real, T::Real, R::Real, N::Int, B::GaussianRandomField;
    initialize::Bool=false)
    ω = 2π / T
    function x(t::Real)::Tuple{Real,Real}
        θ = mod(ω * t, 2π)
        return (R * cos(θ), R * sin(θ))
    end

    Ψ = 1 / √2 .* [1 + 0im, 1 + 0im]

    return OneSpinModel(Ψ, t, N, B, x; initialize=initialize)
end

function OneSpinRaceTrackModel(t::Real, T::Real, r::Real, l::Real, N::Int, B::GaussianRandomField;
    initialize::Bool=false)
    v = (2l + 2π*r) / T
    function x(t::Real)::Tuple{Real,Real}
        t = mod(t, T)
        s = v * t
        l1 = l
        l2 = π * r
        if 0 <= s < l1
            return (s, 0.0)
        elseif l1 <= s < l1 + l2
            phi = (s - l1) / r
            return (l + r * sin(phi), r - r * cos(phi))
        elseif l1 + l2 <= s < 2l1 + l2
            u = s - (l1 + l2)
            return (l - u, 2r)
        elseif 2l1 + l2 <= s <= 2l1 + 2l2
            phi = (s - (2l1 + l2)) / r
            return (-r * sin(phi), r + r * cos(phi))
        else
            return (0.0, 0.0)
        end
    end
    Ψ = 1 / √2 .* [1+0im, 1+0im]
    return OneSpinModel(Ψ, t, N, B, x, initialize=initialize)
end

function OneSpinHexagonModel(t::Real, T::Real, R::Real, N::Int, B::GaussianRandomField;
    initialize::Bool=false)
    function x(t::Real)::Tuple{<:Real,<:Real}
        # Wrap time into one period [0, T)
        τ = mod(t, T)
        Δt = T / 6                 # duration of each edge

        # Which edge are we on? seg ∈ {0,1,2,3,4,5}
        seg = floor(Int, τ / Δt)

        # Local parameter along this edge, s ∈ [0,1)
        s = (τ - seg * Δt) / Δt

        # Precompute hexagon vertices (centered at origin, CCW)
        # k = 0..5
        verts = [(R * cos(k * π/3), R * sin(k * π/3)) for k in 0:5]

        # Start vertex index (1-based)
        i1 = seg + 1
        # End vertex index (wrap around after 6)
        i2 = (seg == 5) ? 1 : (seg + 2)

        x1, y1 = verts[i1]
        x2, y2 = verts[i2]

        # Linear interpolation between vertices
        x = (1 - s) * x1 + s * x2
        y = (1 - s) * y1 + s * y2

        return x, y
    end
    return OneSpinModel([1, 1+0im]/sqrt(2), t, N, B, x->x(x), initialize=initialize)
end


"""

# Arguments
- `n::Int`: number of spins
- `v::Real`: velocity
- `d::Real`: distance between spins in the same channel

# Returns
- `Tuple{Vararg{Function}, n}`: a vector of functions
"""
X_seq_shuttle(n::Int, v::Real, d::Real) = ntuple(k -> (t::Real) -> (v * t + (k - 1) * d, 0.0), n)

"""

# Arguments
- `n::Int`: number of spins
- `v::Real`: velocity
- `d::Real`: distance between two parallel chennels 

# Returns
- `Vector{Function}`: a vector of functions
"""
X_prl_shuttle(n::Int, v::Real, d::Real) =
    ntuple(k -> (t::Real) -> (v * t, (k - 1) * d), n)


"""
# Arguments
- `v::Real`: velocity
- `d1::Real`: distance between the spins in the first channel
- `d2::Real`: distance between the parallel channel
- `θ::Real`:: 

# Returns
- `Vector{Function}`: a vector of functions, only for 3 spins
"""
function X_tri_shuttle(v::Real, d1::Real, d2::Real)
    return return X_tri_shuttle(v, d1, d2, 0.0)
end

function X_tri_shuttle(v::Real, d1::Real, d2::Real, θ::Real)
    f1 = (t::Real) -> (v * t, 0.0)
    f2 = (t::Real) -> (d1 + v * t, 0.0)
    f3 = (t::Real) -> (v * t + θ * d1, d2)
    return (f1, f2, f3)
end

function X_rec_shuttle(v::Real, d1::Real, d2::Real)
    f1 = (t::Real) -> (v * t, 0.0)
    f2 = (t::Real) -> (d1 + v * t, 0.0)
    f3 = (t::Real) -> (d1 + v * t, d2)
    f4 = (t::Real) -> (v * t, d2)
    return (f1, f2, f3, f4)
end

"""

# Arguments
- `t::Real`: time
- `v::Real`: velocity
- `t1::Real`: time to start moving
- `t2::Real`: time to stop moving
"""
function X_padding(t::Real, v::Real, t1::Real, t2::Real)::Real
    if t < t1
        return 0
    elseif t < t2
        return v * (t - t1)
    else
        return v * (t2 - t1)
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
function X_seq_shuttle_delay(n::Int, v::Real, τ::Real, l::Real, d::Real=0.0)
    ntuple(k -> (t::Real) -> X_padding(t - (k - 1) * τ, v, 0, l / v) + (n - k) * d, n)
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
    f1 = (t::Real) -> (X_padding(t, v, 0, l / v), 0.0)
    f2 = (t::Real) -> (X_padding(t, v, τ, l / v + τ), 0.0)
    f3 = (t::Real) -> (X_padding(t, v, 0, l / v), d)
    return (f1, f2, f3)
end

"""

# Arguments
- `v::Real`: velocity
- `τ::Real`: time delay between two spins
- `l::Real`: length of the channel
- `d::Real`: distance between parallel channels

# Returns
- `Vector{Function}`: a vector of functions, only for 4 spins
"""
function X_rec_shuttle_delay(v::Real, τ::Real, l::Real, d::Real)
    f1 = (t::Real) -> (X_padding(t, v, 0, l / v), 0.0)
    f2 = (t::Real) -> (X_padding(t, v, τ, l / v + τ), 0.0)
    f3 = (t::Real) -> (X_padding(t, v, 0, l / v), d)
    f4 = (t::Real) -> (X_padding(t, v, 0, l / v + τ), d)
    return (f1, f2, f3, f4)
end


