
"""

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
One spin shuttling with a square trace 
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
    return OneSpinRectangleModel(T, L, L, N, B)
end