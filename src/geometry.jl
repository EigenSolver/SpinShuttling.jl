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

function OneSpinSquareModel(T::Real, L::Real, N::Int, B::GaussianRandomField)
    v = L/T
    function x(t::Real)::Tuple{Real,Real}
        if 0 ≤ t < T
            return (v * t, 0.0)
        elseif T ≤ t < 2T
            return (L, v*(t-T))
        elseif 2T ≤ t < 3T
            return (L-v*(t-2T), L)
        elseif 3T ≤ t ≤ 4T
            return (0.0, L-v*(t-3T))
        else
            return (0.0,0.0)
        end
    end

    Ψ = 1 / √2 .* [1+0im,1+0im]
    return OneSpinModel(Ψ, 4T, N, B, x)
end

function ExchangeOnlyQubit(T::Real, L::Real, N::Int, B::GaussianRandomField)
    
end