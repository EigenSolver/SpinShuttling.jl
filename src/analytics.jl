function C1(β::Real,γ::Real,τ::Real)::Real
    if β<=τ
        return ℯ^(-β-γ-τ)*(1+ℯ^(2β)-2ℯ^τ+2ℯ^(β+τ)*(-β+τ))
    else
        return ℯ^(-β-γ-τ)*(-1+ℯ^τ)^2
    end
end

function C2(β::Real,γ::Real,τ::Real)::Real
    if β<=τ
        return ℯ^(-γ)*β*((ℯ^(-τ)*(-ℯ^β+ℯ^γ))/(β-γ)+(ℯ^(-β)*(γ-2ℯ^β*(β+γ)+ℯ^(β+γ)*(2β+γ)))/(γ*(β+γ)))
    else
        return (ℯ^(-(((β+γ)*(β+3τ))/β))*β*(2ℯ^(β+γ+3τ+(2*γ*τ)/β)*(-1+ℯ^((γ*τ)/β))*β^2-ℯ^((2+(3*γ)/β)*τ)*(-1+ℯ^τ)*γ*(ℯ^τ*(β-γ)+ℯ^(β+γ)*(β+γ))))/((β-γ)*γ*(β+γ))
    end
end

function C3(β::Real,γ::Real,τ::Real)::Real
    if β<=τ
        return (ℯ^(-β-γ-τ)*β^2*(2(-1+ℯ^(2β))*β*γ+(1+ℯ^(2β)+2ℯ^(β+γ)*(-1+γ))*γ^2+β^2*(1+ℯ^(2β)-2ℯ^(β+γ)*(1+γ))))/(β^2-γ^2)^2
    else
        return (1/((β^2-γ^2)^2))ℯ^(-(((β+γ)*(β+τ))/β))*β^2*(ℯ^((γ*τ)/β)*(β-γ)^2+ℯ^((2+γ/β)*τ)*(β-γ)^2-2ℯ^(β+γ+(γ*τ)/β)*(-((-1+γ)*γ^2)+β^2*(1+γ))+2ℯ^(β+γ+τ)*(β^3-β*(-2+γ)*γ-β^2*τ+γ^2*τ))
    end
end

function C4(β::Real,γ::Real,τ::Real)::Real
    if β<=τ
        return ℯ^(-γ)*β*((ℯ^-τ*(-ℯ^β+ℯ^γ))/(β-γ)+(ℯ^(-β)*(γ-2ℯ^β*(β+γ)+ℯ^(β+γ)*(2β+γ)))/(γ*(β+γ)))
    else
        return (ℯ^(-(((β+γ)*(β+τ))/β))*β*(2ℯ^(β+γ+τ)*(-1+ℯ^((γ*τ)/β))*β^2-ℯ^((γ*τ)/β)*(-1+ℯ^τ)*(ℯ^(β+γ)+ℯ^τ)*β*γ+ℯ^((γ*τ)/β)*(-1+ℯ^τ)*(-ℯ^(β+γ)+ℯ^τ)*γ^2))/(β^2*γ-γ^3)
    end
end

function P1(β::Real,γ::Real)::Real
    return -((2β^2 *(1-ℯ^(-β-γ)-β-γ))/(β+γ)^2)
end

function P2(β::Real,γ::Real,τ::Real)::Real
    return 2*(-1+ℯ^-τ+τ)
end

function P3(β::Real,γ::Real,τ::Real)::Real
    return (ℯ^(-β-γ-τ)*(-1+ℯ^(β+γ))*(-1+ℯ^τ)*β)/(β+γ)
end

function F1(β::Real,γ::Real,τ::Real)::Real
    return P1(β,γ)+P2(β,γ,τ)+2*P3(β,γ,τ)
end

function F2(β::Real,γ::Real,τ::Real)::Real
    return C1(β,γ,τ)+C2(β,γ,τ)+C3(β,γ,τ)+C4(β,γ,τ)
end

function Χ(T0::Real,T1::Real,L::Real,κₜ::Real,κₓ::Real,σ::Real)::Real
    τ = κₜ*T0
    β = κₜ*T1
    γ = κₓ*L
    exp(-σ^2/(4*κₜ*κₓ)/κₜ^2*(F1(β, γ, τ)-F2(β, γ, τ)))
end

function Χ(T::Real,L::Real,κₜ::Real,κₓ::Real,σ::Real)::Real
    return exp(- σ^2/(4*κₜ*κₓ)/κₜ^2*P1(κₜ*T,κₓ*L)/2)
end


function ϕ(T::Real, γ::Tuple{Real,Real})::Real
    a(T::Real, γ::Tuple{Real,Real})::Real = expinti(-γ[2]*T)-expinti(-γ[1]*T)
    b(T::Real, γ::Tuple{Real,Real})::Real = (2- exp(-γ[2]*T))/γ[2]-(2- exp(-γ[1]*T))/γ[1]
    c(T::Real, γ::Tuple{Real,Real})::Real = (1- exp(-γ[2]*T))/γ[2]^2-(1- exp(-γ[1]*T))/γ[1]^2
    return (a(T,γ)*T^2-b(T,γ)*T+c(T,γ))/log(γ[2]/γ[1])
end

function Λ(β::Tuple{Real,Real},γ::Real)::Real
    a(β::Real,γ::Real)=(1-γ)*log((β+γ)/β)
    b(β::Real,γ::Real)=γ*(gamma(0,β+γ)+(1-exp(-(β+γ)))/(β+γ))
    c(β::Real,γ::Real)=exp(-γ)*expint(-β)-expint(-(β+γ))
    return 1/γ^2*log(β[2]/β[1])*(a(β[2],γ)-a(β[1],γ)+b(β[2],γ)-b(β[1],γ)+c(β[2],γ)-c(β[1],γ))
end
