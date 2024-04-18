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

function P4(β::Real,γ::Real)::Real
    return ((ℯ^(-2β)-1)*(γ/β)-2*ℯ^(-β-γ)+ℯ^(-2β)+1)/(1-γ^2/β^2)
end

"""
Ancillary function for the dephasing of the sequential shuttling pf 
Bell state under OU sheets of noise
"""
function F1(β::Real,γ::Real,τ::Real)::Real
    return P1(β,γ)+P2(β,γ,τ)+2*P3(β,γ,τ)
end

function F2(β::Real,γ::Real,τ::Real)::Real
    return C1(β,γ,τ)+C2(β,γ,τ)+C3(β,γ,τ)+C4(β,γ,τ)
end


"""
Ancillary function for the dephasing of the Pink-Brownian noise
"""
function F3(β::Tuple{Real,Real},γ::Real)::Real
    F(β::Real)=1/2*(expinti(-β)+(1-exp(-β))/β^2+(exp(-β)-2)/β)
    F(β::Real,γ::Real)::Real=1/γ^2*(exp(-γ)*expinti(-β)+(γ-1)*(expinti(-β-γ)-log((β+γ)/β))-γ*((1-exp(-β-γ))/(β+γ)))
    if γ==0 # pure 1/f noise
        return (F(β[2])-F(β[1]))/log(β[2]/β[1])
    else
        return (F(β[2],γ)-F(β[1],γ))/log(β[2]/β[1])
    end
end


