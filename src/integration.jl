const ArrayOrSubArray{T,N} = Union{Array{T,N}, SubArray{T,N}}

"""
1D Simpson integral of function `f(x)` on a given array of `y=f(x₁), f(x₂)...` with constant
increment `h`

# Arguments
- `y::Vector{<:Real}`: f(x).
- `h::Real`: the step of integral.
- `method::Symbol=:simpson`: the method of integration 

"""
function integrate(y::ArrayOrSubArray{<:Real,1}, h::Real; method::Symbol=:simpson)::Real
    n = length(y)-1
    if method==:simpson
        n % 2 == 0 || error("`y` length (number of intervals) must be odd")
        s = sum(y[1:2:n] + 4*y[2:2:n] + y[3:2:n+1])
        return h/3 * s
    elseif method==:trapezoid
        s = sum(y[1:n]+y[2:n+1])
        return h / 2 * s
    else
        error("invalid integration method specified!")
    end
end

"""
1D integral of a function `f(x)` over the range [a, b] using Simpson's rule or the Trapezoidal rule.

# Arguments
- `f::Function`: The function to be integrated. It should accept a single real number and return a real number.
- `a::Real`: The lower bound of the integration range.
- `b::Real`: The upper bound of the integration range.
- `n::Int`: The number of points at which to evaluate `f(x)`. For Simpson's rule, `n` should be an odd number.
- `method::Symbol=:simpson`: The numerical integration method to use. Options are `:simpson` for Simpson's rule and `:trapezoid` for the Trapezoidal rule.

# Returns
- `Real`: The estimated value of the integral of `f(x)` over [a, b].

# Examples
```julia
result = integrate(sin, 0, π, 101, method=:simpson)
```
"""
function integrate(f::Function, a::Real, b::Real, n::Int; 
    method::Symbol=:simpson)::Real
    h = (b - a) / (n-1)
    y = f.(range(a,b,n))
    return integrate(y, h, method=method)
end

function integrate(f::Function, x_range::Tuple{Real,Real,Int}; method::Symbol=:simpson)::Real
    return integrate(f, x_range..., method=method)
end

"""
2D integral of a function `f(x, y)` over the ranges [x_min, x_max] and [y_min, y_max] using Simpson's rule or the Trapezoidal rule applied successively in each dimension.

# Arguments
- `f::Function`: The function to be integrated. It should accept two real numbers (x, y) and return a real number.
- `x_range::Tuple{Real, Real, Int}`: A tuple representing the range and number of points for the x-dimension: (x_min, x_max, num_points_x).
- `y_range::Tuple{Real, Real, Int}`: A tuple representing the range and number of points for the y-dimension: (y_min, y_max, num_points_y).
- `method::Symbol=:simpson`: The numerical integration method to use. Options are `:simpson` for Simpson's rule and `:trapezoid` for the Trapezoidal rule.

# Returns
- `Real`: The estimated value of the integral of `f(x, y)` over the defined 2D area.

# Examples
```julia
result = integrate((x, y) -> x * y, (0, 1, 50), (0, 1, 50), method=:simpson)
```
"""
function integrate(f::Function, x_range::Tuple{Real,Real,Int}, y_range::Tuple{Real,Real,Int}; 
    method::Symbol=:simpson)::Real
    g = y-> integrate(x->f(x,y), x_range, method = method)
    return integrate(g, y_range, method = method)
end

"""
2D Simpson integral of function `f(x, y)` on a given matrix of `z=f(x_i, y_j)` with constant
increments `h_x` and `h_y`.
...
# Arguments
- `z::Matrix{<:Real}`: f(x, y).
- `h_x::Real`: the step size of the integral in the x direction.
- `h_y::Real`: the step size of the integral in the y direction.
- `method::Symbol=:simpson`: the method of integration.
...
"""
function integrate(z::ArrayOrSubArray{<:Real,2}, h_x::Real, h_y::Real; method::Symbol=:simpson)::Real
    nrows, ncols = size(z)
    # Integrate along x direction for each y
    integral_x_direction = [integrate((@view z[:, j]), h_x, method=method) for j = 1:ncols]
    # Integrate the result in y direction
    return integrate(integral_x_direction, h_y, method=method)
end

integrate(z::ArrayOrSubArray{<:Real,2}, h; method::Symbol=:simpson)=integrate(z, h, h; method=method)

"""
Special methods for the double integral on symmetric matrix with singularity on diagonal entries.
"""
function integrate(z::Symmetric, h::Real)::Real
    n, _ = size(z)
    @assert n%2 == 1 "The matrix must be of odd size"
    m=(n+1)÷2
    _integrate(x::ArrayOrSubArray) = integrate(x, h; method = :trapezoid)
    # Integrate along upper half trapzoid
    int_upper = [_integrate((@view z[j:n, j])) for j = 1:m]|> _integrate
    # Integrate along lower half trapzoid
    int_lower = [_integrate((@view z[1:j, j])) for j = m:n]|> _integrate
    # Integrate the duplicated box
    int_box = _integrate((@view z[m:n, 1:m]))
    return 2*(int_upper + int_lower - int_box)
end