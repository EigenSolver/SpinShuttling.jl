using LinearAlgebra
include("../src/integration.jl")

##
@testset begin "test numerical integration"
    f = @. x -> sqrt(x)
    x1 = 0.0
    x2 = 6.0
    n = 600
    r = 4 * sqrt(6)
    y = f.(range(x1, x2, n+1))
    h = (x2 - x1) / n
    s1=integrate(y, h)
    s2=integrate(f, x1, x2, n+1)
    @test s1==s2
    # println(abs(s1-r))
    @test isapprox(s1, r; rtol=1e-5)

    s3=integrate(y, h, method=:trapezoid)
    s4=integrate(f, x1, x2, n, method=:trapezoid)
    @test isapprox(s3, s4, rtol=1e-5)
    # println(abs(s3-r))
    @test isapprox(s3, r; rtol=1e-2)

    s5 = quadgk(f, x1, x2)[1]
    # println(abs(s5-r))
    @test isapprox(s5, r; rtol=1e-5)

    g = (x, y) -> x^2 + y^2
    x_range=(-0.0,2.0,21);y_range=(-2,2.0,11);
    @test isapprox(integrate(g, x_range,y_range), 64/3; rtol=1e-8);
    z=[g(x,y) for x in range(x_range...) for y in range(y_range...)]
    z=reshape(z, y_range[3], x_range[3])
    h_x=(x_range[2]-x_range[1])/(x_range[3]-1);
    h_y=(y_range[2]-y_range[1])/(y_range[3]-1);
    @test isapprox(integrate(g, (-0.0,2.0,21),(-2,2.0,11)), 64/3; rtol=1e-8);
    @test z isa Matrix{<:Real} && h_x isa Real && h_y isa Real
    @test isapprox(integrate(z, h_x, h_y), 64/3; rtol=1e-8);
end
##
@testset "cone" begin
    g = (x, y) -> sqrt(x^2 + y^2)
    x_range = (0.0, 1.0, 21)
    y_range = (0.0, 1.0, 21)
    expected_result = 0.765196
    z = [g(x, y) for x in range(x_range...) for y in range(y_range...)]
    z = reshape(z, y_range[3], x_range[3])
    h_x = (x_range[2] - x_range[1]) / (x_range[3] - 1)
    h_y = (y_range[2] - y_range[1]) / (y_range[3] - 1)
    @test isapprox(integrate(z, h_x, h_y), expected_result; rtol=1e-3)
end

##
@testset "saddle point" begin
    g = (x, y) -> x^2 - y^2
    x_range = (-1.0, 1.0, 21)
    y_range = (-1.0, 1.0, 21)
    expected_result = 0
    z = [g(x, y) for x in range(x_range...) for y in range(y_range...)]
    z = reshape(z, y_range[3], x_range[3])
    h_x = (x_range[2] - x_range[1]) / (x_range[3] - 1)
    h_y = (y_range[2] - y_range[1]) / (y_range[3] - 1)
    @test isapprox(integrate(z, h_x, h_y), expected_result; atol=1e-3)
end

##
@testset "Gaussian-Kronrod vs Special Function" begin
    λ1=0.0001;
    λ2=10000;
    t=0.00000000001;
    (f1,f1_err)=quadgk(λ->exp(-λ*t)/λ, λ1, λ2)
    println("quad err", f1_err)
    f2=expinti(-λ2*t)-expinti(-λ1*t)
    println(f1)
    println(f2)
    @test isapprox(f1,f2, rtol=1e-6)
end