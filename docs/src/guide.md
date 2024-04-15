# Quick Start

## Generating a noise series from a stochastic process
Construct time-position array and define a Gaussian random field
```julia
P=hcat(t, v.*t) 
B=OrnsteinUhlenbeckField(0,[κₜ,κₓ],σ) 
```
Define a Gaussian random process (random function) by projecting the Gaussian random field along the time-space array `P`. Then we can use `R()` to invoke the process and generating a random time series.
```julia
R=RandomFunction(P, B) 
plot(R()) 
```

## Shuttling of a single spin
Define premeters
```julia
σ = sqrt(2) / 20; # variance of the process
κₜ=1/20; # temporal correlation
κₓ=1/0.1; # spatial correlation
```
Create an Ornstein-Uhlenbeck Process
```julia
B=OrnsteinUhlenbeckField(0,[κₜ,κₓ],σ)
```

Consider the shuttling of a single spin at constant velocity `v`. 
We need to specify the initial state, travelling time `T` and length `L=v*T`, 
and the stochastic noise expreienced by the spin qubit.
```julia
T=400; # total time
L=10; # shuttling length
v=L/T;
```
The package provided a simple encapsulation for the single spin shuttling, namely
by `OneSpinModel`. 
We need to specify the discretization size and monte-carlo size to create a model.
```julia
M = 10000; # monte carlo sampling size
N=301; # discretization size
model=OneSpinModel(T,L,N,B)
println(model)
```
The fidelity of the spin state after shuttling can be calculated using numerical integration of the covariance matrix.  
```julia
f1=averagefidelity(model)
```
The fidelity can also be obtained from Monte-Carlo sampling.
```julia
f2, f2_err=sampling(model, fidelity, M)
```
For the single spin shuttling at constant velocity, analytical solution is also available. 
```julia
f3=1/2*(1+W(T,L,B))
```
We can compare the results form the three methods.
```julia
@assert isapprox(f1, f3,rtol=1e-2)
@assert isapprox(f2, f3, rtol=1e-2) 
println("NI:", f1)
println("MC:", f2)
println("TH:", f3)
```

## Shuttling of entangled spin pairs. 