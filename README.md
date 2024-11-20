# SpinShuttling.jl
[![](https://img.shields.io/badge/Documentation-dev-blue.svg)](https://eigensolver.github.io/SpinShuttling.jl/dev/)
![action-ci](https://github.com/eigensolver/SpinShuttling.jl/actions/workflows/runtest.yml/badge.svg)

Simulate the open-system dynamics of a moving spin under correlated stochastic noise.

Install the package using the Julia package manager.
From the Julia REPL, type `]` to enter the Pkg REPL mode and run

```julia
pkg> add SpinShuttling
```

## Example: 

Consider the sequential shuttling of a pair of entangled spins preparing in Bell state. If the distance between the two spins is small enough, then the noise experienced by them will have a non-zero covariance. The spatially correlated noise can be modeled by a 2D Gaussian random field, $W(t, x)$. Since the wavefunction of the spin is highly constrained by the moving potential well. We can describe the position of a spin carrier (electron or holes), as a function of time $x(t)$. Such a trajectory will determine a route-specific projection on the 2D field $W(t, x(t))$ which gives the noise experienced by the spin. 

```julia
L=10; σ =sqrt(2)/20; M=5000; N=501; T1=100; T0=25; κₜ=1/20; κₓ=1/0.1;

B=OrnsteinUhlenbeckField(0,[κₜ,κₓ],σ)
model=TwoSpinSequentialModel(T0, T1, L, N, B)

Ψ= model.Ψ
ρ=Ψ*Ψ'
w=dephasingmatrix(model)

ρt=w.*ρ

println(ρ)
```

```
f1=statefidelity(model)
f2, f2_err=sampling(model, statefidelity, M)
f3=1/2*(1+W(T0, T1, L,B))

println("NI:", f1)
println("MC:", f2)
println("TH:", f3)
```

![Sequential shuttling of a pair of entangled qubit](./docs/src/assets/animation2spins.gif)


The quickstart page has a longer example. See the [documentation](https://eigensolver.github.io/SpinShuttling.jl/dev/) for details.

If you use SpinShuttling.jl in your research, please [cite](CITATION.bib) our work.
