# SpinShuttling.jl 
[![forthebadge](https://forthebadge.com/badges/built-with-science.svg)](https://forthebadge.com)

[![](https://img.shields.io/badge/Documentation-dev-blue.svg)](https://eigensolver.github.io/SpinShuttling.jl/dev/)
![action-ci](https://github.com/eigensolver/SpinShuttling.jl/actions/workflows/runtest.yml/badge.svg)


Simulate the open-system dynamics of a moving spin under correlated stochastic noise.

Install the package using the Julia package manager.
From the Julia REPL, type `]` to enter the Pkg REPL mode and run

```julia
pkg> add SpinShuttling
```

Consider two spin qubits sited on a close pair of quantum dots (QDots). It can be easily imagined that the two electrons or holes interact with the same ensemble of charge defects, nuclear spins, and other disorders. The same environment will decohere our qubits with similar or, more precisely, correlated noises. 

In a more general situation, we can consider a register of spin qubits sited on a 2D array of quantum dots. The statistical fluctuations, or noise, associated with each spin qubit will also share similar properties due to their spatial neighborhoods. That means the noises experienced by any two qubits will have a non-zero cross-covariance. As the distance between the two qubits goes further, their noise will be less correlated since they will share less common environments. 
For multiple spin qubits sited within a microscopic nano device, their collective dephasing behavior will be non-trivially impacted by the covariance between noises. 

From a statistical point of view, this physical picture implies that noises associated with the spins are correlated not only in time but also in space. 
A Gaussian random field, $W(t, \boldsymbol{x})$, can be used to model the temporally and spatially correlated noise. 
As suggested by the name "quantum dot", the wavefunction of the single electron or hole in a QDot is highly confined by the potential well. Thus, it is reasonable to assume a "position", a dot $\boldsymbol{x}$, for an electronic qubit, which is just the center of its wavefunction.  

The same random field model of noise can also be used to treat the dephasing of spin shuttling, where a spin qubit is conveyed by a moving potential well. 
The moving of the dot can be captured by a simple function, \boldsymbol{x}(t). Then, he noise experienced by the spin can be given the route-specific projection on the noise field $W(t, \boldsymbol{x}(t))$. 



## Example: 

Sequential shuttling of a pair of entangled qubits. 

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
![Sequential shuttling of a pair of entangled qubit](./docs/src/assets/animation2spins.gif)

```
f1=statefidelity(model)
f2, f2_err=sampling(model, statefidelity, M)
f3=1/2*(1+W(T0, T1, L,B))

println("NI:", f1)
println("MC:", f2)
println("TH:", f3)
```

The quickstart page has a longer example. See the [documentation](https://eigensolver.github.io/SpinShuttling.jl/dev/) for details.

If you use SpinShuttling.jl in your research, please [cite](CITATION.bib) our work.

