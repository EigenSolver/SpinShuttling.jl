# Quick Start


## Basic model
```math
H_{noise}(t)=g \mu_B \sum_j \left[B_0(x^c_j)+\tilde{B}(x^c_{j},t)\right] S_j^z
```

We assume the electrons are adiabatically transported in moving-wave potential with their wave functions well localized at $x_j^c$. Then, the effective magnetic noise $\tilde{B}(x_j^c, t)$ can be modeled by a Gaussian random field. 

In the case of pure dephasing, the system dynamics can be explicitly written out. 

```math
U(t)=\exp(-\frac{i}{\hbar} \int_0^t H_{noise}(\tau)\mathrm{d} \tau)
```

If we label a realization of the random process by $k$, then the pure dephasing channel can be expressed as a mixing unitary process.
```math
\mathcal{E}(\rho)=\frac{1}{M} \sum_{k=1}^M U_k \rho U_k^\dagger
=\sum_k E_k \rho E_k^\dagger, \quad E_k= U_k /\sqrt{M}
```

The pure dephasing of such a system can be analytically solved and efficiently obtained via a matrix of dephasing factors, while more general system dynamics involving other interactions can be numerically solved by Monte-Carlo sampling.

```math
\mathcal{H}(t)=H_{noise}(t)+H_{int}(t)
```

## Generating a noise series from a stochastic field
Import the package.
```@example quickstart
using SpinShuttling
using Plots
```
We first define a 2D Ornstein-Uhlenbeck field, specified by three parameters. 
```@example quickstart
κₜ=1/20; # inverse correlation time
κₓ=1/0.1; # inverse correlation length
σ = 1; # noise strength
B=OrnsteinUhlenbeckField(0,[κₜ,κₓ],σ); # mean is zero
nothing
```
Specify a trajectory `(t,x(t))` on the 2D plane; in this example case, it's just a line. 
```@example quickstart
t=range(1,20,200); # time step
v=2; #velocity
P=collect(zip(t, v.*t));
```
A Gaussian random process (random function) can be obtained by projecting the Gaussian random field along the time-space array `P`. Then, we can use `R()` to invoke the process and generate a random time series.
```@example quickstart
R=GaussianRandomFunction(P, B) 
plot(t, R(), xlabel="t", ylabel="B(t)", size=(400,300)) 
```


## Shuttling of a single spin
We can follow the above approach to define a single-spin shuttling model.
```@example quickstart
σ = sqrt(2) / 20; # variance of the process
κₜ=1/20; # temporal correlation
κₓ=1/0.1; # spatial correlation
B=OrnsteinUhlenbeckField(0,[κₜ,κₓ],σ);

nothing
```

Consider the shuttling of a single spin at constant velocity `v`. 
We need to specify the initial state, traveling time `T`, and length `L=v*T`, 
and the stochastic noise experienced by the spin qubit.
```@example quickstart
T=400; # total time
L=10; # shuttling length
v=L/T;
```
The package provided a simple encapsulation for the single spin shuttling, namely
by `OneSpinModel`. 
We must specify the discretization and Monte-Carlo sizes to create a model.
```@example quickstart
M = 10000; # monte carlo sampling size
N=301; # discretization size
model=OneSpinModel(T,L,N,B)
println(model)
```
The `println` function provides us with an overview of the model. It's a single spin shuttling problem with the initial state `Ψ₀` and an Ornstein-Uhlenbeck noise. The total time of simulation is `T`, which is discretized into `N` steps.

The effective noise of this spin qubit is completely characterized by its covariance matrix.  
```@example quickstart
heatmap(collect(sqrt.(model.R.Σ)), title="sqrt cov, 1-spin one-way shuttling", 
size=(400,300), 
xlabel="t1", ylabel="t2", dpi=300,
right_margin=5Plots.mm)
```
The state fidelity after such a quantum process can be obtained using numerical integration of the covariance matrix.  
```@example quickstart
f1=statefidelity(model); # direct integration

f2, f2_err=sampling(model, statefidelity, M); # Monte-Carlo sampling
```
An analytical solution is also available for single-spin shuttling at a constant velocity. 
```@example quickstart
f3=1/2*(1+W(T,L,B));
```
We can compare the results form the three methods and check their consistency.
```@example quickstart
@assert isapprox(f1, f3,rtol=1e-2)
@assert isapprox(f2, f3, rtol=1e-2) 
println("NI:", f1)
println("MC:", f2)
println("TH:", f3)
```

The pure dephasing channel is computationally simple and can be represented by a dephasing matrix $w$, such that the final density state after the channel is given by $\mathcal{E}(\rho)=w \odot\rho$. Here $\odot$ is an element-wise Hadmard product. 
```@example quickstart
Ψ= model.Ψ
ρ=Ψ*Ψ'
w=dephasingmatrix(model)
ρt=w.*ρ
```

We can check that the fidelity between the initial and final state is consistent with the results above. 
```@example quickstart
f=(Ψ'*ρt*Ψ)
```

## Dephasing of entangled spin pairs during shuttling. 
Following the approach above, we can further explore the multi-spin system. 
The general abstraction on such a problem is given by the data type `ShuttlingModel`.  
```julia
ShuttlingModel(n, Ψ, T, N, B, X, R)
```
Users can freely define an n-qubit system with an arbitrary initial state. Here, `X=[x1,x2...]` is an array of functions, containing spin trajectories $x_i(t)$. `R` is a random function constructed from the specific noise process.  

One more example is the shuttling of two spin pairs. We can define such a two-spin system. 
```@example quickstart
L=10; σ =sqrt(2)/20; M=5000; N=501; T1=100; T0=25; κₜ=1/20; κₓ=1/0.1;
B=OrnsteinUhlenbeckField(0,[κₜ,κₓ],σ)
model=TwoSpinSequentialModel(T0, T1, L, N, B)
println(model)
```

The system is initialized in the Bell state $\ket{\Psi^-}$. 
The model encapsulated a model of two spins shuttled in a sequential manner, as we can see from the two trajectories `x1(t)` and `x2(t)`. One spin goes first and then follows another, with a waiting time of `T0`. This is modeled by the piece-wise linear trajectories. 
We can see some quite interesting covariance from such a system.
```@example quickstart
plot(model.R.P[1:N,1], label="x1(t)",
xlabel="t", ylabel="x",size=(400,300), dpi=300
)
plot!(model.R.P[N+1:2N,1], label="x2(t)")
```


```@example quickstart
heatmap(collect(model.R.Σ)*1e3, title="covariance, 2-spin sequential shuttling", 
size=(400,300), 
xlabel="t1", ylabel="t2", dpi=300,
right_margin=5Plots.mm)
```

We can check the dephasing of the system and calculate its fidelity as before. 
```@example quickstart 
f1=statefidelity(model)
f2, f2_err=sampling(model, statefidelity, M)
f3=1/2*(1+W(T0, T1, L,B))

println("NI:", f1)
println("MC:", f2)
println("TH:", f3)
```


The density matrix after the channel can be given by the dephasing matrix.
```@example quickstart
Ψ= model.Ψ
ρ=Ψ*Ψ'
w=dephasingmatrix(model)

ρt=w.*ρ
```
