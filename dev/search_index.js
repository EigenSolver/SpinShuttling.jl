var documenterSearchIndex = {"docs":
[{"location":"guide/#Quick-Start","page":"Quick Start","title":"Quick Start","text":"","category":"section"},{"location":"guide/#Shuttling-of-a-single-spin","page":"Quick Start","title":"Shuttling of a single spin","text":"","category":"section"},{"location":"guide/","page":"Quick Start","title":"Quick Start","text":"Define premeters","category":"page"},{"location":"guide/","page":"Quick Start","title":"Quick Start","text":"σ = sqrt(2) / 20; # variance of the process\nκₜ=1/20; # temporal correlation\nκₓ=1/0.1; # spatial correlation","category":"page"},{"location":"guide/","page":"Quick Start","title":"Quick Start","text":"Create an Ornstein-Uhlenbeck Process","category":"page"},{"location":"guide/","page":"Quick Start","title":"Quick Start","text":"B=OrnsteinUhlenbeckField(0,[κₜ,κₓ],σ)","category":"page"},{"location":"guide/","page":"Quick Start","title":"Quick Start","text":"Consider the shuttling of a single spin at constant velocity v.  We need to specify the initial state, travelling time T and length L=v*T,  and the stochastic noise expreienced by the spin qubit.","category":"page"},{"location":"guide/","page":"Quick Start","title":"Quick Start","text":"T=400; # total time\nL=10; # shuttling length\nv=L/T;","category":"page"},{"location":"guide/","page":"Quick Start","title":"Quick Start","text":"The package provided a simple encapsulation for the single spin shuttling, namely by OneSpinModel.  We need to specify the discretization size and monte-carlo size to create a model.","category":"page"},{"location":"guide/","page":"Quick Start","title":"Quick Start","text":"M = 10000; # monte carlo sampling size\nN=301; # discretization size\nmodel=OneSpinModel(T,L,M,N,B)\nprintln(model)","category":"page"},{"location":"guide/","page":"Quick Start","title":"Quick Start","text":"The fidelity of the spin state after shuttling can be calculated using numerical integration of the covariance matrix.  ","category":"page"},{"location":"guide/","page":"Quick Start","title":"Quick Start","text":"f1=averagefidelity(model)","category":"page"},{"location":"guide/","page":"Quick Start","title":"Quick Start","text":"The fidelity can also be obtained from Monte-Carlo sampling.","category":"page"},{"location":"guide/","page":"Quick Start","title":"Quick Start","text":"f2, f2_err=sampling(model, fidelity)","category":"page"},{"location":"guide/","page":"Quick Start","title":"Quick Start","text":"For the single spin shuttling at constant velocity, analytical solution is also available. ","category":"page"},{"location":"guide/","page":"Quick Start","title":"Quick Start","text":"f3=1/2*(1+Χ(T,L,B))","category":"page"},{"location":"guide/","page":"Quick Start","title":"Quick Start","text":"We can compare the results form the three methods.","category":"page"},{"location":"guide/","page":"Quick Start","title":"Quick Start","text":"@assert isapprox(f1, f3,rtol=1e-2)\n@assert isapprox(f2, f3, rtol=1e-2) \nprintln(\"NI:\", f1)\nprintln(\"MC:\", f2)\nprintln(\"TH:\", f3)","category":"page"},{"location":"guide/#Shuttling-of-entangled-spin-pairs.","page":"Quick Start","title":"Shuttling of entangled spin pairs.","text":"","category":"section"},{"location":"#SpinShuttling.jl","page":"Home","title":"SpinShuttling.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Simulate the multiple-spin shuttling problem under correlated stochastic noise.","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"SpinShuttling can be installed using the Julia package manager. From the Julia REPL, type ] to enter the Pkg REPL mode and run","category":"page"},{"location":"","page":"Home","title":"Home","text":"pkg> add SpinShuttling","category":"page"},{"location":"#APIs","page":"Home","title":"APIs","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = SpinShuttling","category":"page"},{"location":"","page":"Home","title":"Home","text":"OneSpinModel","category":"page"},{"location":"#SpinShuttling.OneSpinModel","page":"Home","title":"SpinShuttling.OneSpinModel","text":"General one spin shuttling model initialized at initial state |Ψ₀⟩,  with arbitrary shuttling path x(t). \n\nArguments\n\nΨ::Vector{<:Number}: Initial state of the spin system, the length of the vector must be `2^n\nT::Real: Maximum time\nM::Int:  Monte-Carlo sampling size\nN::Int: Time discretization\nB::GaussianRandomField: Noise field\nx::Function: Shuttling path\n\n\n\n\n\nOne spin shuttling model initialzied at |Ψ₀⟩=|+⟩. The qubit is shuttled at constant velocity along the path x(t)=L/T*t,  with total time T in μs and length L in μm.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"TwoSpinModel","category":"page"},{"location":"#SpinShuttling.TwoSpinModel","page":"Home","title":"SpinShuttling.TwoSpinModel","text":"General two spin shuttling model initialized at initial state |Ψ₀⟩, with arbitrary shuttling paths x₁(t), x₂(t).\n\nArguments\n\nΨ::Vector{<:Number}: Initial state of the spin system, the length of the vector must be `2^n\nT::Real: Maximum time\nM::Int:  Monte-Carlo sampling size\nN::Int: Time discretization\nB::GaussianRandomField: Noise field\nx₁::Function: Shuttling path for the first spin\nx₂::Function: Shuttling path for the second spin\n\n\n\n\n\nTwo spin shuttling model initialized at the singlet state |Ψ₀⟩=1/√2(|↑↓⟩-|↓↑⟩). The qubits are shuttled at constant velocity along the path x₁(t)=L/T₁*t and x₂(t)=L/T₁*(t-T₀).  The delay between the them is T₀ and the total shuttling time is T₁+T₀. It should be noticed that due to the exclusion of fermions, x₁(t) and x₂(t) cannot overlap.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"fidelity","category":"page"},{"location":"#SpinShuttling.fidelity","page":"Home","title":"SpinShuttling.fidelity","text":"Sample a phase integral of the process.  The integrate of a random function should be obtained  from directly summation without using high-order interpolation  (Simpson or trapezoid). \n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"sampling","category":"page"},{"location":"#SpinShuttling.sampling","page":"Home","title":"SpinShuttling.sampling","text":"Monte-Carlo sampling of any objective function.  The function must return Tuple{Real,Real} or Tuple{Vector{<:Real},Vector{<:Real}}\n\n\n\n\n\nSampling an observable that defines on a specific spin shuttling model  objective(mode, randseq, vector)\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"averagefidelity","category":"page"},{"location":"#SpinShuttling.averagefidelity","page":"Home","title":"SpinShuttling.averagefidelity","text":"Calculate the average fidelity of a spin shuttling model using numerical integration  of the covariance matrix.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"OrnsteinUhlenbeckField","category":"page"},{"location":"#SpinShuttling.OrnsteinUhlenbeckField","page":"Home","title":"SpinShuttling.OrnsteinUhlenbeckField","text":"Ornstein-Uhlenbeck field, the correlation function of which is  σ^2 * exp(-|t₁ - t₂|/θ_t) * exp(-|x₁-x₂|/θ_x)  where t is time and x is position.\n\n\n\n\n\n","category":"type"},{"location":"","page":"Home","title":"Home","text":"PinkBrownianField","category":"page"},{"location":"#SpinShuttling.PinkBrownianField","page":"Home","title":"SpinShuttling.PinkBrownianField","text":"Pink-Brownian Field, the correlation function of which is σ^2 * (expinti(-γ[2]abs(t₁ - t₂)) - expinti(-γ[1]abs(t₁ - t₂)))/log(γ[2]/γ[1]) * exp(-|x₁-x₂|/θ) where expinti is the exponential integral function.\n\n\n\n\n\n","category":"type"},{"location":"","page":"Home","title":"Home","text":"RandomFunction","category":"page"},{"location":"#SpinShuttling.RandomFunction","page":"Home","title":"SpinShuttling.RandomFunction","text":"Similar type of RandomFunction in Mathematica. Generate a time series on a given time array subject to  a Gaussian random process traced from a Gaussian random field.\n\n\n\n\n\n","category":"type"},{"location":"","page":"Home","title":"Home","text":"Χ","category":"page"},{"location":"#SpinShuttling.Χ","page":"Home","title":"SpinShuttling.Χ","text":"Theoretical fidelity of a sequenced two-spin EPR pair shuttling model.\n\n\n\n\n\nTheoretical fidelity of a one-spin shuttling model.\n\n\n\n\n\nTheoretical fidelity of a one-spin shuttling model for a pink noise.\n\n\n\n\n\n","category":"function"}]
}
