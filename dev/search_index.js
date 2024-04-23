var documenterSearchIndex = {"docs":
[{"location":"guide/#Quick-Start","page":"Quick Start","title":"Quick Start","text":"","category":"section"},{"location":"guide/#Generating-a-noise-series-from-a-stochastic-process","page":"Quick Start","title":"Generating a noise series from a stochastic process","text":"","category":"section"},{"location":"guide/","page":"Quick Start","title":"Quick Start","text":"Construct time-position array and define a Gaussian random field","category":"page"},{"location":"guide/","page":"Quick Start","title":"Quick Start","text":"P=collect(zip(t, v.*t))\nB=OrnsteinUhlenbeckField(0,[κₜ,κₓ],σ) ","category":"page"},{"location":"guide/","page":"Quick Start","title":"Quick Start","text":"Define a Gaussian random process (random function) by projecting the Gaussian random field along the time-space array P. Then we can use R() to invoke the process and generating a random time series.","category":"page"},{"location":"guide/","page":"Quick Start","title":"Quick Start","text":"R=RandomFunction(P, B) \nplot(R()) ","category":"page"},{"location":"guide/#Shuttling-of-a-single-spin","page":"Quick Start","title":"Shuttling of a single spin","text":"","category":"section"},{"location":"guide/","page":"Quick Start","title":"Quick Start","text":"Define premeters","category":"page"},{"location":"guide/","page":"Quick Start","title":"Quick Start","text":"σ = sqrt(2) / 20; # variance of the process\nκₜ=1/20; # temporal correlation\nκₓ=1/0.1; # spatial correlation","category":"page"},{"location":"guide/","page":"Quick Start","title":"Quick Start","text":"Create an Ornstein-Uhlenbeck Process","category":"page"},{"location":"guide/","page":"Quick Start","title":"Quick Start","text":"B=OrnsteinUhlenbeckField(0,[κₜ,κₓ],σ)","category":"page"},{"location":"guide/","page":"Quick Start","title":"Quick Start","text":"Consider the shuttling of a single spin at constant velocity v.  We need to specify the initial state, travelling time T and length L=v*T,  and the stochastic noise expreienced by the spin qubit.","category":"page"},{"location":"guide/","page":"Quick Start","title":"Quick Start","text":"T=400; # total time\nL=10; # shuttling length\nv=L/T;","category":"page"},{"location":"guide/","page":"Quick Start","title":"Quick Start","text":"The package provided a simple encapsulation for the single spin shuttling, namely by OneSpinModel.  We need to specify the discretization size and monte-carlo size to create a model.","category":"page"},{"location":"guide/","page":"Quick Start","title":"Quick Start","text":"M = 10000; # monte carlo sampling size\nN=301; # discretization size\nmodel=OneSpinModel(T,L,N,B)\nprintln(model)","category":"page"},{"location":"guide/","page":"Quick Start","title":"Quick Start","text":"The fidelity of the spin state after shuttling can be calculated using numerical integration of the covariance matrix.  ","category":"page"},{"location":"guide/","page":"Quick Start","title":"Quick Start","text":"f1=averagefidelity(model)","category":"page"},{"location":"guide/","page":"Quick Start","title":"Quick Start","text":"The fidelity can also be obtained from Monte-Carlo sampling.","category":"page"},{"location":"guide/","page":"Quick Start","title":"Quick Start","text":"f2, f2_err=sampling(model, fidelity, M)","category":"page"},{"location":"guide/","page":"Quick Start","title":"Quick Start","text":"For the single spin shuttling at constant velocity, analytical solution is also available. ","category":"page"},{"location":"guide/","page":"Quick Start","title":"Quick Start","text":"f3=1/2*(1+W(T,L,B))","category":"page"},{"location":"guide/","page":"Quick Start","title":"Quick Start","text":"We can compare the results form the three methods.","category":"page"},{"location":"guide/","page":"Quick Start","title":"Quick Start","text":"@assert isapprox(f1, f3,rtol=1e-2)\n@assert isapprox(f2, f3, rtol=1e-2) \nprintln(\"NI:\", f1)\nprintln(\"MC:\", f2)\nprintln(\"TH:\", f3)","category":"page"},{"location":"guide/#Shuttling-of-entangled-spin-pairs.","page":"Quick Start","title":"Shuttling of entangled spin pairs.","text":"","category":"section"},{"location":"manual/#APIs","page":"Manual","title":"APIs","text":"","category":"section"},{"location":"manual/","page":"Manual","title":"Manual","text":"CurrentModule = SpinShuttling","category":"page"},{"location":"manual/#Spin-Shuttling-Models","page":"Manual","title":"Spin Shuttling Models","text":"","category":"section"},{"location":"manual/","page":"Manual","title":"Manual","text":"OneSpinModel","category":"page"},{"location":"manual/#SpinShuttling.OneSpinModel","page":"Manual","title":"SpinShuttling.OneSpinModel","text":"General one spin shuttling model initialized at initial state |Ψ₀⟩,  with arbitrary shuttling path x(t). \n\nArguments\n\nΨ::Vector{<:Number}: Initial state of the spin system, the length of the vector must be `2^n\nT::Real: Maximum time\nN::Int: Time discretization\nB::GaussianRandomField: Noise field\nx::Function: Shuttling path\n\n\n\n\n\nOne spin shuttling model initialzied at |Ψ₀⟩=|+⟩. The qubit is shuttled at constant velocity along the path x(t)=L/T*t,  with total time T in μs and length L in μm.\n\n\n\n\n\n","category":"function"},{"location":"manual/","page":"Manual","title":"Manual","text":"OneSpinForthBackModel","category":"page"},{"location":"manual/#SpinShuttling.OneSpinForthBackModel","page":"Manual","title":"SpinShuttling.OneSpinForthBackModel","text":"One spin shuttling model initialzied at |Ψ₀⟩=|+⟩. The qubit is shuttled at constant velocity along a forth-back path  x(t, T, L) = t<T/2 ? 2L/T*t : 2L/T*(T-t),  with total time T in μs and length L in μm.\n\nArguments\n\nT::Real: Maximum time\nL::Real: Length of the path\nN::Int: Time discretization\nB::GaussianRandomField: Noise field\nv::Real: Velocity of the shuttling\n\n\n\n\n\n","category":"function"},{"location":"manual/","page":"Manual","title":"Manual","text":"TwoSpinModel","category":"page"},{"location":"manual/#SpinShuttling.TwoSpinModel","page":"Manual","title":"SpinShuttling.TwoSpinModel","text":"General two spin shuttling model initialized at initial state |Ψ₀⟩, with arbitrary shuttling paths x₁(t), x₂(t).\n\nArguments\n\nΨ::Vector{<:Number}: Initial state of the spin system, the length of the vector must be `2^n\nT::Real: Maximum time\nN::Int: Time discretization\nB::GaussianRandomField: Noise field\nx₁::Function: Shuttling path for the first spin\nx₂::Function: Shuttling path for the second spin\n\n\n\n\n\nTwo spin shuttling model initialized at the singlet state |Ψ₀⟩=1/√2(|↑↓⟩-|↓↑⟩). The qubits are shuttled at constant velocity along the path x₁(t)=L/T₁*t and x₂(t)=L/T₁*(t-T₀).  The delay between the them is T₀ and the total shuttling time is T₁+T₀. It should be noticed that due to the exclusion of fermions, x₁(t) and x₂(t) cannot overlap.\n\n\n\n\n\n","category":"function"},{"location":"manual/","page":"Manual","title":"Manual","text":"TwoSpinParallelModel","category":"page"},{"location":"manual/#SpinShuttling.TwoSpinParallelModel","page":"Manual","title":"SpinShuttling.TwoSpinParallelModel","text":"Two spin shuttling model initialized at the singlet state |Ψ₀⟩=1/√2(|↑↓⟩-|↓↑⟩). The qubits are shuttled at constant velocity along the 2D path  x₁(t)=L/T*t, y₁(t)=0 and x₂(t)=L/T*t, y₂(t)=D. The total shuttling time is T and the length of the path is L in μm.\n\n\n\n\n\n","category":"function"},{"location":"manual/#Fidelity-Metric","page":"Manual","title":"Fidelity Metric","text":"","category":"section"},{"location":"manual/","page":"Manual","title":"Manual","text":"fidelity","category":"page"},{"location":"manual/#SpinShuttling.fidelity","page":"Manual","title":"SpinShuttling.fidelity","text":"Sample a phase integral of the process.  The integrate of a random function should be obtained  from directly summation without using high-order interpolation  (Simpson or trapezoid). \n\n\n\n\n\n","category":"function"},{"location":"manual/","page":"Manual","title":"Manual","text":"sampling","category":"page"},{"location":"manual/#SpinShuttling.sampling","page":"Manual","title":"SpinShuttling.sampling","text":"Monte-Carlo sampling of any objective function.  The function must return Tuple{Real,Real} or Tuple{Vector{<:Real},Vector{<:Real}}\n\nArguments\n\nsamplingfunction::Function: The function to be sampled\nM::Int: Monte-Carlo sampling size\n\nReturns\n\nTuple{Real,Real}: The mean and variance of the sampled function\nTuple{Vector{<:Real},Vector{<:Real}}: The mean and variance of the sampled function\n\nExample\n\nf(x) = x^2\nsampling(f, 1000)\n\nReference\n\nhttps://en.wikipedia.org/wiki/Standarddeviation#Rapidcalculation_methods\n\n\n\n\n\nSampling an observable that defines on a specific spin shuttling model \n\nArguments\n\nmodel::ShuttlingModel: The spin shuttling model\nobjective::Function: The objective function objective(mode::ShuttlingModel; randseq)`\nM::Int: Monte-Carlo sampling size\n\n\n\n\n\n","category":"function"},{"location":"manual/","page":"Manual","title":"Manual","text":"averagefidelity","category":"page"},{"location":"manual/#SpinShuttling.averagefidelity","page":"Manual","title":"SpinShuttling.averagefidelity","text":"Calculate the average fidelity of a spin shuttling model using numerical integration  of the covariance matrix.\n\nArguments\n\nmodel::ShuttlingModel: The spin shuttling model\n\n\n\n\n\n","category":"function"},{"location":"manual/","page":"Manual","title":"Manual","text":"W","category":"page"},{"location":"manual/#SpinShuttling.W","page":"Manual","title":"SpinShuttling.W","text":"Analytical dephasing factor of a one-spin shuttling model.\n\nArguments\n\nT::Real: Total time\nL::Real: Length of the path\nB<:GaussianRandomField: Noise field, Ornstein-Uhlenbeck or Pink-Brownian\npath::Symbol: Path of the shuttling model, :straight or :forthback\n\n\n\n\n\nAnalytical dephasing factor of a sequenced two-spin EPR pair shuttling model.\n\n\n\n\n\n","category":"function"},{"location":"manual/#Stochastics","page":"Manual","title":"Stochastics","text":"","category":"section"},{"location":"manual/","page":"Manual","title":"Manual","text":"OrnsteinUhlenbeckField","category":"page"},{"location":"manual/#SpinShuttling.OrnsteinUhlenbeckField","page":"Manual","title":"SpinShuttling.OrnsteinUhlenbeckField","text":"Ornstein-Uhlenbeck field, the correlation function of which is  σ^2 * exp(-|t₁ - t₂|/θ_t) * exp(-|x₁-x₂|/θ_x)  where t is time and x is position.\n\n\n\n\n\n","category":"type"},{"location":"manual/","page":"Manual","title":"Manual","text":"PinkBrownianField","category":"page"},{"location":"manual/#SpinShuttling.PinkBrownianField","page":"Manual","title":"SpinShuttling.PinkBrownianField","text":"Pink-Brownian Field, the correlation function of which is σ^2 * (expinti(-γ[2]abs(t₁ - t₂)) - expinti(-γ[1]abs(t₁ - t₂)))/log(γ[2]/γ[1]) * exp(-|x₁-x₂|/θ) where expinti is the exponential integral function.\n\n\n\n\n\n","category":"type"},{"location":"manual/","page":"Manual","title":"Manual","text":"RandomFunction","category":"page"},{"location":"manual/#SpinShuttling.RandomFunction","page":"Manual","title":"SpinShuttling.RandomFunction","text":"Similar type of RandomFunction in Mathematica. Can be used to generate a time series on a given time array subject to  a Gaussian random process traced from a Gaussian random field.\n\nArguments\n\nμ::Vector{<:Real}: mean of the process\nP::Vector{<:Point}: time-position array\nΣ::Symmetric{<:Real}: covariance matrices\nC::Cholesky: Cholesky decomposition of the covariance matrices\n\n\n\n\n\n","category":"type"},{"location":"manual/","page":"Manual","title":"Manual","text":"CompositeRandomFunction","category":"page"},{"location":"manual/#SpinShuttling.CompositeRandomFunction","page":"Manual","title":"SpinShuttling.CompositeRandomFunction","text":"Create a new random function composed by a linear combination of random processes. The input random function represents the direct sum of these processes.  The output random function is a tensor contraction from the input.\n\n\n\n\n\n","category":"function"},{"location":"manual/","page":"Manual","title":"Manual","text":"characteristicfunction","category":"page"},{"location":"manual/#SpinShuttling.characteristicfunction","page":"Manual","title":"SpinShuttling.characteristicfunction","text":"Compute the characteristic functional of the process from the  numerical quadrature of the covariance matrix. Using Simpson's rule by default.\n\n\n\n\n\n","category":"function"},{"location":"manual/","page":"Manual","title":"Manual","text":"characteristicvalue","category":"page"},{"location":"manual/#SpinShuttling.characteristicvalue","page":"Manual","title":"SpinShuttling.characteristicvalue","text":"Compute the final phase of the characteristic functional of the process from the  numerical quadrature of the covariance matrix. Using Simpson's rule by default.\n\n\n\n\n\n","category":"function"},{"location":"manual/","page":"Manual","title":"Manual","text":"covariancematrix","category":"page"},{"location":"manual/#SpinShuttling.covariancematrix","page":"Manual","title":"SpinShuttling.covariancematrix","text":"Covariance matrix of a Gaussian random field.  When P₁=P₂, it is the auto-covariance matrix of a Gaussian random process.  When P₁!=P₂, it is the cross-covariance matrix between two Gaussian random processes.\n\nArguments\n\nP₁::Vector{<:Point}: time-position array\nP₂::Vector{<:Point}: time-position array\nprocess::GaussianRandomField: a Gaussian random field\n\n\n\n\n\nAuto-Covariance matrix of a Gaussian random process.\n\nArguments\n\nP::Vector{<:Point}: time-position array\nprocess::GaussianRandomField: a Gaussian random field\n\nReturns\n\nSymmetric{Real}: auto-covariance matrix\n\n\n\n\n\n","category":"function"},{"location":"manual/","page":"Manual","title":"Manual","text":"covariance","category":"page"},{"location":"manual/#SpinShuttling.covariance","page":"Manual","title":"SpinShuttling.covariance","text":"Covariance function of Gaussian random field.\n\nArguments\n\np₁::Point: time-position array\np₂::Point: time-position array\nprocess<:GaussianRandomField: a Gaussian random field, e.g. OrnsteinUhlenbeckField or PinkBrownianField\n\n\n\n\n\n","category":"function"},{"location":"#SpinShuttling.jl","page":"Home","title":"SpinShuttling.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Simulate the multiple-spin shuttling problem under correlated stochastic noise.","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"SpinShuttling can be installed using the Julia package manager. From the Julia REPL, type ] to enter the Pkg REPL mode and run","category":"page"},{"location":"","page":"Home","title":"Home","text":"pkg> add SpinShuttling","category":"page"}]
}
