# SpinShuttling.jl

*Simulate the multiple-spin shuttling problem under correlated stochastic noise.*

## Installation

SpinShuttling can be installed using the Julia package manager.
From the Julia REPL, type `]` to enter the Pkg REPL mode and run

```
pkg> add SpinShuttling
```

## APIs

```@meta
CurrentModule = SpinShuttling
```

### Spin Shuttling Models

```@docs
OneSpinModel
```

```@docs
OneSpinForthBackModel
```

```@docs
TwoSpinModel
```

```@docs
TwoSpinParallelModel
```

### Fidelity Metric

```@docs
fidelity
```

```@docs
sampling
```

```@docs
averagefidelity
```

```@docs
W
```

### Stochastics

```@docs
OrnsteinUhlenbeckField
```

```@docs
PinkBrownianField
```

```@docs
RandomFunction
```

```@docs
CompositeRandomFunction
```

```@docs
characteristicfunction
```

```@docs
characteristicvalue
```

```@docs
covariancematrix
```

```@docs
covariance
```
