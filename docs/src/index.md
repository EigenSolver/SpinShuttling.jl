# SpinShuttling.jl

*Simulate the multiple-spin shuttling problem under correlated stochastic noise.*

## Installation
<!-- ### Github Repository -->
`SpinShuttling.jl` can be installed by cloning the repository from github.
```shell
git clone https://github.com/EigenSolver/SpinShuttling.jl.git
```
Go to the directory of the project in terminal. 
```shell
cd ./SpinShuttling.jl
```
From the Julia REPL, type `]` to enter the Pkg REPL mode and run
```julia
pkg> add .
```
<!-- ### Julia Package Manager
From the Julia REPL, type `]` to enter the Pkg REPL mode and run
```shell
pkg> add SpinShuttling
``` -->

## What does this package do
This package provides a set of abstractions and numericals tools to simulating dynamics of multi-spin system under correlated noises, based on the Gaussian random field approach. 


While we provided specially optimized models for spin shuttling problems. This package can also be used to simulate more general *correlated open-quantum dynamics*.

The following two approaches are supported.
- Direct numerical integration for pure dephasing.
- Monte-Carlo sampling for open-system dynamics. 

## About spin shuttling
Spin shuttling has recently emerged as a pivotal technology for large-scale semiconductor quantum computing. By transporting qubits between quantum dots, spin shuttling enables entanglement between non-neighboring qubits, which is essential for quantum error correction. However, the spin qubit becomes decohered by magnetic noise during the shuttling process. Since the noise varies in time and space in a correlated manner, the associated dephasing in a system of several entangled spins often cannot be treated using the standard theory of random processes and requires more advanced mathematical instruments. 
In our latest work, we employ the Gaussian random field (GRF) to model the magnetic noise varying in both space and time. By projecting trajectories of spin qubits onto the random field, the correlated noises experienced by multi-spin system can be effectively captured, enabling further study on spin dynamics, dephasing and quantum information applications. 
