# Introduction

[`LinearCovarianceModels.jl`](https://github.com/saschatimme/LinearCovarianceModels) is a package for
computing Maximum Likelihood degrees of linear covariance models using numerical nonlinear algebra.
In particular [HomotopyContinuation.jl](https://www.JuliaHomotopyContinuation.org).

## Introduction by Example


## Linear Covariance Models

The linear covariance models are wrapped in the `LCModel` type:
```@docs
LCModel
model
dim
```

### Default models
The following linear covariance models are provided by default
```@docs
generic_subspace
generic_diagonal
toeplitz
tree
trees
```


## ML Degree

```@docs
ml_degree_witness
MLDegreeWitness
ml_degree
parameters
solutions
is_dual
verify
```

## Compute MLE for specific instances

```@docs
mle
critical_points
covariance_matrix
logl
gradient_logl
hessian_logl
classify_point
```
