# Introduction

[`LinearCovarianceModels.jl`](https://github.com/saschatimme/LinearCovarianceModels) is a package for
computing Maximum Likelihood degrees of linear covariance models using numerical nonlinear algebra.
In particular [HomotopyContinuation.jl](www.JuliaHomotopyContinuation.org).

## Introduction by Example


## Reference

```@docs
ml_degree_witness
MLDegreeWitness
MLModel
model
parameters
solutions
is_dual
verify
```

### Models
```@docs
generic_subspace
generic_diagonal
toeplitz
tree
trees
```

### Compute MLE for specific instances
    # solve specific instance
```@docs
mle
solve
covariance_matrix
logl
gradient_logl
hessian_logl
classify_point
```
