# BLUEs

Best Linear Unbiased Estimators. I guess that's why they call them the BLUEs. 

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ggebbie.github.io/BLUEs.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ggebbie.github.io/BLUEs.jl/dev/)
[![Build Status](https://github.com/ggebbie/BLUEs.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ggebbie/BLUEs.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ggebbie/BLUEs.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ggebbie/BLUEs.jl)

## Motivation

A wide range of techniques in mathematics, physics, and engineering, including least-squares methods, Kalman filters, and adjoint smoothers, rely upon combining information from models and observations. More generally, information from multiple sources is combined, where uncertainty information is used to determine how to best form the combination. Julia's ecosystem already includes utilities for describing information and its uncertainty in the Measurements.jl package, but covariance information cannot be added. Here a vector of `Measurement`s is extended into an `Estimate` that is described by a vector of central estimates and their uncertainty (i.e., covariance) matrix. 

## Workflow

Consider a simple workflow for combining two forms of information.
```julia
M = 5 # number of observations

# assign a standard error to each observation using a variable `a` with type `Measurement` 
a = randn(M) .± rand(M)

# retrieve the central estimate and standard error
aval = Measurements.value.(a)
aerr = Measurements.uncertainty.(a);

# define an `Estimate` that permits covariance/uncertainty information to be saved
x  = Estimate(aval, aerr) # just provide standard error

# inspect covariance/uncertainty matrix
x.P

# a second `Estimate` can be formed even more simply
y = Estimate(randn(M), rand(M))

# combine the two estimates `x` and `y` 
# where the algorithm is the Best Linear Unbiased Estimator (BLUE) 
z = combine(x, y)
```

## Features:

- Inputs are streamlined by bundling all information with its central estimate and uncertainty. The Estimate type has this information and it is now used for solutions, first guesses, and observations.
	
- It is recommended for `combine` to replace `solve` with the idea that information from two Estimates is combined to make a new Estimate. `solve` previously required an underdetermined or overdetermined problem to be specified, but now there is limited logic to do this automatically.
	
- Operations can be performed on arbitrary `AlgebraicArray`s using the AlgebraicArrays.jl package.
	
- This package consistently composes with `DimArray`s from DimensionalData.jl including for coefficients.
	
- `combine` does not require the observational operation to be in linear or matrix form. It currently accepts a Function argument which makes a priori impulse reponse calculations unnecessary.

- Units are included in an extension and are optional.

## Future features:

Support for UnitfulLinearAlgebra.jl is experimental at this time.
