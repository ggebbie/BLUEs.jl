# BLUEs

Best Linear Unbiased Estimators. I guess that's why they call them the BLUEs. 

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ggebbie.github.io/BLUEs.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ggebbie.github.io/BLUEs.jl/dev/)
[![Build Status](https://github.com/ggebbie/BLUEs.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ggebbie/BLUEs.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ggebbie/BLUEs.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ggebbie/BLUEs.jl)

## Features:

    Inputs are streamlined by bundling all information with its central estimate and uncertainty. The Estimate type has this information and it is now used for solutions, first guesses, and observations.
	
    Recommended for `combine` to replace `solve` with the idea that information from two Estimates is combined to make a new Estimate. `solve` previously required an underdetermined or overdetermined problem to be specified, but now there is limited logic to do this automatically.
	
    Operations can be performed on arbitrary `AlgebraicArray`s using the AlgebraicArrays.jl package.
	
    Consistently composes with `DimArray`s, including for coefficients
	
    `combine` does not require the observational operation to be in linear or matrix form. It currently accepts a Function argument which makes a priori impulse reponse calculations unnecessary.

## Examples

Three objective mapping notebooks now exist in the notebooks branch.

- `2.9_objectivemapping.jl` - pedagogical example from Dynamical Insights From Data class

- `objectivemapping_blues.jl` - objective mapping that uses BLUEs to solve the Gauss-Markov problem

- `objectivemapping_BLUES_AlgebraicArrays.jl` - full featured notebook using whole ecosystem, including BLUEs for the Gauss-Markov problem, AlgebraicArrays.jl to handle conversion from a 2D grid to matrices and vectors, and DimensionalData.jl for labeling the variables.

## Future features:

    Units will be included in an extension and are optional

