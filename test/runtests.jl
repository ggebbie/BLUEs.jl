using Revise
using BLUEs
using Test
using LinearAlgebra
using Statistics
using Unitful
using UnitfulLinearAlgebra
using ToeplitzMatrices
using SparseArrays
using DimensionalData
using DimensionalData:@dim
using AlgebraicArrays 
const K = u"K"; const K² = u"K^2"; m = u"m"; s = u"s";
const permil = Unitful.FixedUnits(u"permille")

ENV["UNITFUL_FANCY_EXPONENTS"] = true

# requires Revise
includet("test_functions.jl")

@testset "BLUEs.jl" begin

    @testset "without units" begin 
        global use_units = false
        include("test_estimate.jl")
        include("test_algebraic_arrays.jl")
        include("test_dimensional_data.jl")
    end

    @testset "with units" begin 
        global use_units = true
        include("test_estimate.jl")
        include("test_algebraic_arrays.jl")
        include("test_dimensional_data.jl") 
        include("test_unitful_linear_algebra.jl")
         #include("test_DD_ULA.jl")
    end

end
