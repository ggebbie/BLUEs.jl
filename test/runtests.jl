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
const permil = u"permille"; const K = u"K"; const K² = u"K^2"; m = u"m"; s = u"s";
ENV["UNITFUL_FANCY_EXPONENTS"] = true

include("test_functions.jl")

@testset "BLUEs.jl" begin

    @testset "without units" begin 
        use_units = false
        include("test_estimate.jl")
        #include("test_dimestimate.jl")
    end

    @testset "with units" begin 
        use_units = true
        include("test_estimate.jl")
        #include("test_dimestimate.jl")
    end

end
