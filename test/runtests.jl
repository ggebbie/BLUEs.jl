using Revise
using BLUEs
using Test
using LinearAlgebra
using Statistics
# includet("test_functions.jl")

@testset "BLUEs.jl" begin

    @testset "BLUEs core" begin
        include("test_blues.jl")
    end 
        
    @testset "extensions without units" begin 
        global use_units = false
        include("test_blues_algebraic_arrays.jl")
        include("test_dimensional_data.jl")
    end

    @testset "extensions with units" begin
        using Unitful
        # using UnitfulLinearAlgebra
        # using ToeplitzMatrices
        # using SparseArrays
        const K = u"K"; const K² = u"K^2"; m = u"m"; s = u"s";
        const permil = Unitful.FixedUnits(u"permille")

        ENV["UNITFUL_FANCY_EXPONENTS"] = true

        global use_units = true
        include("test_blues_unitful.jl")
        include("test_algebraic_arrays.jl")
        include("test_dimensional_data.jl") 
        include("test_unitful_linear_algebra.jl")
    end

end
