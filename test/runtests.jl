using Revise
using BLUEs
using Test
using LinearAlgebra
using Statistics

includet("test_functions.jl")

@testset "BLUEs.jl" begin

    @testset "BLUEs Estimate type" begin
        include("test_estimate.jl")
    end 
        
    @testset "extensions without units" begin 
        include("test_algebraic_arrays.jl")
        include("test_dimensional_data.jl")
    end

    @testset "extensions with unitful" begin
        global K = u"K"; global K² = u"K^2"; global m = u"m"; global s = u"s";
        global permil = Unitful.FixedUnits(u"permille")
        include("test_estimate_unitful.jl")
        include("test_unitful_linear_algebra.jl")
        include("test_algebraic_arrays_unitful.jl")

        # issue: case 1 works, case 2 works, case 3 doesn't work
        include("test_dimensional_data_unitful.jl")

        # needs updating
       # include("test_DD_AA.jl")
    end

    # @testset "extensions with DynamicQuantities" begin
    #     global K = u"K"; global K² = u"K^2"; global m = u"m"; global s = u"s";
    #     global permil = Unitful.FixedUnits(u"permille")
    #     include("test_estimate_dynamicquantities.jl")
    # end
end
