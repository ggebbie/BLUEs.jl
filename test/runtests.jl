using Revise
using BLUEs
using Test
using LinearAlgebra
using Statistics
using Unitful
includet("test_functions.jl")

@testset "BLUEs.jl" begin

    @testset "BLUEs core" begin
        include("test_blues.jl")
    end 
        
    @testset "extensions without units" begin 
        include("test_blues_algebraic_arrays.jl")
        include("test_blues_dimensional_data.jl")
    end

    @testset "extensions with units" begin
        global K = u"K"; global K² = u"K^2"; global m = u"m"; global s = u"s";
        global permil = Unitful.FixedUnits(u"permille")
        include("test_blues_unitful.jl")
        include("test_blues_algebraic_arrays_unitful.jl")
        include("test_blues_dimensional_data_unitful.jl") 
        include("test_unitful_linear_algebra.jl")
    end
end
