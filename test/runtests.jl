using Revise, BLUEs
using Test
using LinearAlgebra, Statistics, Unitful, UnitfulLinearAlgebra, Measurements
const permil = u"permille"; const K = u"K"; const	K² = u"K^2"; m = u"m"; s = u"s"; MMatrix = BestMultipliableMatrix
ENV["UNITFUL_FANCY_EXPONENTS"] = true
MMatrix = BestMultipliableMatrix

@testset "BLUEs.jl" begin

    @testset "just determined left pseudoinverse" begin
        N = 2
        for M in 2:4
            σₓ = rand()
            # exact = false to work
            E = MMatrix(randn(M,N),fill(m,M),fill(m,N),exact=true)
            Cnn⁻¹ = Diagonal(fill(σₓ^-1,M),unitrange(E).^-1,unitrange(E),exact=true)
            x = randn(N)m
            y = E*x
            x̃ = solve(y,E,Cnn⁻¹)
            @test x ≈ x̃.v
        end
    end

    @testset "trend analysis" begin

        M = 10  # number of obs
        t = collect(0:1:M-1)s

        a = randn()m # intercept
        b = randn()m/s # slope

        y = a .+ b.*t .+ randn(M)m

        E = MMatrix(hcat(ones(M),ustrip.(t)),fill(m,M),[m,m/s],exact=true)
        Cnn⁻¹ = Diagonal(fill(1.0,M),fill(m^-1,M),fill(m,M),exact=true)
        x̃ = solve(y,E,Cnn⁻¹)
        #E⁺ = (E'*(W⁻¹*E)) \ (E'*W⁻¹)
        #@test x ≈ x̃.v
        
    end
    

end
