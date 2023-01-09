using BLUEs
using Test
using LinearAlgebra, Statistics, Unitful, UnitfulLinearAlgebra, Measurements
const permil = u"permille"; const K = u"K"; const	K² = u"K^2"; m = u"m"; s = u"s"; MMatrix = BestMultipliableMatrix
ENV["UNITFUL_FANCY_EXPONENTS"] = true
MMatrix = BestMultipliableMatrix

@testset "BLUEs.jl" begin

    @testset "error propagation" begin

        a = randn(5) .± rand(5)
        E = randn(5,5)

        aerr = Measurements.uncertainty.(a);
        #x = Estimate(Measurements.value.(a),
        #             aerr*transpose(aerr));

        x = Estimate(Measurements.value.(a),
                     Diagonal(aerr.^2))

        @test Measurements.value.(E*a) ≈ (E*x).v
        @test Measurements.uncertainty.(E*a) ≈ (E*x).σ

    end

    @testset "mixed signals: dimensionless matrix" begin
        N = 2
        for M in 2:4
            σₓ = rand()
            # exact = false to work
            E = MMatrix(randn(M,N),fill(m,M),fill(m,N),exact=true)
            Cnn⁻¹ = Diagonal(fill(σₓ^-1,M),unitrange(E).^-1,unitrange(E),exact=true)
            x = randn(N)m
            y = E*x

            problem = OverdeterminedProblem(y,E,Cnn⁻¹)
            #x̃ = solve(y,E,Cnn⁻¹)
            x̃ = solve(problem)
            @test x ≈ x̃.v
            @test cost(x̃,problem) < 1e-5 # no noise in obs

            @test x ≈ pinv(problem) * y # inefficient way to solve problem
        end
    end

    @testset "trend analysis: left-uniform matrix" begin

        M = 10  # number of obs
        t = collect(0:1:M-1)s

        a = randn()m # intercept
        b = randn()m/s # slope

        y = a .+ b.*t .+ randn(M)m

        E = MMatrix(hcat(ones(M),ustrip.(t)),fill(m,M),[m,m/s],exact=true)
        Cnn⁻¹ = Diagonal(fill(1.0,M),fill(m^-1,M),fill(m,M),exact=true)

        problem = OverdeterminedProblem(y,E,Cnn⁻¹)
        x̃ = solve(problem)
        @test cost(x̃,problem) < 3M # rough guide, could get unlucky and not pass

    end

    @testset "left-uniform problem with prior info" begin
	y = [-1.9permil]
	σₙ = 0.2permil
	a = -0.24permil*K^-1
	γδ = 1.0permil^-2 # to keep units correct, have two γ (tapering) variables
	γT = 1.0K^-2
        E = MMatrix(ustrip.([1 a]),[permil],[permil,K],exact=true) # problem with exact E and error propagation
        x₀ = [-1.0permil, 4.0K]

        # "overdetermined, problem 1.4" 
        Cxx⁻¹ = Diagonal(ustrip.([γδ,γT]),[permil^-1,K^-1],[permil,K],exact=true)
	Cnn⁻¹  = Diagonal([σₙ.^-2],[permil^-1],[permil])
        oproblem = OverdeterminedProblem(y,E,Cnn⁻¹,Cxx⁻¹,x₀)

        # "underdetermined, problem 2.1" 
	Cnn  = Diagonal([σₙ.^2],[permil],[permil^-1])
        Cxx = Diagonal(ustrip.([γδ,γT].^-1),[permil,K],[permil^-1,K^-1],exact=true)
        uproblem = UnderdeterminedProblem(y,E,Cnn,Cxx,x₀)

        x̃1 = solve(oproblem)
        @test cost(x̃1,oproblem) < 1 # rough guide, coul

        x̃2 = solve(uproblem)
        @test cost(x̃2,uproblem) < 1 # rough guide, could ge
        # same answer both ways?
        @test cost(x̃2,uproblem) ≈ cost(x̃1,oproblem)
    end

    @testset "polynomial fitting, problem 2.3" begin

    end

    @testset "overdetermined problem for mean with autocovariance, problem 4.1" begin

    end

    @testset "objective mapping, problem 4.3" begin

    end

    # additional problem: 5.1 model of exponential decay 

end
