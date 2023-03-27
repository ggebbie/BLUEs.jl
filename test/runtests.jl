#using Revise
using BLUEs
using Test
using LinearAlgebra
using Statistics
using Unitful
using UnitfulLinearAlgebra
using Measurements
using ToeplitzMatrices
using SparseArrays
using DimensionalData
using DimensionalData:@dim
const permil = u"permille"; const K = u"K"; const K² = u"K^2"; m = u"m"; s = u"s";
ENV["UNITFUL_FANCY_EXPONENTS"] = true

include("test_functions.jl")

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
            E = UnitfulMatrix(randn(M,N),fill(m,M),fill(m,N),exact=true)
            # NOTE: unitrange, unitdomain have different units. requires kludge below.
            Cnn⁻¹ = Diagonal(fill(σₓ^-1,M),unitrange(E).^-1,unitrange(E).^1,exact=true)
            x = UnitfulMatrix(randn(N)m)
            y = E*x

            problem = OverdeterminedProblem(y,E,Cnn⁻¹)
            #x̃ = solve(y,E,Cnn⁻¹)
            x̃ = solve(problem,alg=:hessian)
            x̃ = solve(problem,alg=:textbook)
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

        y = UnitfulMatrix(a .+ b.*t .+ randn(M)m)

        E = UnitfulMatrix(hcat(ones(M),ustrip.(t)),fill(m,M),[m,m/s],exact=true)
        Cnn⁻¹ = Diagonal(fill(1.0,M),fill(m^-1,M),fill(m,M),exact=true)

        problem = OverdeterminedProblem(y,E,Cnn⁻¹)
        x̃ = solve(problem,alg=:textbook)
        x̃1 = solve(problem,alg=:hessian)
        @test cost(x̃,problem) < 3M # rough guide, could get unlucky and not pass

    end

    @testset "left-uniform problem with prior info" begin
	y = UnitfulMatrix([-1.9permil])
	σₙ = 0.2permil
	a = -0.24permil*K^-1
	γδ = 1.0permil^-2 # to keep units correct, have two γ (tapering) variables
	γT = 1.0K^-2
        E = UnitfulMatrix(ustrip.([1 a]),[permil],[permil,K],exact=true) # problem with exact E and error propagation
        x₀ = UnitfulMatrix([-1.0permil, 4.0K])

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
        g = u"g"
        kg = u"kg"
        d = u"d"
	M = 50
        t = (1:M)d
        Eparent = ustrip.(hcat(t.^0, t, t.^2, t.^3))
        E = UnitfulMatrix(Eparent,fill(g/kg,M),[g/kg,g/kg/d,g/kg/d^2,g/kg/d^3],exact=true)

        σₙ = 0.1g/kg
        Cₙₙ = Diagonal(fill(ustrip(σₙ^2),M),fill(g/kg,M),fill(kg/g,M),exact=true) 
        Cₙₙ¹² = cholesky(Cₙₙ)
        Cₙₙ⁻¹ = Diagonal(fill(ustrip(σₙ^-2),M),fill(kg/g,M),fill(g/kg,M),exact=true) 
        Cₙₙ⁻¹² = cholesky(Cₙₙ⁻¹)

	γ = [1.0e1kg^2/g^2, 1.0e2kg^2*d^2/g^2, 1.0e3kg^2*d^4/g^2, 1.0e4kg^2*d^6/g^2]

	Cxx⁻¹ = Diagonal(ustrip.(γ),[kg/g,kg*d/g,kg*d^2/g,kg*d^3/g],[g/kg,g/kg/d,g/kg/d^2,g/kg/d^3],exact=true)
        Cxx = inv(Cxx⁻¹)
        Cxx¹² = cholesky(Cxx)

        N = size(Cxx⁻¹,1)
        x₀ = UnitfulMatrix(zeros(N).*unitdomain(Cxx⁻¹))
        x = Cxx¹².L*randn(N)
        y = E*x
        oproblem = OverdeterminedProblem(y,E,Cₙₙ⁻¹,Cxx⁻¹,x₀)

        # not perfect data fit b.c. of prior info
        x̃ = solve(oproblem,alg=:hessian)
        x̃ = solve(oproblem,alg=:textbook)
        @test cost(x̃,oproblem) < 3M
    end

    @testset "overdetermined problem for mean with autocovariance, problem 4.1" begin

    end

    @testset "objective mapping, problem 4.3" begin
        
        yr = u"yr"; cm = u"cm"
        τ = range(0.0yr,5.0yr,step=0.1yr)
        ρ = exp.(-τ.^2/(1yr)^2)
        n = length(ρ)
        Cxx = UnitfulMatrix(SymmetricToeplitz(ρ),fill(cm,n),fill(cm^-1,n),exact=true) + Diagonal(fill(1e-6,n),   fill(cm,n),fill(cm^-1,n),exact=true)

        M = 11
        σₙ = 0.1cm
        Cnn = Diagonal(fill(ustrip(σₙ),M),fill(cm,M),fill(cm^-1,M),exact=true)

        Enm = sparse(1:M,1:5:n,fill(1.0,M))
        E = UnitfulMatrix(Enm,fill(cm,M),fill(cm,n),exact=true)

        Cxx¹² = cholesky(Cxx)
        x₀ = UnitfulMatrix(zeros(n)cm)
        x = Cxx¹².L*randn(n)
        y = E*x

        uproblem = UnderdeterminedProblem(y,E,Cnn,Cxx,x₀)

        x̃ = solve(uproblem)
        @test cost(x̃,uproblem) < 3M 

    end

    # additional problem: 5.1 model of exponential decay 
    @testset "overdetermined named tuple E,y" begin
        N = 2
        M = 1
        σₓ = rand()
        # exact = false to work
        E1 = UnitfulMatrix(randn(M,N),fill(m,M),fill(m,N),exact=true)
        E2 = UnitfulMatrix(randn(M,N),fill(m,M),fill(m,N),exact=true)
        E = (one=E1,two=E2)

        Cnn⁻¹1 = Diagonal(fill(σₓ^-1,M),unitrange(E1).^-1,unitrange(E1).^1,exact=true)

        Cnn⁻¹ = (one=Cnn⁻¹1, two =Cnn⁻¹1)
        x = UnitfulMatrix(randn(N)m) 

        # create perfect data
        y = E*x

        problem = OverdeterminedProblem(y,E,Cnn⁻¹)

        #x̃ = solve(y,E,Cnn⁻¹)
        x̃ = solve(problem,alg=:hessian)
        
        @test x ≈ x̃.v # no noise in obs
        @test cost(x̃,problem) < 1e-5 # no noise in obs

        #@test x ≈ pinv(problem) * y # inefficient way to solve problem

        # contaminate observations, check if error bars are correct
    end

    @testset "source water inversion: obs at one time, no circulation lag" begin

        #using DimensionalData
        #using DimensionalData: @dim
        @dim YearCE "years Common Era"
        @dim SurfaceRegion "surface location"
        @dim InteriorLocation "interior location"

        M = 5
        E,x = random_source_water_matrix_vector_pair(M)

        # Run model to predict interior location temperature
        #y = uconvert.(K,E*x)
        y = E*x

        # do slices work?
        E[At(:loc1),At(:NATL)]
        E[:,At(:ANT)]

        # invert for x.  
        x̃ = E\y

        σₙ = 1.0
        Cnndims = (first(dims(E)),first(dims(E)))
        #Cnn⁻¹ = Diagonal(fill(σₓ^-1,M),unitrange(E).^-1,unitrange(E).^1,dims=Cnndims,exact=true)
        Cnn⁻¹ = UnitfulDimMatrix(Diagonal(fill(σₙ^-1,M)),unitrange(E).^-1,unitrange(E).^1,dims=Cnndims,exact=true)

        problem = OverdeterminedProblem(y,E,Cnn⁻¹)
        x̃ = solve(problem,alg=:hessian)
        x̃ = solve(problem,alg=:textbook)

        #@test x ≈ x̃.v
        @test cost(x̃,problem) < 1e-5 # no noise in ob
        @test within(x̃.v,x,1.0e-5)
    end

    @testset "source water inversion: obs at one time, many surface regions, with circulation lag" begin

        @dim YearCE "years Common Era"
        @dim SurfaceRegion "surface location"
        @dim InteriorLocation "interior location"

        nτ = 5 # how much of a lag is possible?
        lags = (0:(nτ-1))yr

        # the dimensions of the state variable
        surfaceregions = [:NATL,:ANT,:SUBANT]
        years = (1990:2000)yr

        n = length(surfaceregions)

        M = source_water_matrix_with_lag(surfaceregions,lags,years)

        x= source_water_solution(surfaceregions,years)

        # Run model to predict interior location temperature
        # convolve E and x
        y = convolve(x,M)

        # could also use this format
        y = predictobs(convolve,x,M)

        ## invert for y for x̃
        # Given, M and y. Make first guess for x.
        # add adjustment
        # DimArray is good enough. This is an array, not necessarily a matrix.
        x₀ = DimArray(zeros(size(x))K,(Ti(years),last(dims(M))))

        # probe to get E matrix. Use function convolve
        E = impulseresponse(convolve,x₀,M)

        # Does E matrix work properly?
        ỹ = E*UnitfulMatrix(vec(x))
        @test isapprox(y,getindexqty(ỹ,1))

        x̂ = E\UnitfulMatrix([y]) 
        @test isapprox(y,getindexqty(E*x̂,1))

        # now in a position to use BLUEs to solve
        # should handle matrix left divide with
        # unitful scalars in UnitfulLinearAlgebra
        
        σₙ = 0.01
        σₓ = 100.0

        Cnn = UnitfulMatrix(Diagonal(fill(σₙ,length(y))),fill(unit.(y).^1,length(y)),fill(unit.(y).^-1,length(y)),exact=false)

        Cxx = UnitfulMatrix(Diagonal(fill(σₓ,length(x₀))),unit.(x₀)[:],unit.(x₀)[:].^-1,exact=false)

        problem = UnderdeterminedProblem(UnitfulMatrix([y]),E,Cnn,Cxx,x₀)
        x̃ = solve(problem)
        @test within(y[1],getindexqty(E*x̃.v,1),3σₙ) # within 3-sigma

        @test cost(x̃,problem) < 5e-2 # no noise in ob
    end

    @testset "source water inversion: obs timeseries, many surface regions, with circulation lag" begin

        @dim YearCE "years Common Era"
        @dim SurfaceRegion "surface location"
        @dim InteriorLocation "interior location"
        yr = u"yr"
        m = 5 # how much of a lag is possible?
        M,x = source_water_DimArray_vector_pair_with_lag(m)
        n = size(M,2) # surface regions
        Tx = first(dims(x)) #years = (1990:2000)yr
        x₀ = DimArray(zeros(size(x))K,(Tx,last(dims(M))))

        # get synthetic observations
        y = convolve(x,M,Tx)

        E = impulseresponse(convolve,x₀,M,Tx)
                        
        # Does E matrix work properly?
        ỹ = E*UnitfulMatrix(vec(x))
        for jj in eachindex(vec(y))
            @test isapprox.(vec(y)[jj],getindexqty(ỹ,jj))
        end

        x̂ = E\UnitfulMatrix(vec(y))
        for jj in eachindex(vec(y))
            @test isapprox.(vec(y)[jj],getindexqty(E*x̂,jj))
        end

        σₙ = 0.01
        σₓ = 100.0
        #Cnndims = (first(dims(E)),first(dims(E)))
        #Cnn⁻¹ = Diagonal(fill(σₓ^-1,M),unitrange(E).^-1,unitrange(E).^1,dims=Cnndims,exact=true)
        Cnn = UnitfulMatrix(Diagonal(fill(σₙ,length(y))),vec(unit.(y)).^1,vec(unit.(y)).^-1,exact=true)

        Cxx = UnitfulMatrix(Diagonal(fill(σₓ,length(x₀))),vec(unit.(x₀)),vec(unit.(x₀)).^-1,exact=true)

        #problem = UnderdeterminedProblem(UnitfulMatrix([y]),E,Cnn)
        #problem = UnderdeterminedProblem(UnitfulMatrix([y]),E,Cnn,Cxx,UnitfulMatrix(x₀[:]))
        problem = UnderdeterminedProblem(UnitfulMatrix(vec(y)),E,Cnn,Cxx,x₀)
        x̃ = solve(problem)
        for jj in eachindex(vec(y))
            @test within(y[jj],getindexqty(E*x̃.v,jj),3σₙ) # within 3-sigma
        end

        # no noise in obs but some control penalty
        @test cost(x̃,problem) < 0.5 # ad-hoc choice

    end

    @testset "source water inversion: obs timeseries, many surface regions, with circulation lag" begin

        @dim YearCE "years Common Era"
        @dim SurfaceRegion "surface location"
        @dim InteriorLocation "interior location"
        yr = u"yr"
        m = 5 # how much of a lag is possible?
        M,x = source_water_DimArray_vector_pair_with_lag(m)
        n = size(M,2) # surface regions
        Tx = first(dims(x)) #years = (1990:2000)yr
        x₀ = DimArray(zeros(size(x))K,(Tx,last(dims(M))))

        # get synthetic observations
        y = convolve(x,M,Tx)

        E = impulseresponse(convolve,x₀,M,Tx)
                        
        # Does E matrix work properly?
        ỹ = E*UnitfulMatrix(vec(x))
        for jj in eachindex(vec(y))
            @test isapprox.(vec(y)[jj],getindexqty(ỹ,jj))
        end

        x̂ = E\UnitfulMatrix(vec(y))
        for jj in eachindex(vec(y))
            @test isapprox.(vec(y)[jj],getindexqty(E*x̂,jj))
        end

        σₙ = 0.01
        σₓ = 100.0
        #Cnndims = (first(dims(E)),first(dims(E)))
        #Cnn⁻¹ = Diagonal(fill(σₓ^-1,M),unitrange(E).^-1,unitrange(E).^1,dims=Cnndims,exact=true)
        Cnn = UnitfulMatrix(Diagonal(fill(σₙ,length(y))),vec(unit.(y)).^1,vec(unit.(y)).^-1,exact=true)

        Cxx = UnitfulMatrix(Diagonal(fill(σₓ,length(x₀))),vec(unit.(x₀)),vec(unit.(x₀)).^-1,exact=true)

        #problem = UnderdeterminedProblem(UnitfulMatrix([y]),E,Cnn)
        #problem = UnderdeterminedProblem(UnitfulMatrix([y]),E,Cnn,Cxx,UnitfulMatrix(x₀[:]))
        problem = UnderdeterminedProblem(UnitfulMatrix(vec(y)),E,Cnn,Cxx,x₀)
        x̃ = solve(problem)
        for jj in eachindex(vec(y))
            @test within(y[jj],getindexqty(E*x̃.v,jj),3σₙ) # within 3-sigma
        end

        # no noise in obs but some control penalty
        @test cost(x̃,problem) < 0.5 # ad-hoc choice
    end

    @testset "source water inversion: many obs timeseries, many surface regions, with circulation lag" begin

        @dim YearCE "years Common Era"
        @dim SurfaceRegion "surface location"
        @dim InteriorLocation "interior location"
        yr = u"yr"

        nτ = 5 # how much of a lag is possible?
        lags = (0:(nτ-1))yr

        m = 6 # how many observational locations?
        interiorlocs = [Symbol("loc"*string(nloc)) for nloc = 1:m]

        # the dimensions of the state variable
        surfaceregions = [:NATL,:ANT,:SUBANT]
        years = (1990:2000)yr

        n = length(surfaceregions)

        # pre-allocate a 3D DimArray
        M = DimArray(zeros(nτ,n,m),(Ti((0:(nτ-1))yr),SurfaceRegion(surfaceregions),InteriorLocation(interiorlocs)))

        # fill it at each location
        for loc in InteriorLocation(interiorlocs)
            M[:,:,At(Symbol(loc))] = source_water_matrix_with_lag(surfaceregions,lags,years)
        end

        # true solution
        x= source_water_solution(surfaceregions,years)

        # first guess of solution 
        x₀ = DimArray(zeros(size(x))K,dims(x))

        # get synthetic observations
        y = convolve(x,M,Tx)

        E = impulseresponse(convolve,x₀,M,Tx)
                        
        # Does E matrix work properly?
        ỹ = E*UnitfulMatrix(vec(x))
        for jj in eachindex(vec(y))
            @test isapprox.(vec(y)[jj],getindexqty(ỹ,jj))
        end

        x̂ = E\UnitfulMatrix(vec(y))
        for jj in eachindex(vec(y))
            @test isapprox.(vec(y)[jj],getindexqty(E*x̂,jj))
        end

        σₙ = 0.01
        σₓ = 100.0
        #Cnndims = (first(dims(E)),first(dims(E)))
        #Cnn⁻¹ = Diagonal(fill(σₓ^-1,M),unitrange(E).^-1,unitrange(E).^1,dims=Cnndims,exact=true)
        Cnn = UnitfulMatrix(Diagonal(fill(σₙ,length(y))),vec(unit.(y)).^1,vec(unit.(y)).^-1,exact=true)

        Cxx = UnitfulMatrix(Diagonal(fill(σₓ,length(x₀))),vec(unit.(x₀)),vec(unit.(x₀)).^-1,exact=true)

        problem = UnderdeterminedProblem(UnitfulMatrix(vec(y)),E,Cnn,Cxx,x₀)
        x̃ = solve(problem)
        for jj in eachindex(vec(y))
            @test within(y[jj],getindexqty(E*x̃.v,jj),3σₙ) # within 3-sigma
        end

        # no noise in obs but some control penalty
        @test cost(x̃,problem) < 0.5 # ad-hoc choice
    end

end
