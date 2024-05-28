#using Revise
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

    @testset "error propagation" begin
        using Measurements

        a = randn(5) .± rand(5)
        E = randn(5,5)

        aerr = Measurements.uncertainty.(a);
        x = Estimate(Measurements.value.(a),
                     Diagonal(aerr.^2))

        @test Measurements.value.(E*a) ≈ (E*x).v
        @test Measurements.uncertainty.(E*a) ≈ (E*x).σ

    end

    @testset "issue 41" begin
        #what if the type is a Quantity but no units

        @dim YearCE "years Common Era"
        @dim StateVariable "state variable"
        years = (1990:2000)u"yr"
        statevariables = [:θ, :δ¹⁸O] 

        nx = length(statevariables); ny = length(years)
        xdims = (StateVariable(statevariables),
            YearCE(years))
            
        xparent = Matrix{Quantity}(undef, nx, ny)
        xx = 1
        stateunits = [u"K", u"permille"]
        for xx in 1:nx
            for yy in 1:ny
                xparent[xx,yy] = Quantity(rand(),stateunits[xx])
            end
        end
        y3 = DimArray(xparent, xdims)

        Pparent = Matrix(undef,nx,ny)
        for xx in 1:nx
            for yy in 1:ny
                colparent = Matrix{Quantity}(undef, nx, ny)
                for xx2 in 1:nx
                    for yy2 in 1:ny
                        if xx2 == xx && yy2 == yy
                            colparent[xx2,yy2] = Quantity(1.0,stateunits[xx]*stateunits[xx2])
                        else
                            colparent[xx2,yy2] = Quantity(0.0,stateunits[xx]*stateunits[xx2])
                        end
                    end
                end
                Pparent[xx, yy] = DimArray(colparent, xdims)
            end
        end
        P3 = DimArray(Pparent, xdims)
        de3 = DimEstimate(y3, P3)
    end
    
    @testset "mixed signals: dimensionless matrix" begin
        N = 2
        for M in 2:4
            σₓ = rand()
            # exact = false to work
            E = UnitfulMatrix(randn(M,N),fill(m,M),fill(m,N),exact=true)
            Cnn⁻¹ = Diagonal(fill(σₓ^-1,M),unitrange(E).^-1,unitrange(E))
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
        Cnn⁻¹ = Diagonal(fill(1.0,M),fill(m^-1,M),fill(m,M))

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
        Cxx⁻¹ = Diagonal(ustrip.([γδ,γT]),[permil^-1,K^-1],[permil,K])
	Cnn⁻¹  = Diagonal([σₙ.^-2],[permil^-1],[permil])
        oproblem = OverdeterminedProblem(y,E,Cnn⁻¹,Cxx⁻¹,x₀)

        # "underdetermined, problem 2.1" 
	Cnn  = Diagonal([σₙ.^2],[permil],[permil^-1])
        Cxx = Diagonal(ustrip.([γδ,γT].^-1),[permil,K],[permil^-1,K^-1])
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
        Cₙₙ = Diagonal(fill(ustrip(σₙ^2),M),fill(g/kg,M),fill(kg/g,M)) 
        Cₙₙ¹² = cholesky(Cₙₙ)
        Cₙₙ⁻¹ = Diagonal(fill(ustrip(σₙ^-2),M),fill(kg/g,M),fill(g/kg,M)) 
        Cₙₙ⁻¹² = cholesky(Cₙₙ⁻¹)

	γ = [1.0e1kg^2/g^2, 1.0e2kg^2*d^2/g^2, 1.0e3kg^2*d^4/g^2, 1.0e4kg^2*d^6/g^2]

	Cxx⁻¹ = Diagonal(ustrip.(γ),[kg/g,kg*d/g,kg*d^2/g,kg*d^3/g],[g/kg,g/kg/d,g/kg/d^2,g/kg/d^3])
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
        Cxx = UnitfulMatrix(SymmetricToeplitz(ρ),fill(cm,n),fill(cm^-1,n),exact=true) + Diagonal(fill(1e-6,n),fill(cm,n),fill(cm^-1,n))

        M = 11
        σₙ = 0.1cm
        Cnn = Diagonal(fill(ustrip(σₙ),M),fill(cm,M),fill(cm^-1,M))

        Enm = sparse(1:M,1:5:n,fill(1.0,M))
        E = UnitfulMatrix(Enm,fill(cm,M),fill(cm,n))

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
        E1 = UnitfulMatrix(randn(M,N),fill(m,M),fill(m,N))
        E2 = UnitfulMatrix(randn(M,N),fill(m,M),fill(m,N))
        E = (one=E1,two=E2)

        Cnn⁻¹1 = Diagonal(fill(σₓ^-1,M),unitrange(E1).^-1,unitrange(E1))

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

    @testset "source water inversions" begin

        # fixed parameters
        @dim YearCE "years Common Era"
        @dim SurfaceRegion "surface location"
        @dim InteriorLocation "interior location"
        @dim StateVariable "state variable"
        yr = u"yr"
        surfaceregions = [:NATL,:ANT,:SUBANT]
        n = length(surfaceregions)
        years = (1990:2000)yr
        statevariables = [:θ, :δ¹⁸O] 
        m = 5 # Interior Locations with obs

        @testset "E matrix = UnitfulDimMatrix (many surface regions, one state variable, multiple obs locations but no obs timeseries, no circulation lag)" begin

            # 3 options
            # 1) 1 state variable (implicit)
            # 2) Obs at one end time
            # 3) No circulation lag
            
            E,x = random_source_water_matrix_vector_pair(m)

            # Run model to predict interior location temperature
            y = E*x # E::UnitfulDimMatrix

            # do slices work?
            E[At(:loc1),At(:NATL)]
            E[:,At(:ANT)]

            # invert for x.  
            x̃ = E\y

            σₙ = 1.0
            Cnndims = (first(dims(E)),first(dims(E)))
            Cnn⁻¹ = Diagonal(fill(σₙ^-1,m),unitrange(E).^-1,unitrange(E),Cnndims,exact=true)

            problem = OverdeterminedProblem(y,E,Cnn⁻¹)
            x̃ = solve(problem,alg=:hessian)
            x̃ = solve(problem,alg=:textbook)

            #@test x ≈ x̃.v
            @test cost(x̃,problem) < 1e-5 # no noise in ob
            #@test within(x̃.v,x,1.0e-5) # deprecate `within`?
            @test isapprox(x̃.v,x,atol=1.0e-5)
        end

        @testset "E:: by impulse response, M:: 2D DimArray (almost complete parameter suite but observations at one location only)" begin
            # many surface regions, with circulation lag, obs:one time or timeseries, one or two state variables" 

            # 3 options
            # 1) 2 state variables
            # 2) timeseries of obs
            # 3) Circulation with lag
            cases = ((false,false,false),(true,false,false),(true,true,true))

            #(statevars,timeseries,lag) = cases[3] # for interactive use
            for (statevars,timeseries,lag) in cases
                println("statevars,timeseries,lag = ",statevars,timeseries,lag)

                lag ? nτ = 5 : nτ = 1
                lags = (0:(nτ-1))yr

                # take all years or just the end year
                timeseries ? yrs = years : yrs = [years[end]]

                M = source_water_matrix_with_lag(surfaceregions,lags)

                # Step 1: get synthetic solution
                if !statevars

                    x = source_water_solution(surfaceregions,yrs)

                    # DimArray is good enough. This is an array, not necessarily a matrix.
                    x₀ = DimArray(zeros(size(x))K,(Ti(yrs),last(dims(M))))

                elseif statevars

                    x = source_water_solution(surfaceregions,years, statevariables)
                    coeffs = UnitfulMatrix(rand(2,1), [NoUnits, K], [K/permil])
                    x₀ = copy(x).*0

                else
                    error("no case")
                end

                # Step 2: get observational operator.
                if timeseries && statevars
                    #Tx = first(dims(x)) # timeseries of observations at these times
                    #global predict(x) = convolve(x,M,Tx,coeffs)
                    global predict(x) = convolve(x,M,first(dims(x)),coeffs)
                elseif statevars
                    global predict(x) = convolve(x,M,coeffs)
                elseif timeseries
                    Tx = first(dims(x)) # timeseries of observations at these times
                    global predict(x) = convolve(x,M,Tx)
                else
                    global predict(x) = convolve(x,M)
                end

                # probe to get E matrix. Use function convolve
                E = impulseresponse(predict,x₀)

                # Given, M and x. Make synthetic data for observations.
                println("Synthetic data")
                y = predict(x)
                println(y)
            
                # test compatibility
                @test first(E*UnitfulMatrix(vec(x₀))) .== first(predict(x₀))
            
                # Julia doesn't have method to make scalar into vector
                !(y isa AbstractVector) && (y = [y])

                # Does E matrix work properly?
                ỹ = E*UnitfulMatrix(vec(x))
                for jj in eachindex(y)
                    @test isapprox(vec(y)[jj],vec(ỹ)[jj])
                end
                x̂ = E\y
                for jj in eachindex(y)
                    @test isapprox(vec(y)[jj],vec(E*x̂)[jj])
                end

                # now in a position to use BLUEs to solve
                σₙ = 0.01
                σₓ = 100.0

                Cnn = Diagonal(fill(σₙ^2,length(y)),unit.(y),unit.(y).^-1)
                Cxx = Diagonal(fill(σₓ^2,length(x₀)),vec(unit.(x₀)),vec(unit.(x₀)).^-1)
                problem = UnderdeterminedProblem(UnitfulMatrix(y),E,Cnn,Cxx,x₀)

                # when x₀ is a DimArray, then x̃ is a DimEstimate
                x̃ = solve(problem) # ::DimEstimate
                @test within(y[1],vec((E*x̃).v)[1],3σₙ) # within 3-sigma
                @test cost(x̃,problem) < 5e-2 # no noise in ob
                @test cost(x̃, problem) == datacost(x̃, problem) + controlcost(x̃, problem)

            end
        end

        @testset "E by impulse response, M:: 3D DimArray (all options turned on, multiple obs locations)" begin
            # many surface regions, with circulation lag, obs:one time or timeseries, obs: one or more locations, one or two state variables"

            coeffs = DimArray([0.2permil/K, 1], (StateVariable(statevariables)))
            cases = ((false,true,false,true),(false,true,true,true), (true, true, true, true))
        
            for (statevars,timeseries,lag,manylocs) in cases

                lag ? nτ = 5 : nτ = 1
                lags = (0:(nτ-1))yr
                manylocs ? m = 6 : m = 1
                interiorlocs = [Symbol("loc"*string(nloc)) for nloc = 1:m]

                # pre-allocate a 3D DimArray
                M = DimArray(zeros(nτ,n,m),(Ti((0:(nτ-1))yr),SurfaceRegion(surfaceregions),InteriorLocation(interiorlocs)))

                # fill it at each location
                for loc in InteriorLocation(interiorlocs)
                    M[:,:,At(Symbol(loc))] = source_water_matrix_with_lag(surfaceregions,lags)
                end

                # Step 1: get synthetic solution
                x = statevars ? source_water_solution(surfaceregions,years, statevariables) : source_water_solution(surfaceregions, years)
                Tx = first(dims(x)) # timeseries of observations at these times, had to shift, not sure how this originally worked ??? 
                x₀ = copy(x).*0
            
                # Step 2: get observational operator.
                global predict(x) = statevars ? sum([coeffs[At(s)] * convolve(x[:, :, At(s)],M,Tx) for s in statevariables]) : convolve(x,M,Tx)
            
                E = impulseresponse(predict,x₀)

                # probe to get E matrix. 
                println("Synthetic data")
                y = predict(x)

                # test compatibility
                @test first(E*UnitfulMatrix(vec(x₀))) .== first(predict(x₀))
            
                # Does E matrix work properly?
                ỹ = E*UnitfulMatrix(vec(x))
                for jj in eachindex(y)
                    @test isapprox(vec(y)[jj],vec(ỹ)[jj])
                end

                # Warning: singular error
                #x̂ = E\vec(y)  # ok to use vec here, this is a check, not a key, repeatable step.
                # issue with previous line: hard to harmonize with other examples
                #for jj in eachindex(y)
                #    @test isapprox(vec(y)[jj],vec(E*x̂)[jj])
                #end

                # now in a position to use BLUEs to solve
                σₙ = 0.01
                σₓ = 10.0
                Cnn = Diagonal(fill(σₙ^2,length(y)),vec(unit.(y)),vec(unit.(y).^-1))
                Cxx = Diagonal(fill(σₓ^2,length(x₀)),vec(unit.(x₀)),vec(unit.(x₀)).^-1)
                problem = UnderdeterminedProblem(UnitfulMatrix(vec(y)),E,Cnn,Cxx,x₀)

                # when x₀ is a DimArray, then x̃ is a DimEstimate
                x̃ = solve(problem)
                @test within(first(y),first(vec((E*x̃).v)),3σₙ) # within 3-sigma
                @test cost(x̃,problem) < 1 # should think more about what a good value would be
                @test cost(x̃, problem) == datacost(x̃, problem) + controlcost(x̃, problem)
            end
        end
    end
end
