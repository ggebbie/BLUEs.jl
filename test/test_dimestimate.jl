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
        σₙ = 1.0
        E,x = random_source_water_matrix_vector_pair(m)
        if !use_units
            E = DimArray(parent(E),dims(E))
            x = DimArray(parent(x),dims(x))
        end
        
        # Run model to predict interior location temperature
        y = E*x # E::UnitfulDimMatrix or DimArray

        # do slices work?
        E[At(:loc1),At(:NATL)]
        E[:,At(:ANT)]

        # invert for x.  
        x̃ = E\y # extend backslash (matrix left divide) for DimMatrix

        if use_units
            Cnndims = (first(dims(E)),first(dims(E)))
            Cnn⁻¹ = Diagonal(fill(σₙ^-1,m),unitrange(E).^-1,unitrange(E),Cnndims,exact=true)
        else
            Cnndims = (first(dims(E)),first(dims(E)))
            Cnn⁻¹ = DimArray(Diagonal(fill(σₙ^-1,m)),Cnndims)
        end

        problem = OverdeterminedProblem(y,E,Cnn⁻¹)
        x̃ = solve(problem,alg=:hessian)
        x̃ = solve(problem,alg=:textbook)

        @test cost(x̃,problem) < 1e-5 # no noise in ob
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
