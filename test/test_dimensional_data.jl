@testset "dimensional data" begin

    MatrixDimArray = MatrixArray{T, M, N, R} where {M, T, N, R<:AbstractDimArray{T, M}}
    VectorDimArray = VectorArray{T, N, A} where {T, N, A <: DimensionalData.AbstractDimArray}

    @testset "state vectors" begin

        # fixed parameters
        @dim YearCE "years Common Era"
        @dim SurfaceRegion "surface location"
        @dim InteriorLocation "interior location"
        @dim StateVariable "state variable"
        yr = u"yr"
        surfaceregions = [:NATL,:ANT,:SUBANT]
        N = length(surfaceregions)
        years = (1990:2000)yr
        statevariables = [:θ, :δ¹⁸O] 
        M = 5 # Interior Locations with obs

        # 3 options
        # 1) 2 state variables
        # 2) timeseries of obs
        # 3) Circulation with lag

        # define (statevars,timeseries,lag)
        cases = ((false,false,false),(true,false,false),(true,true,true))

        #        (statevars,timeseries,lag) = cases[1] # for interactive use
        for (statevars,timeseries,lag) in cases
            println("statevars,timeseries,lag = ",statevars, " ", timeseries, " ", lag)

            #define constants
            lag ? nτ = 5 : nτ = 1
            lags = (0:(nτ-1))yr
            σn = 0.01K
            σₓ = 10.0K

            # take all years or just the end year
            timeseries ? yrs = years : yrs = [years[end]]

            M = source_water_matrix_with_lag(surfaceregions,lags)

            # Step 1: get synthetic solution
            if !statevars
                x = source_water_solution(surfaceregions,yrs)

                if use_units
                    #d  = AlgebraicArray(fill(ustrip.(σₓ)^2,size(x)),dims(x)) .*(unit.(x)).^2
                    #d  = AlgebraicArray(fill(σₓ.^2,size(x)),dims(x)) .*(unit.(x)).^2
                    d = VectorArray(fill(σₓ,dims(x)))
                    #Diagonal(VectorArray(fill(σₓ.^2,dims(x))))
                    #d = VectorArray(fill(σₓ.^2,dims(x)))
                else
                    x = ustrip.(x) # extra step: remove units
                    d = VectorArray(fill(ustrip.(σₓ),dims(x)))
                    #x₀ = 0.0 * x #zeros(dims(x),:VectorArray)
                    #d  = AlgebraicArray(fill(ustrip.(σₓ)^2,size(x)),dims(x))
                end
                x₀ = 0.0 * x #zeros(dims(x),:VectorArray) .* unit.(x)
                #Px0 = Diagonal(d)
                x0 = Estimate( x₀, d)
                    
            elseif statevars

                x = source_water_solution(surfaceregions, yrs, statevariables)
                coeffs = DimArray(rand(2).*[permil/K,NoUnits], StateVariable(statevariables))
                if use_units

                    # unit.(x) doesn't work for non-uniform matrices
                    d = VectorArray(fill(ustrip.(σₓ),dims(x))) .* unit.(x) # when multiple units

                else
                    x = ustrip.(x) # extra step: remove units
                    coeffs = ustrip.(coeffs)
                    d = VectorArray(fill(ustrip.(σₓ),dims(x)))
                end
                x₀ = 0.0 * x
                x0 = Estimate(x₀,d)
            else
                error("no case")
            end

            # Step 2: get observational operator.
            if timeseries && statevars
                Tx = first(dims(x)) # timeseries of observations at these times
                global observe(x) = convolve(x,M,Tx,coeffs)
            elseif statevars
                global observe(x) = convolve(x,M,coeffs)
                # elseif timeseries
                #     Tx = first(dims(x)) # timeseries of observations at these times
                #     global observe(x) = convolve(x,M,Tx)
            else
                global observe(x) = convolve(x,M)
            end

            # Given, M and x. Make synthetic data for observations.
            println("Synthetic data")
            ytrue = observe(x)
            ny = length(ytrue)
            if use_units
                #Py  = VectorArray(fill(ustrip.(σn)^2,length(ytrue)), rangedims(ytrue)).*unit.(ytrue)
                d  = VectorArray(fill(σn, rangedims(ytrue)))
            else
                #Py  = VectorArray(fill(ustrip.(σn)^2,length(ytrue)), rangedims(ytrue))
                d  = VectorArray(fill(ustrip.(σn), rangedims(ytrue)))
            end
            y = Estimate( ytrue, d)

            Px0 = x0.P
            # test pieces of combine
            @test observe(Px0) isa MatrixArray
            @test observe(Px0) isa MatrixDimArray 
            @test parent(observe(Px0)) isa DimArray # workaround
            
            x1 = combine(x0,y,observe)

            # check whether obs are reproduced
            ytilde = observe(x1.v)
            @test isapprox(y.v,ytilde,rtol= 1e-2) 

            # if use_units
            #     @test isapprox(ustrip.(y.v),ustrip.(ytilde), atol=1e-3)  
            # else
            #     @test isapprox(y.v,ytilde, atol=1e-3) 
            # end
            
            # # check functional form of observational operator
            # Imatrix = DiagonalDimArray(ones(length(x)), dims(x))
            # Ematrix = observe(Imatrix)
            # # uses Core.Matrix instead of MultipliableDimArrays, don't know why            
            # Etest = MultipliableDimArrays.Matrix(Ematrix)

            # # or solve the impulse response in one step
            # E = MultipliableDimArrays.Matrix(observe(Imatrix)) # not tested if E elements have units

            # unsolved issues with least-squares methods
            # problem = UnderdeterminedProblem(y.v,
            #     E,
            #     BLUEs.algebraic_object(y.P),
            #     BLUEs.algebraic_object(x0.P),
            #     x0.v)

            # # when x₀ is a DimArray, then x̃ is a DimEstimate
            # x̃ = solve(problem) # ::DimEstimate
            # @test within(y[1],vec((E*x̃).v)[1],3σn) # within 3-sigma
            # @test cost(x̃,problem) < 5e-2 # no noise in ob
            # @test cost(x̃, problem) == datacost(x̃, problem) + controlcost(x̃, problem)

        end
    end

    # consider adding following experiments
    #     @testset "E by impulse response, M:: 3D DimArray (all options turned on, multiple obs locations)" begin
    
    # @testset "issue 41" begin
end
