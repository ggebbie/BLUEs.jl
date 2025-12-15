using DimensionalData
using DimensionalData:@dim

# set necessary functions first (may get overwritten and complain)
function source_water_matrix_with_lag(surfaceregions,lags)
    Mparent = rand(length(lags),length(surfaceregions))#*100percent
    # normalize to conserve mass
    for nrow = 1:size(Mparent,1)
        Mparent /= sum(Mparent)
    end
    return M = DimArray(Mparent,(Ti(lags),SurfaceRegion(surfaceregions)))
end

function source_water_solution(surfaceregions,years)
    m = length(years)
    n = length(surfaceregions)
    x = VectorArray(DimArray(randn(m,n),(Ti(years),SurfaceRegion(surfaceregions))))
    return x
end

function source_water_solution(surfaceregions, years, statevar)
    m = length(years)
    n = length(surfaceregions)
    mat = cat(randn(m, n, 1), randn(m, n, 1); dims = 3)
    x = VectorArray(DimArray(mat, (Ti(years), SurfaceRegion(surfaceregions), StateVariable(statevar))))
    return x
end

@testset "dimensional data" begin

    MatrixDimArray = MatrixArray{T, M, N, R} where {M, T, N, R<:AbstractDimArray{T, M}}
    VectorDimArray = VectorArray{T, N, A} where {T, N, A <: DimensionalData.AbstractDimArray}

    @testset "uniform state vectors" begin

        # fixed parameters
        @dim YearCE "years Common Era"
        @dim SurfaceRegion "surface location"
        @dim InteriorLocation "interior location"
        @dim StateVariable "state variable"
        surfaceregions = [:NATL,:ANT,:SUBANT]
        N = length(surfaceregions)
        years = (1990:2000)
        statevariables = [:θ, :δ¹⁸O] 
        M = 5 # Interior Locations with obs

        # 3 options
        # 1) 2 state variables
        # 2) timeseries of obs
        # 3) Circulation with lag

        # define (statevars,timeseries,lag)
        cases = ((false,false,false),(true,false,false),(true,true,true))

        # for interactive use, 1 test fails if all cases are run, is this intended?
        (statevars,timeseries,lag) = cases[1]
        println("statevars,timeseries,lag = ",statevars, " ", timeseries, " ", lag)

        #define constants
        lag ? nτ = 5 : nτ = 1
        lags = (0:(nτ-1))
        σn = 0.01
        σₓ = 10.0

        # take all years or just the end year
        timeseries ? yrs = years : yrs = [years[end]]
        M = source_water_matrix_with_lag(surfaceregions,lags)

        # Step 1: get synthetic solution
        x = source_water_solution(surfaceregions,yrs)
        d = VectorArray(fill(σₓ,dims(x)))
        x₀ = 0.0 * x #zeros(dims(x),:VectorArray) .* unit.(x)
        x0 = Estimate( x₀, d)
                    
        global observe(x) = convolve(x,M)

        # Given, M and x. Make synthetic data for observations.
        println("Synthetic data")
        ytrue = observe(x)
        ny = length(ytrue)
        d  = VectorArray(fill(σn, rangedims(ytrue)))
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
    end

    # consider adding following experiments
    #     @testset "E by impulse response, M:: 3D DimArray (all options turned on, multiple obs locations)" begin
    @testset "state vectors but no units" begin

        # fixed parameters
        @dim YearCE "years Common Era"
        @dim SurfaceRegion "surface location"
        @dim InteriorLocation "interior location"
        @dim StateVariable "state variable"
        surfaceregions = [:NATL,:ANT,:SUBANT]
        N = length(surfaceregions)
        years = (1990:2000)
        statevariables = [:θ, :δ¹⁸O] 
        M = 5 # Interior Locations with obs

        # 3 options
        # 1) 2 state variables
        # 2) timeseries of obs
        # 3) Circulation with lag

        # define (statevars,timeseries,lag)
        cases = ((true,false,false),(true,true,true))

        #        (statevars,timeseries,lag) = cases[1] # for interactive use
        for (statevars,timeseries,lag) in cases
            println("statevars,timeseries,lag = ",statevars, " ", timeseries, " ", lag)

            #define constants
            lag ? nτ = 5 : nτ = 1
            lags = (0:(nτ-1))
            σn = 0.01
            σₓ = 10.0

            # take all years or just the end year
            timeseries ? yrs = years : yrs = [years[end]]

            M = source_water_matrix_with_lag(surfaceregions,lags)

            # Step 1: get synthetic solution
            x = source_water_solution(surfaceregions, yrs, statevariables)
            coeffs = DimArray(rand(2), StateVariable(statevariables))
            d = VectorArray(fill(σₓ,dims(x)))
            x₀ = 0.0 * x
            x0 = Estimate(x₀,d)

            # Step 2: get observational operator.
            if timeseries && statevars
                Tx = first(dims(x)) # timeseries of observations at these times
                global observe(x) = convolve(x,M,Tx,coeffs)
            elseif statevars
                global observe(x) = convolve(x,M,coeffs)
            else
                global observe(x) = convolve(x,M)
            end

            # Given, M and x. Make synthetic data for observations.
            println("Synthetic data")
            ytrue = observe(x)
            ny = length(ytrue)
            d  = VectorArray(fill(σn, rangedims(ytrue)))
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
        end
    end
end 
