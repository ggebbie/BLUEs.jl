using DimensionalData
using DimensionalData:@dim

# set necessary functions first (may get overwritten and complain)
function source_water_matrix_with_lag(surfaceregions,lags)
    yr = u"yr"
    K = u"K"
    Mparent = rand(length(lags),length(surfaceregions))#*100percent
    # normalize to conserve mass
    for nrow = 1:size(Mparent,1)
        Mparent /= sum(Mparent)
    end
    return M = DimArray(Mparent,(Ti(lags),SurfaceRegion(surfaceregions)))
end

function source_water_solution(surfaceregions,years)
    yr = u"yr"
    K = u"K"
    m = length(years)
    n = length(surfaceregions)
    x = VectorArray(DimArray(randn(m,n)K,(Ti(years),SurfaceRegion(surfaceregions))))
    return x
end

# function source_water_solution(surfaceregions, years, statevar)
#     yr = u"yr"
#     K = u"K"
#     permil = u"permille"
#     m = length(years)
#     n = length(surfaceregions)
#     mat = cat(randn(m, n, 1)K, randn(m, n, 1)permil; dims = 3)
#     x = VectorArray(DimArray(mat, (Ti(years), SurfaceRegion(surfaceregions), StateVariable(statevar))))
#     return x
# end

@testset "dimensional data + unitful" begin

    MatrixDimArray = MatrixArray{T, M, N, R} where {M, T, N, R<:AbstractDimArray{T, M}}
    VectorDimArray = VectorArray{T, N, A} where {T, N, A <: DimensionalData.AbstractDimArray}

    # if use_units
    #     @testset "objective mapping with DimensionalData and AlgebraicArrays" begin
    #         include("test_objective_mapping_AlgebraicArrays.jl")
    #     end
    # end
    
    
    @testset "uniform state vectors" begin

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

        (statevars,timeseries,lag) = cases[1] # for interactive use
        
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
end

