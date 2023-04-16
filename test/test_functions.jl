
"""
    Are two matrices within a certain tolerance?
    Use to simplify tests.
    """
within(A,B,tol) =  maximum(abs.(ustrip.(A - B))) < tol
within(A::UnitfulLinearAlgebra.AbstractUnitfulType,B::UnitfulLinearAlgebra.AbstractUnitfulType,tol) =  maximum(abs.(parent(A - B))) < tol

"""
    function random_source_water_matrix_vector_pair(M)

    M: number of interior locations, observations
    N=3: number of surface regions (fixed), solution or state

    return E,x
"""
function random_source_water_matrix_vector_pair(M)

    #using DimensionalData
    #using DimensionalData: @dim
    #cm = u"cm"
    yr = u"yr"
    K = u"K"
    percent = u"percent"
    surfaceregions = [:NATL,:ANT,:SUBANT]
    interiorlocs = [Symbol("loc"*string(nloc)) for nloc = 1:M]
    N = length(surfaceregions)
    
    # observations have units of temperature
    urange = fill(K,M)
    # solution also has units of temperature
    udomain = fill(K,N)
    Eparent = rand(M,N)#*100percent

    # normalize to conserve mass
    for nrow = 1:size(Eparent,1)
        Eparent[nrow,:] ./= sum(Eparent[nrow,:])
    end
    
    E = UnitfulDimMatrix(ustrip.(Eparent),urange,udomain,dims=(InteriorLocation(interiorlocs),SurfaceRegion(surfaceregions)))

    x = UnitfulDimMatrix(randn(N),fill(K,N),dims=(SurfaceRegion(surfaceregions)))
    return E,x
end

"""
    function random_source_water_matrix_vector_pair_with_lag(M)

    M: number of interior locations, observations
    N=3: number of surface regions (fixed), solution or state

    return E,x
"""
function source_water_matrix_vector_pair_with_lag(M)

    #using DimensionalData
    #using DimensionalData: @dim
    #cm = u"cm"
    yr = u"yr"
    K = u"K"
    #percent = u"percent"
    surfaceregions = [:NATL,:ANT,:SUBANT]
    lags = (0:(M-1))yr
    years = (1990:2000)yr

    Myears = length(years)
    N = length(surfaceregions)
    
    # observations have units of temperature
    urange1 = fill(K,M)
    urange2 = fill(K,Myears)
    # solution also has units of temperature
    udomain = fill(K,N)
    Eparent = rand(M,N)#*100percent

    # normalize to conserve mass
    for nrow = 1:size(Eparent,1)
        Eparent /= sum(Eparent)
    end
    
    E = UnitfulDimMatrix(ustrip.(Eparent),urange1,udomain,dims=(Ti(lags),SurfaceRegion(surfaceregions)))
    x = UnitfulDimMatrix(randn(Myears,N),urange2,fill(unit(1.0),N),dims=(Ti(years),SurfaceRegion(surfaceregions)))
    return E,x
end

function source_water_matrix_with_lag(surfaceregions,lags,years)
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
    x = DimArray(randn(m,n)K,(Ti(years),SurfaceRegion(surfaceregions)))
    return x
end

function source_water_solution(surfaceregions, years, statevar)
    yr = u"yr"
    K = u"K"
    permil = u"permille"
    m = length(years)
    n = length(surfaceregions)
    mat = cat(randn(m, n, 1)K, randn(m, n, 1)permil; dims = 3)
    x = DimArray(mat, (Ti(years), SurfaceRegion(surfaceregions), StateVariable(statevar)))
    return x
end

"""
    function source_water_matrix_vector_pair_with_lag(M)

    M: number of interior locations, observations
    N=3: number of surface regions (fixed), solution or state

    return E,x
"""
function source_water_DimArray_vector_pair_with_lag(M)

    #using DimensionalData
    #using DimensionalData: @dim
    #cm = u"cm"
    yr = u"yr"
    K = u"K"
    #percent = u"percent"
    surfaceregions = [:NATL,:ANT,:SUBANT]
    lags = (0:(M-1))yr
    years = (1990:2000)yr

    Myears = length(years)
    N = length(surfaceregions)
    
    Eparent = rand(M,N)#*100percent
    # normalize to conserve mass
    for nrow = 1:size(Eparent,1)
        Eparent /= sum(Eparent)
    end
    
    M = DimArray(Eparent,(Ti(lags),SurfaceRegion(surfaceregions)))
    x = DimArray(randn(Myears,N)K,(Ti(years),SurfaceRegion(surfaceregions)))
    return M,x
end
