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
#    x = DimArray(randn(N).*fill(K,N),(SurfaceRegion(surfaceregions)))
    return E,x
end

"""
    function random_source_water_matrix(M)

    M: number of interior locations, observations

    return E
"""
function random_source_water_matrix(M)

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
    
    E = UnitfulDimMatrix(ustrip.(Eparent),urange,udomain,dims=(InteriorLocation(interiorlocs), SurfaceRegion(surfaceregions)))

    return E
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

function synthetic_timeseries(obs, t_obs, σ_obs, t_om, ρ, u)
    n = length(obs)
    m = length(t_om)
    Cnn = Diagonal(fill(ustrip(σ_obs), n), fill(u, n), fill(u^-1, n), exact = true)
    Cxx = UnitfulMatrix(SymmetricToeplitz(ρ), fill(u, m), fill(u^-1, m), exact = true) + Diagonal(fill(1e-6, m), fill(u, m), fill(u^-1, m), exact = true)
    Enm = sparse(1:n, findall(x->x∈t_obs, t_om), fill(1.0, n))
    E = UnitfulMatrix(Enm, fill(u, n), fill(u, m), exact = true)
    x₀ = UnitfulMatrix(ones(m) .* mean(obs))
    x̃ = solve(UnderdeterminedProblem(obs, E, Cnn, Cxx, x₀))
    return x̃           
end

