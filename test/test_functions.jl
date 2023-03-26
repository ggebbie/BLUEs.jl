
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


# function predictobs(u,x₀,M)
#     x = addcontrol(x₀,u) 
#     return y = convolve(M,x)
# end


# function expectedunits(y,x₀)
#     Eunits = Matrix{Unitful.FreeUnits}(undef,length(y),length(x₀))
#     for ii in eachindex(y)
#         for jj in eachindex(x₀)
#             if length(y) == 1 && length(x) ==1
#                 Eunits[ii,jj] = unit(y)/unit(x₀)
#             elseif length(x) == 1
#                 Eunits[ii,jj] = unit.(y)[ii]/unit(x₀)
#             elseif length(y) ==1
#                 Eunits[ii,jj] = unit(y)/unit.(x₀)[jj]
#             else
#                 Eunits[ii,jj] = unit.(y)[ii]/unit.(x₀)[jj]
#             end
#         end
#     end
#     return Eunits
# end

# function impulseresponse(x₀,M)
#     Eunits = expectedunits(y,x₀)
#     Eu = zeros(1,length(x₀)).*Eunits
#     u = zeros(length(x₀)).*unit.(x₀)[:]
#     y₀ = predictobs(u,M,x₀)
#     for rr in eachindex(x₀)
#         u = zeros(length(x₀)).*unit.(x₀)[:]
#         Δu = 1*unit.(x₀)[rr]
#         u[rr] += Δu
#         y = predictobs(u,M,x₀)
#         Eu[rr] = (y - y₀)/Δu
#     end
#     return E = UnitfulMatrix(Eu)
# end
