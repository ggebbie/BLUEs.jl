module AA_DD_ULA_Ext

using BLUEs
using AlgebraicArrays
using DimensionalData
using UnitfulLinearAlgebra

"""
function convolve(x::AbstractDimArray, E::AbstractDimArray, coeffs::UnitfulMatrix}
    the `coeffs` argument signifies that x is a 3D array (i.e. >1 state variables)

    this function both convolves, and linearly combines the propagated state variables
"""
# function convolve(x::AbstractDimArray,E::AbstractDimArray, coeffs::UnitfulMatrix)
#     statevars = x.dims[3]
#     mat = UnitfulMatrix(transpose([convolve(x[:,:,At(s)], E) for s in statevars])) * coeffs
#     return getindexqty(mat, 1,1)
# end

#coeffs signifies that x is 3D 
function BLUEs.convolve(x::AbstractDimArray, E::AbstractDimArray, t::Number, coeffs::UnitfulMatrix)
    statevars = x.dims[3]
    mat = UnitfulMatrix(transpose([convolve(x[:,:,At(s)], E, t) for s in statevars]))*coeffs
    return getindexqty(mat, 1,1) 
end

#don't handle the ndims(M) == 3 case here but I'll get back to it
function BLUEs.convolve(x::VectorArray, M::AbstractDimArray, Tx::Union{Ti, Vector}, coeffs::UnitfulMatrix)
    if ndims(M) == 2
        return DimArray([convolve(x,M,Tx[tt], coeffs) for (tt, yy) in enumerate(Tx)], Tx)
    elseif ndims(M) == 3
        error("some code should go here")
    else
        error("M has wrong number of dims") 
    end
end

end
