module AlgebraicArraysExt

using BLUEs
using Measurements
using AlgebraicArrays

function BLUEs.Estimate(v::AbstractArray{T}) where T <: Measurement
    # assume it is a vector?
    vval = AlgebraicArray(Measurements.value.(v),(size(v),))
    verr = AlgebraicArray(Measurements.uncertainty.(v), (size(v),))
    return Estimate(vval, verr) # just provide standard error
end

end 
