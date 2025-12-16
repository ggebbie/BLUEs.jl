module UnitfulAlgebraicArraysExt

using BLUEs
using Unitful
using Measurements
using AlgebraicArrays

function BLUEs.Estimate(v::AbstractArray{Q}) where {T <: Measurement, Q <: Quantity{T}}
    vval = VectorArray(Measurements.value.(v))
    verr = VectorArray(Measurements.uncertainty.(v))
    return Estimate(vval, verr) # just provide standard error
end 

end 
