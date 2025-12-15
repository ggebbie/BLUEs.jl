module AlgebraicArraysExt

using BLUEs
using Unitful
using Measurements
using AlgebraicArrays

function BLUEs.Estimate(v::AbstractArray{T}) where T <: Measurement
    vval = VectorArray(Measurements.value.(v))
    verr = VectorArray(Measurements.uncertainty.(v))
    return Estimate(vval, verr) # just provide standard error
end

# duplicated code from algebraic_arrays, should only be run with Unitful + AlgebraicArrays 
function BLUEs.Estimate(v::AbstractArray{Q}) where {T <: Measurement, Q <: Quantity{T}}
    vval = VectorArray(Measurements.value.(v))
    verr = VectorArray(Measurements.uncertainty.(v))
    return Estimate(vval, verr) # just provide standard error
end 

end 
