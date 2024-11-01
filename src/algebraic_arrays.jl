function Estimate(v::AbstractArray{T}) where T <: Measurement
    vval = VectorArray(Measurements.value.(v))
    verr = VectorArray(Measurements.uncertainty.(v))
    return Estimate(vval, verr) # just provide standard error
end 

