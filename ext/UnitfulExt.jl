# make this an extension
module UnitfulExt

using BLUEs
using Unitful
using Measurements
using LinearAlgebra

# translate Vector{Measurement} to Estimate
# let it error rather than restricting types at compile-time
# combine this with function in base BLUEs package
function BLUEs.Estimate(v::AbstractVector{T}) where T <: Quantity{<:Measurement}
        vval = Measurements.value.(v)
        verr = Measurements.uncertainty.(v);
        return Estimate(vval, verr) # just provide standard error
end 

function Base.:\(A::Diagonal{Quantity{Ta,Sa,Va}},
    b::AbstractVector{Quantity{Tb,Sb,Vb}}) where {Ta,Sa,Va,Tb,Sb,Vb}
    uA = unit(first(A))
    return (1/uA)*( ustrip.(A)\ b )
end

response(y::Quantity,y₀,Δu) = ustrip((y - y₀)/Δu)
response(y::AbstractVector{Quantity},y₀,Δu) = vec(ustrip.((y - y₀)/Δu))

end #module
