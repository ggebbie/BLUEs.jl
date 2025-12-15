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

# function expectedunits(y,x)
#     Eunits = Matrix{Unitful.FreeUnits}(undef,length(y),length(x))
#     for ii in eachindex(y)
#         for jj in eachindex(x)
#             if length(y) == 1 && length(x) ==1
#                 Eunits[ii,jj] = unit(y)/unit(x)
#             elseif length(x) == 1
#                 Eunits[ii,jj] = unit.(y)[ii]/unit(x)
#             elseif length(y) ==1
#                 Eunits[ii,jj] = unit(y)/unit.(x)[jj]
#             else
#                 Eunits[ii,jj] = unit.(y)[ii]/unit.(x)[jj]
#             end
#         end
#     end
#     return Eunits
# end

end #module
