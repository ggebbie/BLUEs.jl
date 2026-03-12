module DynamicQuantitiesExt

using BLUEs
using DynamicQuantities

using Measurements
using LinearAlgebra

function BLUEs.Estimate(v::AbstractVector{T}) where T <: Quantity{<: Measurement}
        vval = Measurements.value.(v)
        verr = Measurements.uncertainty.(v);
        return Estimate(vval, verr) # just provide standard error
end 

function LinearAlgebra.:(\ )(A::Diagonal{Q, QA},
    B::Diagonal{Q, QA}) where Q <: Quantity where QA <: QuantityArray
    return QuantityArray( ustrip.(A) \ ustrip.(B),
        DynamicQuantities.dimension(B)/DynamicQuantities.dimension(A))
end
function LinearAlgebra.:inv(A::Diagonal{Q, QA}) where Q <: Quantity where QA <: QuantityArray
    return QuantityArray( inv(ustrip.(A)), inv(DynamicQuantities.dimension(A)))
end
function Base.:+(A::Diagonal{Q, QA}, B::Diagonal{Q, QA}) where Q <: Quantity where QA <: QuantityArray
    if dimension(A) == dimension(B)
        return QuantityArray( A.value + B.value, dimension(A))
    else
        error("can't add with different units")
    end
end

end
