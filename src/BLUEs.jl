module BLUEs

using LinearAlgebra, Statistics, Unitful, UnitfulLinearAlgebra, Measurements

export solve, show

import Base: show, getproperty, propertynames

#struct Measurement{T<:AbstractFloat} <: AbstractFloat

struct Estimate{Tv <: Number,TC <: Number} 
    v :: AbstractVector{Tv}
    C :: AbstractMatrix{TC}
end

function show(io::IO, mime::MIME{Symbol("text/plain")}, x::Estimate{<:Number})
    summary(io, x); println(io)
    println(io, "Estimate and 1σ uncertainty")
    show(io, mime, x.x)
    # println(io, "\nsingular values:")
    # show(io, mime, F.S)
    # println(io, "\nV (right singular vectors):")
    # show(io, mime, F.V)
end

"""
    function getproperty(x::Estimate, d::Symbol)

# Fields of Estimate
- `x::Vector{Measurement}`: Estimate and 1σ uncertainty
- `σ::Vector{Number}`: 1σ uncertainty
- `v::Vector{Number}`: central value of estimate
- `C::Matrix{Number}`: estimate uncertainty matrix
"""
function getproperty(x::Estimate, d::Symbol)
    if d === :σ
        return .√diag(x.C)
    elseif d === :x
        return measurement.(x.v,x.σ)
        #return x.v .± x.σ
    else
        return getfield(x, d)
    end
end

propertynames(x::Estimate) = (:x, :σ, fieldnames(typeof(x))...)

# Code to make some property names private
#Base.propertynames(x::Estimate, private::Bool=false) =
#    private ? (:U, :U⁻¹, :V, :V⁻¹,  fieldnames(typeof(F))...) : (:U, :U⁻¹, :S, :V, :V⁻¹)

function solve(y::AbstractVector,E::AbstractMatrix,Cnn⁻¹::AbstractMatrix)
    CE = Cnn⁻¹*E
    ECE = transpose(E)*CE
    return Estimate( ECE \ (transpose(CE)*y), inv(ECE))
end

end
