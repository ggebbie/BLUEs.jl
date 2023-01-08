module BLUEs

using LinearAlgebra, Statistics, Unitful, UnitfulLinearAlgebra, Measurements

export Estimate, OverdeterminedProblem
export solve, show, cost

import Base: show, getproperty, propertynames, (*)

import LinearAlgebra: pinv

#struct Measurement{T<:AbstractFloat} <: AbstractFloat

struct Estimate{Tv <: Number,TC <: Number} 
    v :: AbstractVector{Tv}
    C :: AbstractMatrix{TC}
end

struct OverdeterminedProblem
    y :: AbstractVector
    E :: AbstractMatrix
    Cnn⁻¹ :: AbstractMatrix
    Cxx⁻¹ :: Union{AbstractMatrix,Missing}
    x₀ :: Union{AbstractVector,Missing}
end

# constructor for case without prior information
OverdeterminedProblem(y::AbstractVector,E::AbstractMatrix,Cnn⁻¹::AbstractMatrix) = OverdeterminedProblem(y,E,Cnn⁻¹,missing,missing)

struct UnderdeterminedProblem
    y :: AbstractVector
    E :: AbstractMatrix
    Cnn :: AbstractMatrix
    Cxx :: Union{AbstractMatrix}
    x₀ :: Union{AbstractVector,Missing}
end

# constructor for case without prior information
UnderdeterminedProblem(y::AbstractVector,E::AbstractMatrix,Cnn::AbstractMatrix,Cxx::AbstractMatrix) = OverdeterminedProblem(y,E,Cnn,Cxx,missing)

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

"""
    Solve overdetermined problem
"""
function solve(op::OverdeterminedProblem)
    CE = op.Cnn⁻¹*op.E
    ECE = transpose(op.E)*CE
    if ismissing(op.Cxx⁻¹)  
        return Estimate( ECE \ (transpose(CE)*op.y), inv(ECE))
    else
        # prior information available
        ECE += op.Cxx⁻¹
        rhs = transpose(CE)*op.y
        (~ismissing(op.x₀)) && (rhs += op.Cxx⁻¹*op.x₀)
        return Estimate( ECE \ rhs, inv(ECE))
    end
end

"""
    function pinv

    Left pseudo-inverse
"""
function pinv(op::OverdeterminedProblem)
    CE = op.Cnn⁻¹*op.E
    ECE = transpose(op.E)*CE
    return ECE \ transpose(CE)
end

"""    
    Matrix multiplication for Estimate includes
    error propagation.
"""
*(F::AbstractMatrix,x::Estimate) = Estimate(F*x.v,F*x.C*transpose(F))

"""
    Compute cost function
"""
function cost(x̃::Estimate,op::OverdeterminedProblem)

    Jdata = datacost(x̃,op)

    (~ismissing(op.x₀) && ~ismissing(op.Cxx⁻¹)) ? Jcontrol = controlcost(x̃,op) : Jcontrol = nothing

    isnothing(Jcontrol) ? J = Jdata : J = Jdata + Jcontrol

    return J
end

"""
    Cost function contribution from observations
"""
function datacost( x̃::Estimate, op::OverdeterminedProblem)
    n = op.y - op.E*x̃.v
    return transpose(n)*(op.Cnn⁻¹*n)
end

"""
    Cost function contribution from control vector
"""
function controlcost( x̃::Estimate, op::OverdeterminedProblem)
    # not implemented yet
    return nothing
end

end # module
