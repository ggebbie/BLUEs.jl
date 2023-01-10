module BLUEs

using LinearAlgebra, Statistics, Unitful, UnitfulLinearAlgebra, Measurements

export Estimate, OverdeterminedProblem, UnderdeterminedProblem
export solve, show, cost, datacost, controlcost

import Base: show, getproperty, propertynames, (*), (+)

import LinearAlgebra: pinv

#struct Measurement{T<:AbstractFloat} <: AbstractFloat

struct Estimate{Tv <: Number,TC <: Number} 
    v :: AbstractVector{Tv}
    C :: AbstractMatrix{TC}
end

struct OverdeterminedProblem
    y :: Union{<: AbstractVector,NamedTuple}
    E :: Union{<: AbstractMatrix, NamedTuple}
    Cnn⁻¹ :: Union{<: AbstractMatrix, NamedTuple}
    Cxx⁻¹ :: Union{<: AbstractMatrix, Missing}
    x₀ :: Union{<: AbstractVector, Missing}
end

# constructor for case without prior information
OverdeterminedProblem(y::Union{<: AbstractVector,NamedTuple},E::Union{<: AbstractMatrix,NamedTuple},Cnn⁻¹::Union{<:Matrix,NamedTuple}) = OverdeterminedProblem(y,E,Cnn⁻¹,missing,missing)

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

    optional alg= :textbook or :hessian
"""
function solve(op::OverdeterminedProblem; alg=:textbook)
    if alg == :textbook
        solve_textbook(op)
    else
        alg == :hessian
        solve_hessian(op)
    end
end
function solve_textbook(op::OverdeterminedProblem)
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
function solve_hessian(op::OverdeterminedProblem)
    if ismissing(op.x₀)
        n = op.y 
    else
        n = op.y - op.E*op.x₀
    end
    #∂J∂n = (op.Cnn⁻¹*n)
    ∂J∂n = 2 *(op.Cnn⁻¹*n) # issue with "2"
    ∂J∂x = -(transpose(op.E)*∂J∂n) # annoying

    H⁻¹ = inv(hessian(op))
    x = -(1//2)*H⁻¹*∂J∂x
    (~ismissing(op.x₀)) && (x += op.x₀)
    return Estimate( x, H⁻¹)
end

#parse error
#hessian(op::OverdeterminedProblem) = ismissing(op.Cxx⁻¹) ? return transpose(op.E)*(op.Cnn⁻¹*op.E) : return transpose(op.E)*(op.Cnn⁻¹*op.E) + op.Cxx⁻¹
function hessian(op::OverdeterminedProblem)
    if ismissing(op.Cxx⁻¹)
        return transpose(op.E)*(op.Cnn⁻¹*op.E)
    else
        return transpose(op.E)*(op.Cnn⁻¹*op.E) + op.Cxx⁻¹
    end
end

"""
    function pinv

    Left pseudo-inverse (i.e., least-squares estimator)
"""
function pinv(op::OverdeterminedProblem)
    CE = op.Cnn⁻¹*op.E
    ECE = transpose(op.E)*CE
    return ECE \ transpose(CE)
end

"""
    Solve underdetermined problem
"""
function solve(up::UnderdeterminedProblem)

    if ismissing(up.x₀)
        n = up.y
    else
        n = up.y - up.E*up.x₀
    end
    Cxy = up.Cxx*transpose(up.E)
    Cyy = up.E*Cxy + up.Cnn
    v = Cxy*(Cyy \ n)
    (~ismissing(up.x₀)) && (v += up.x₀)
    P = up.Cxx - Cxy*(Cyy\(transpose(Cxy)))
    return Estimate(v,P)
end

"""    
    Matrix multiplication for Estimate includes
    error propagation.
"""
*(F::AbstractMatrix,x::Estimate) = Estimate(F*x.v,F*x.C*transpose(F))

"""    
    Matrix addition for Estimate includes
    error propagation. Follow pp. 95, Sec. 2.5.5,
    Recursive Least Squares, "Dynamical Insights from Data" class notes
"""
+(x::Estimate,y::Estimate) = error("not implemented yet")

"""
    Compute cost function
"""
function cost(x̃::Estimate,op::OverdeterminedProblem)
    Jdata = datacost(x̃,op)
    (~ismissing(op.x₀) && ~ismissing(op.Cxx⁻¹)) ? Jcontrol = controlcost(x̃,op) : Jcontrol = nothing
    isnothing(Jcontrol) ? J = Jdata : J = Jdata + Jcontrol
    return J
end
function cost(x̃::Estimate,up::UnderdeterminedProblem)
    Jdata = datacost(x̃,up)
    Jcontrol = controlcost(x̃,up) 
    J = Jdata + Jcontrol
    return J
end

"""
    Cost function contribution from observations
"""
function datacost( x̃::Estimate, p::Union{OverdeterminedProblem,UnderdeterminedProblem})
    n = p.y - p.E*x̃.v
    if typeof(p) == UnderdeterminedProblem
        Cnn⁻¹ = inv(p.Cnn)
    elseif typeof(p) == OverdeterminedProblem
        Cnn⁻¹ = p.Cnn⁻¹
    end
    return transpose(n)*(Cnn⁻¹*n)
end

"""
    Cost function contribution from control vector
"""
function controlcost( x̃::Estimate, p::Union{OverdeterminedProblem,UnderdeterminedProblem})
    # not implemented yet
    Δx = x̃.v - p.x₀
    if typeof(p) == UnderdeterminedProblem
        Cxx⁻¹ = inv(p.Cxx)
    elseif typeof(p) == OverdeterminedProblem
        Cxx⁻¹ = p.Cxx⁻¹
    end
    return transpose(Δx)*(Cxx⁻¹*Δx)
end

#Matrix multiply, Mx
"""
    mul
"""
function *(A::NamedTuple, b::Vector) 
    # Update to use parametric type to set type of Vector
    c = Vector(undef, length(A))
    for (i, V) in enumerate(A)
        c[i] = V*b 
    end
    return NamedTuple{keys(A)}(c)
end
function *(A::NamedTuple, b::NamedTuple)
    ~(keys(A) == keys(b)) && error("named tuples don't have consistent fields")
    
    # Update to use parametric type to set type of Vector
    c = Vector(undef, length(A))
    for (i, V) in enumerate(A)
        c[i] = V*b[i] # b index safe?
    end
    return NamedTuple{keys(A)}(c)
end

end # module
