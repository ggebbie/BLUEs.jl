module BLUEs

using LinearAlgebra, Statistics, Unitful, UnitfulLinearAlgebra, Measurements

export Estimate, OverdeterminedProblem, UnderdeterminedProblem
export solve, show, cost, datacost, controlcost

import Base: show, getproperty, propertynames, (*), (+), (-)

import LinearAlgebra: pinv, transpose

#struct Measurement{T<:AbstractFloat} <: AbstractFloat

"""
    struct Estimate
    
    a structure with some vector of values v and associated covariance matrix C 
"""
struct Estimate{Tv <: Number,TC <: Number} 
    v :: AbstractVector{Tv}
    C :: AbstractMatrix{TC}
end

"""
    struct OverdeterminedProblem

    a structure (NamedTuple version) with fields

    - `y::Union{<: AbstractVector,NamedTuple}`: "observations", namedtuples of vectors
    - `E :: Union{<: AbstractMatrix, NamedTuple}`: model matrices 
    - `Cnn⁻¹ :: Union{<: AbstractMatrix, NamedTuple}`: namedtuple of inverse noise covariance matrix
    - `Cxx⁻¹ :: Union{<: AbstractMatrix, Missing}`: tapering matrices, NOT namedtuples 
    - `x₀ :: Union{<: AbstractVector, Missing}`: first guess vector
"""
struct OverdeterminedProblem
    y :: Union{<: AbstractVector,NamedTuple}
    E :: Union{<: AbstractMatrix, NamedTuple}
    Cnn⁻¹ :: Union{<: AbstractMatrix, NamedTuple}
    Cxx⁻¹ :: Union{<: AbstractMatrix, Missing}
    x₀ :: Union{<: AbstractVector, Missing}
end

"""
    function OverdeterminedProblem

    generates OverdeterminedProblem structure with x₀ = missing, Cxx = missing 
"""
OverdeterminedProblem(y::Union{<: AbstractVector,NamedTuple},E::Union{<: AbstractMatrix,NamedTuple},Cnn⁻¹::Union{<:AbstractMatrix,NamedTuple}) = OverdeterminedProblem(y,E,Cnn⁻¹,missing,missing)


"""
    struct UnderdeterminedProblem

    a structure with fields

    - `y::AbstractVector`: vector of "observations"
    - `E::AbstractMatrix`: model matrix 
    - `Cnn::AbstractMatrix`: noise covariance matrix 
    - `Cxx::Union{AbstractMatrix}`: tapering matrix 
    - `x₀::Union{AbstractVector, Missing}`: first guess vector
"""
struct UnderdeterminedProblem
    y :: AbstractVector
    E :: AbstractMatrix
    Cnn :: AbstractMatrix
    Cxx :: Union{AbstractMatrix}
    x₀ :: Union{AbstractVector,Missing}
end


"""
    function UnderdeterminedProblem

    generates UnderdeterminedProblem structure with x₀ = missing, still requires Cxx 
"""
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

# Template code to make some property names private
#Base.propertynames(x::Estimate, private::Bool=false) =
#    private ? (:U, :U⁻¹, :V, :V⁻¹,  fieldnames(typeof(F))...) : (:U, :U⁻¹, :S, :V, :V⁻¹)

"""
    function solve

        Solve overdetermined problem

        optional alg= :textbook or :hessian
"""
function solve(op::OverdeterminedProblem; alg=:textbook)
    if alg == :textbook
        return solve_textbook(op)
    else
        alg == :hessian
        return solve_hessian(op)
    end
end
"""
function solve_textbook

    Solves overdetermined problem
    y = Ex with associated uncertainty Cnn⁻¹ that is used for weighting
           optionally, can have prior information as well 

    x̃ = (EᵀCnn⁻¹E)⁻¹[(Cnn⁻¹E)ᵀy]
    Cx̃x̃ = (EᵀCnn⁻¹E)⁻¹

    If prior information (Cxx⁻¹, x₀) is available
    x̃ = (EᵀCnn⁻¹E + Cxx⁻¹)⁻¹[(Cnn⁻¹E)ᵀy + Cxx⁻¹x₀]
    Cx̃x̃ = (EᵀCnn⁻¹E)⁻¹

    See equations 1.208/1.209 in Dynamical Insights from Data
"""
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
"""
function solve_hessian

    Solving y = Ex

    x̃ = -1/2 (Eᵀ(Cnn⁻¹E))⁻¹ -(Eᵀ 2(Cnn⁻¹y))
    Cx̃x̃ = (Eᵀ(Cnn⁻¹E))⁻¹
    (maybe???? doublecheck with Jake) 
"""
function solve_hessian(op::OverdeterminedProblem)
    #the two following functions will iterate over NamedTuples
    ∂J∂x = gradient(op) #-(Eᵀ∂J∂n) 
    H⁻¹ = inv(hessian(op)) #hessian = Eᵀ(Cnn⁻¹E) or Eᵀ(Cnn⁻¹E) + Cxx⁻¹
    x = -(1//2)*H⁻¹*∂J∂x 
    (~ismissing(op.x₀)) && (x += op.x₀)
    return Estimate( x, H⁻¹)
end

"""
function misfitgradient
    
    returns 2(Cnn⁻¹y) or 2[Cnn⁻¹(y - Ex₀)] 
"""
function misfitgradient(op::OverdeterminedProblem)
    if ismissing(op.x₀)
        n = op.y 
    else
        n = op.y - op.E*op.x₀
    end
    return 2 *(op.Cnn⁻¹*n)
end

"""
function gradient

    compute ∂J∂x = -(Eᵀ∂J∂n) = -2(Eᵀ(Cnn⁻¹y)) or -2Eᵀ[Cnn⁻¹(y - Ex₀)] 
    
"""
function gradient(op::OverdeterminedProblem)
    ∂J∂n = misfitgradient(op) #2(Cnn⁻¹y) or 2[Cnn⁻¹(y - Ex₀)] 
    if typeof(op.E) <: NamedTuple
        return -sum(transpose(op.E)*∂J∂n)
    else
        return -(transpose(op.E)*∂J∂n) # annoying
    end
 end

#parse error
#hessian(op::OverdeterminedProblem) = ismissing(op.Cxx⁻¹) ? return transpose(op.E)*(op.Cnn⁻¹*op.E) : return transpose(op.E)*(op.Cnn⁻¹*op.E) + op.Cxx⁻¹

"""
function hessian

    compute Eᵀ(Cnn⁻¹E) or Eᵀ(Cnn⁻¹E) + Cxx⁻¹
    depending on if prior is available 
"""
function hessian(op::OverdeterminedProblem)
    if ismissing(op.Cxx⁻¹)
        #return transpose(op.E)*(op.Cnn⁻¹*op.E)
        return symmetric_innerproduct(op.E,op.Cnn⁻¹)
    else
        return symmetric_innerproduct(op.E,op.Cnn⁻¹) + op.Cxx⁻¹
        #return transpose(op.E)*(op.Cnn⁻¹*op.E) + op.Cxx⁻¹
    end
end

symmetric_innerproduct(E::Union{AbstractVector,AbstractMatrix}) = transpose(E)*E
symmetric_innerproduct(E::Union{AbstractVector,AbstractMatrix},Cnn⁻¹) = transpose(E)*(Cnn⁻¹*E)
"""
    for NamedTuple, add up each Hessian contribution
"""
symmetric_innerproduct(E::NamedTuple) = sum(transpose(E)*E)
symmetric_innerproduct(E::NamedTuple,Cnn⁻¹::NamedTuple) = sum(transpose(E)*(Cnn⁻¹*E))
                                                    
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
    function solve

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
        Cnn⁻¹ = inv(p.Cnn) # not possible for NamedTuple
    elseif typeof(p) == OverdeterminedProblem
        Cnn⁻¹ = p.Cnn⁻¹
    end
    return symmetric_innerproduct(n,Cnn⁻¹)
end

"""
    Cost function contribution from control vector
"""
function controlcost( x̃::Estimate, p::Union{OverdeterminedProblem,UnderdeterminedProblem})
    # not implemented yet
    Δx = x̃.v - p.x₀
    if typeof(p) == UnderdeterminedProblem
        Cxx⁻¹ = inv(p.Cxx) # not possible for NamedTuple
    elseif typeof(p) == OverdeterminedProblem
        Cxx⁻¹ = p.Cxx⁻¹
    end
    #return transpose(Δx)*(Cxx⁻¹*Δx)
    return symmetric_innerproduct(Δx,Cxx⁻¹)
end

#Matrix multiply, Mx
"""
    multiplication for `NamedTuple`s
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
function *(b::T,A::NamedTuple) where T<:Number
    
    # Update to use parametric type to set type of Vector
    # Overwriting A would be more efficient
    c = Vector(undef, length(A))
    #c = similar(A)
    for (i, V) in enumerate(A)
        c[i] = V*b # b index safe?
    end
    return NamedTuple{keys(A)}(c)
end
function -(A::NamedTuple) 
    
    # Update to use parametric type to set type of Vector
    # Overwriting A would be more efficient
    c = Vector(undef, length(A))
    for (i, V) in enumerate(A)
        c[i] = -V # b index safe?
    end
    return NamedTuple{keys(A)}(c)
end
function -(A::NamedTuple,B::NamedTuple) 
    
    # Update to use parametric type to set type of Vector
    # Overwriting A would be more efficient
    c = Vector(undef, length(A))
    for (i, V) in enumerate(A)
        c[i] = A[i]-B[i] # b index safe?
    end
    return NamedTuple{keys(A)}(c)
end
function +(A::NamedTuple,B::NamedTuple) 
    
    # Update to use parametric type to set type of Vector
    # Overwriting A would be more efficient
    c = Vector(undef, length(A))
    for (i, V) in enumerate(A)
        c[i] = A[i]+B[i] # b index safe?
    end
    return NamedTuple{keys(A)}(c)
end
function sum(A::NamedTuple) 
    # Update to use parametric type to initialize output
    Asum = 0 * A[1] # a kludge
    for (i, V) in enumerate(A)
        Asum += V 
    end
    return Asum
end
function transpose(A::NamedTuple) 
    
    # Update to use parametric type to set type of Vector
    c = Vector(undef, length(A))
    for (i, V) in enumerate(A)
        c[i] = transpose(V) # b index safe?
    end
    return NamedTuple{keys(A)}(c)
end

end # module
