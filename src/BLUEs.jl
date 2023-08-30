module BLUEs

#using Revise
using LinearAlgebra, Statistics, Unitful, UnitfulLinearAlgebra, Measurements
using DimensionalData

export Estimate, DimEstimate, OverdeterminedProblem, UnderdeterminedProblem
export solve, show, cost, datacost, controlcost
export expectedunits, impulseresponse, convolve
export predictobs, addcontrol, addcontrol!, flipped_mult

import Base: show, getproperty, propertynames, (*), (+), (-), sum
#import Base.:*
import LinearAlgebra: pinv, transpose

#struct Measurement{T<:AbstractFloat} <: AbstractFloat

"""
    struct Estimate{Tv <: Number,TC <: Number} 
    
a structure with some vector of values v and associated covariance matrix C

# Fields
-   `v :: AbstractVector{Tv}`
-   `C :: AbstractMatrix{TC}`
"""
struct Estimate{Tv <: Number,TC <: Number} 
    v :: AbstractVector{Tv}
    C :: AbstractMatrix{TC}
end

"""
    struct DimEstimate{Tv <: Number,TC <: Number} 
    
    A structure with some vector of values v and associated covariance matrix C.
Differs from `Estimate` in that axis dimensions are added.

# Fields
-   `v :: AbstractVector{Tv}`
-   `C :: AbstractMatrix{TC}`
-   `dims :: Tuple`
"""
struct DimEstimate{Tv <: Number,TC <: Number} 
    v :: AbstractVector{Tv}
    C :: AbstractMatrix{TC}
    dims :: Tuple
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
    x₀ :: Union{AbstractVector,AbstractDimArray,Missing}
end


"""
    function UnderdeterminedProblem

    generates UnderdeterminedProblem structure with x₀ = missing, still requires Cxx 
"""
UnderdeterminedProblem(y::AbstractVector,E::AbstractMatrix,Cnn::AbstractMatrix,Cxx::AbstractMatrix) = OverdeterminedProblem(y,E,Cnn,Cxx,missing)


function show(io::IO, mime::MIME{Symbol("text/plain")}, x::Union{DimEstimate,Estimate})
    summary(io, x); println(io)
    println(io, "Estimate and 1σ uncertainty")
    show(io, mime, x.x)
end

function show(io::IO, mime::MIME{Symbol("text/plain")}, x::DimArray{Quantity{Float64}, 3})
    summary(io, x); println(io)
    statevars = x.dims[3]
    for (i, s) in enumerate(statevars)
        if i != 1
            println()
        end
        
        println(io, "State Variable " * string(i) * ": " * string(s))
        show(io, mime, x[:,:,At(s)])
    end
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
        # x.v can be a UnitfulVector, so wrap with Matrix
        if x.v isa UnitfulLinearAlgebra.AbstractUnitfulType
            v = Matrix(x.v)
        else
            v = x.v
        end
        return measurement.(v,x.σ)
        #return x.v .± x.σ
    else
        return getfield(x, d)
    end
end
function getproperty(x::DimEstimate, d::Symbol)
    if d === :σ
        return .√diag(x.C)
    elseif d === :x

        # x.v can be a UnitfulVector, so wrap with vec
        # wrapping with Matrix causes type consistency issues
        if x.v isa UnitfulLinearAlgebra.AbstractUnitfulType
            #v = Matrix(x.v)
            v = vec(x.v)
        else
            v = x.v
        end

        # types of v and x.σ should be consistent
        # should there be a conditional for σ (like v) above?
        tmp = measurement.(v,x.σ)
        return DimArray(reshape(tmp,size(x.dims)),x.dims)
    #return x.v .± x.σ
    else
        return getfield(x, d)
    end
end

Base.propertynames(x::Estimate) = (:x, :σ, fieldnames(typeof(x))...)
Base.propertynames(x::DimEstimate) = (:x, :σ, fieldnames(typeof(x))...)

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
symmetric_innerproduct(E::AbstractMatrix,Cnn⁻¹) = transpose(E)*(Cnn⁻¹*E)
symmetric_innerproduct(n::AbstractVector,Cnn⁻¹) = n ⋅ (Cnn⁻¹*n)
# UnitfulLinearAlgebra returns UnitfulMatrix type, convert to scalar
#symmetric_innerproduct(E::AbstractUnitfulMatrix) = Matrix(transpose(E)*E)
#symmetric_innerproduct(E::AbstractUnitfulMatrix,Cnn⁻¹::AbstractUnitfulMatrix) = Matrix(transpose(E)*(Cnn⁻¹*E))
"""
    for NamedTuple, add up each Hessian contribution
"""
symmetric_innerproduct(E::NamedTuple) = sum(transpose(E)*E)
symmetric_innerproduct(E::NamedTuple,Cnn⁻¹::NamedTuple) = sum(transpose(E)*(Cnn⁻¹*E))
                                                    
"""
    function pinv

    Left pseudo-inverse (i.e., least-squares estimator)
"""
function LinearAlgebra.pinv(op::OverdeterminedProblem)
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
        if up.x₀ isa DimArray
            x₀ = UnitfulMatrix(up.x₀[:])
        else
            x₀ = up.x₀
        end
        n = up.y - up.E*x₀
    end
    Cxy = up.Cxx*transpose(up.E)
    Cyy = up.E*Cxy + up.Cnn
    v = Cxy*(Cyy \ n)
    (~ismissing(up.x₀)) && (v += x₀)
    P = up.Cxx - Cxy*(Cyy\(transpose(Cxy)))
    if up.x₀ isa DimArray
        return DimEstimate(v,P,dims(up.x₀))
    else
        return Estimate(v,P)
    end
end

"""    
    Matrix multiplication for Estimate includes
    error propagation.
"""
*(F::AbstractMatrix,x::Estimate) = Estimate(F*x.v,F*x.C*transpose(F))
*(F::UnitfulDimMatrix,x::DimEstimate) = DimEstimate(F*x.v,F*x.C*transpose(F),x.dims)
*(F::UnitfulMatrix,x::DimEstimate) = Estimate(F*x.v,F*x.C*transpose(F))

"""    
    Matrix addition for Estimate includes
    error propagation. Follow pp. 95, Sec. 2.5.5,
    Recursive Least Squares, "Dynamical Insights from Data" class notes
"""
+(x::Estimate,y::Estimate) = error("not implemented yet")
+(x::DimEstimate,y::DimEstimate) = error("not implemented yet")

"""
    Compute cost function
"""
function cost(x̃::Estimate,op::OverdeterminedProblem)
    Jdata = datacost(x̃,op)
    (~ismissing(op.x₀) && ~ismissing(op.Cxx⁻¹)) ? Jcontrol = controlcost(x̃,op) : Jcontrol = nothing
    isnothing(Jcontrol) ? J = Jdata : J = Jdata + Jcontrol
    return J
end
function cost(x̃::Union{Estimate,DimEstimate},up::UnderdeterminedProblem)
   Jdata = datacost(x̃,up)
   Jcontrol = controlcost(x̃,up) 
   J = Jdata + Jcontrol
   return J
end

"""
    Cost function contribution from observations
"""
function datacost( x̃::Union{Estimate,DimEstimate}, p::Union{OverdeterminedProblem,UnderdeterminedProblem})
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
function controlcost( x̃::Union{Estimate,DimEstimate}, p::Union{OverdeterminedProblem,UnderdeterminedProblem})
    if p.x₀ isa DimArray
        Δx = x̃.v - UnitfulMatrix(vec(p.x₀))
    else
        Δx = x̃.v - p.x₀
    end
    
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
function *(A::NamedTuple, b::AbstractVector) 
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
function LinearAlgebra.transpose(A::NamedTuple) 
    
    # Update to use parametric type to set type of Vector
    c = Vector(undef, length(A))
    for (i, V) in enumerate(A)
        c[i] = transpose(V) # b index safe?
    end
    return NamedTuple{keys(A)}(c)
end

"""
function impulseresponse(x₀,M)

    Probe a function to determine its linear response in matrix form.
    Assumes units are needed and available.
    A simpler function to handle cases without units would be nice.

    funk:: function to be probed
    x:: input variable
    args:: the arguments that follow x in `funk`
"""
function impulseresponse(funk::Function,x,args...)
    u = Quantity.(zeros(length(x)),unit.(vec(x)))
    y₀ = funk(x,args...)
    #Eunits = expectedunits(y₀,x)
    #Eu = Quantity.(zeros(length(y₀),length(x)),Eunits)
    Ep = zeros(length(y₀), length(x))
    
    for rr in eachindex(x)
        #u = zeros(length(x)).*unit.(x)[:]
        if length(x) > 1
            u = Quantity.(zeros(length(x)), unit.(vec(x)))
        else
            u = Quantity(0.0,unit(x))
        end
        
        Δu = Quantity(1.0,unit.(x)[rr])
        u[rr] += Δu
        x₁ = addcontrol(x,u)
        y = funk(x₁,args...)
        #println(y)
        if y isa AbstractDimArray
            #Ep[:,rr] .= vec(parent((y - y₀)/Δu))
            Ep[:, rr] .= ustrip.(vec(parent((y - y₀)/Δu)))
   
        else
            tmp = ustrip.((y - y₀)/Δu)
            # getting ugly around here
            if tmp isa Number
                Ep[:,rr] .= tmp
            else
                Ep[:,rr] .= vec(tmp)
            end
        end
    end
    
    
    # This function could use vcat to be cleaner (but maybe slower)
    # Note: Need to add ability to return sparse matrix 
    #return E = UnitfulMatrix(Ep,unit.(vec(y₀)),unit.(vec(x)))
    #return E = UnitfulMatrix_from_input_output(Eu,y₀,x)
    #return E = UnitfulMatrix(Eu)
    if length(x) == 1 && length(y₀) == 1
        return UnitfulMatrix(Ep,[unit(y₀)],[unit(x)])
    elseif length(y₀) == 1
        return UnitfulMatrix(Ep,[unit(y₀)],vec(unit.(x)))
    elseif length(x) == 1
        return UnitfulMatrix(Ep,vec(unit.(y₀)),[unit(x)])
    else
        return UnitfulMatrix(Ep,vec(unit.(y₀)),vec(unit.(x)))
    end
end

function UnitfulMatrix_from_input_output(Eu,y,x)
    if length(x) == 1 && length(y) == 1
        return UnitfulMatrix(ustrip.(Eu),[unit(y)],[unit(x)])
    elseif length(y) == 1
        return UnitfulMatrix(ustrip.(Eu),[unit.(y)],vec(unit.(x)))
    elseif length(x) == 1
        return UnitfulMatrix(ustrip.(Eu),vec(unit.(y)),[unit(x)])
    else
        return UnitfulMatrix(ustrip.(Eu),vec(unit.(y)),vec(unit.(x)))
    end
end

function expectedunits(y,x)
    Eunits = Matrix{Unitful.FreeUnits}(undef,length(y),length(x))
    for ii in eachindex(y)
        for jj in eachindex(x)
            if length(y) == 1 && length(x) ==1
                Eunits[ii,jj] = unit(y)/unit(x)
            elseif length(x) == 1
                Eunits[ii,jj] = unit.(y)[ii]/unit(x)
            elseif length(y) ==1
                Eunits[ii,jj] = unit(y)/unit.(x)[jj]
            else
                Eunits[ii,jj] = unit.(y)[ii]/unit.(x)[jj]
            end
        end
    end
    return Eunits
end

"""
    function convolve(E::AbstractDimArray,x::AbstractDimArray)

    Take the convolution of E and x
    Account for proper overlap of dimensions
    Sum and take into account units.
"""
function convolve(x::AbstractDimArray,E::AbstractDimArray)
    tnow = last(first(dims(x)))
    lags = first(dims(E))
    println("CONVOLVE WARNING: ISSUE 48") 
    return sum([E[ii,:] ⋅ x[Near(tnow-ll),:] for (ii,ll) in enumerate(lags)])
end
"""
function convolve(x::AbstractDimArray, E::AbstractDimArray, coeffs::UnitfulMatrix}
    the `coeffs` argument signifies that x is a 3D array (i.e. >1 state variables)

    this function both convolves, and linearly combines the propagated state variables
"""
function convolve(x::AbstractDimArray,E::AbstractDimArray, coeffs::UnitfulMatrix)
    statevars = x.dims[3]
    mat = UnitfulMatrix(transpose([convolve(x[:,:,At(s)], E) for s in statevars])) * coeffs
    return getindexqty(mat, 1,1)
end

function convolve(x::AbstractDimArray,E::AbstractDimArray,t::Number)
    lags = first(dims(E))  
    #return sum([E[ii,:] ⋅ x[Near(t-ll),:] for (ii,ll) in enumerate(lags)])
    t1 = x.dims[1][1]
    return sum([E[ii,:] ⋅ x[Near(t-ll),:] for (ii,ll) in enumerate(lags) if t - ll >= t1])
end

#coeffs signifies that x is 3D 
function convolve(x::AbstractDimArray, E::AbstractDimArray, t::Number, coeffs::UnitfulMatrix)
    statevars = x.dims[3]
    mat = UnitfulMatrix(transpose([convolve(x[:,:,At(s)], E, t) for s in statevars]))*coeffs
    return getindexqty(mat, 1,1) 
end

#don't handle the ndims(M) == 3 case here but I'll get back to it
function convolve(x::AbstractDimArray, M::AbstractDimArray, Tx::Union{Ti, Vector}, coeffs::UnitfulMatrix)
    if ndims(M) == 2
        return DimArray([convolve(x,M,Tx[tt], coeffs) for (tt, yy) in enumerate(Tx)], Tx)
    elseif ndims(M) == 3
        error("some code should go here")
    else
        error("M has wrong number of dims") 
    end
    
    
end


function convolve(x::AbstractDimArray,M::AbstractDimArray,Tx::Union{Ti,Vector})
    if ndims(M) == 2 
        return DimArray([convolve(x,M,Tx[tt]) for (tt,yy) in enumerate(Tx)],Tx)
    elseif ndims(M) == 3

        # do a sample calculation to get units.
        Msmall = M[:,:,1] # assumes M.dims[3] == y.dims[2]
        yunit = unit.(vec(convolve(x,Msmall,Tx))[1]) # assume y has the same units

        y = DimArray(zeros(length(Tx),last(size(M)))yunit,(Tx,last(dims(M))))
        for (ii,vv) in enumerate(last(dims(M)))
            
            Msmall = M[:,:,ii]
            y[:,ii] = convolve(x,Msmall,Tx)

        end
        return y
    else
        error("M has wrong number of dims")
    end
end

# function convolve(x::AbstractDimArray,F::Tuple)
#     E = F[1]
#     tnow = last(first(dims(x)))
#     lags = first(dims(E))
#     return sum([E[ii,:] ⋅ x[Near(tnow-ll),:] for (ii,ll) in enumerate(lags)])
# end

"""
    function predictobs(funk,x...)

    Get observations derived from function `funk`
    y = funk(x...)
    Turns out to not be useful so far.
"""
function predictobs(funk,x...)
    #x = addcontrol(x₀,u) 
    return y = funk(x...)
end

function addcontrol(x₀::AbstractDimArray,u)

    x = deepcopy(x₀)
    ~isequal(length(x₀),length(u)) && error("x₀ and u different lengths")
    for ii in eachindex(x₀)
        # check units
        ~isequal(unit(x₀[ii]),unit(u[ii])) && error("x₀ and u different units")
        x[ii] += u[ii]
    end
    return x
end

function addcontrol!(x::AbstractDimArray,u)

    ~isequal(length(x),length(u)) && error("x and u different lengths")
    for ii in eachindex(x)
        # check units
        ~isequal(unit(x[ii]),unit(u[ii])) && error("x and u different units")
        x[ii] += u[ii]
    end
    return x
end

"""
function flipped_mult

    multiply in opposite order given, needs to be defined for impulseresponse
"""
flipped_mult(a,b) = b*a

end # module
