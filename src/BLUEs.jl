module BLUEs

using LinearAlgebra, Statistics, Unitful, UnitfulLinearAlgebra, Measurements
using DimensionalData
using DimensionalData:AbstractDimArray
using DimensionalData:AbstractDimMatrix
using DimensionalData:AbstractDimVector

export Estimate, DimEstimate, OverdeterminedProblem, UnderdeterminedProblem
export solve, show, cost, datacost, controlcost
export rmserror, rmscontrol
export expectedunits, impulseresponse, convolve
export predictobs, addcontrol, addcontrol!, flipped_mult
#export DimEstimate

import Base: show, getproperty, propertynames, *, +, -, sum
import LinearAlgebra: pinv, transpose

"""
struct Estimate{Tv <: Number, TP <: Number, Nv, NP}

a structure with some vector of values x and associated uncertainty matrix P

# Fields
-   `v :: AbstractArray{T, Nv}`
-   `P :: AbstractArray{T, NP}`
"""
struct Estimate{Tv <: Number, TP, Nv, NP} 
    v :: AbstractArray{Tv, Nv}
    P :: AbstractArray{TP, NP}
end

#include("dim_estimate.jl")
include("base.jl")
include("unitful.jl")
include("dimensional_data.jl")
include("blockdim.jl")
include("overdetermined_problem.jl")
include("underdetermined_problem.jl")
include("named_tuple.jl")
include("deprecated.jl")

function show(io::IO, mime::MIME{Symbol("text/plain")}, x::Estimate)
    #function show(io::IO, mime::MIME{Symbol("text/plain")}, x::Union{DimEstimate,Estimate})
    summary(io, x); println(io)
    println(io, "Estimate and 1σ uncertainty")
    show(io, mime, x.x)
end

standard_error(P::AbstractArray) = .√diag(P)

"""
    function getproperty(x::Estimate, d::Symbol)

# Fields of Estimate
- `x::Vector{Measurement}`: Estimate and 1σ uncertainty
- `err::Vector{Number}`: 1σ uncertainty
- `val::Vector{Number}`: central value of estimate
- `P::Matrix{Number}`: estimate uncertainty matrix
"""
function getproperty(x::Estimate, d::Symbol)
    if d === :σ
        if x.P isa DimArray # requires N = 2 where `diag` is defined
            return DimArray(.√diag(x.P),first(dims(x.P)))
        elseif x.P isa UnitfulDimMatrix # requires N = 2 where `diag` is defined
        # assumes, does not check, that unitdims are consistent with diag of P 
            return UnitfulDimMatrix(ustrip.(.√diag(x.P)),unitdims(x.v),dims=first(dims(x.P)))
        else
            return standard_error(x.P) # .√diag(x.P)
        end
    elseif d === :x
        if x.v isa UnitfulDimMatrix # to accomodate previous block, perhaps need to expand to AbstractUnitfulType
            tmp = measurement.(parent(x.v),parent(x.σ))
            return UnitfulDimMatrix(tmp, unitdims(x.v), dims= dims(x.v))
        else
            return measurement.(x.v,x.σ)
        end
    else
        return getfield(x, d)
    end
end

Base.propertynames(x::Estimate) = (:x, :σ, fieldnames(typeof(x))...)

symmetric_innerproduct(E::Union{AbstractVector,AbstractMatrix}) = transpose(E)*E
symmetric_innerproduct(E::AbstractMatrix,Cnn⁻¹) = transpose(E)*(Cnn⁻¹*E)
symmetric_innerproduct(n::AbstractVector,Cnn⁻¹) = n ⋅ (Cnn⁻¹*n)
"""
    for NamedTuple, add up each Hessian contribution
"""
symmetric_innerproduct(E::NamedTuple) = sum(transpose(E)*E)
symmetric_innerproduct(E::NamedTuple,Cnn⁻¹::NamedTuple) = sum(transpose(E)*(Cnn⁻¹*E))
                                                    

"""    
    Matrix multiplication for Estimate includes
    error propagation.
"""
*(F::AbstractMatrix,x::Estimate) = Estimate(F*x.v,F*x.P*transpose(F))

"""    
    Matrix addition for Estimate includes
    error propagation. Follow pp. 95, Sec. 2.5.5,
    Recursive Least Squares, "Dynamical Insights from Data" class notes
"""
+(x::Estimate,y::Estimate) = error("not implemented yet")

"""
    Cost function contribution from observations
"""
function datacost( x̃::Estimate, p::Union{OverdeterminedProblem,UnderdeterminedProblem})
    y = p.y
    n = y - p.E*x̃.v
    if typeof(p) == UnderdeterminedProblem
        Cnn⁻¹ = inv(p.Cnn) # not possible for NamedTuple
    elseif typeof(p) == OverdeterminedProblem
        Cnn⁻¹ = p.Cnn⁻¹
    end
    return symmetric_innerproduct(n,Cnn⁻¹)
end

"""
    function rmserror

    compute nᵀn, how closely are we fitting the data?
    unweighted `datacost`
"""
function rmserror(x̃::Estimate, p::Union{OverdeterminedProblem, UnderdeterminedProblem})
#function rmserror(x̃::Union{Estimate, DimEstimate}, p::Union{OverdeterminedProblem, UnderdeterminedProblem})
    n = p.y - p.E*x̃.v
    return sqrt(n ⋅ n)
end

"""
    function rmscontrol

    compute (ũ-u₀)ᵀ(ũ-u₀), how much are we adjusting the control?
    unweighted `controlcost`

    # Args 
    -`x̃`: DimEstimate
    -`p`: problem
    -`dim3`: allows to access third dimension, which is assumed to be state var. dim.
"""
function rmscontrol(x̃::Estimate, p::Union{OverdeterminedProblem, UnderdeterminedProblem}, dim3 = nothing)
    Δx = x̃.v - p.x₀
    return sqrt(Δx ⋅ Δx)
end

"""
    Cost function contribution from control vector
"""
function controlcost( x̃::Estimate, p::Union{OverdeterminedProblem,UnderdeterminedProblem})
    Δx = x̃.v - p.x₀
    
    if typeof(p) == UnderdeterminedProblem
        Cxx⁻¹ = inv(p.Cxx) # not possible for NamedTuple
    elseif typeof(p) == OverdeterminedProblem
        Cxx⁻¹ = p.Cxx⁻¹
    end
    return symmetric_innerproduct(Δx,Cxx⁻¹)
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

"""
    function predictobs(funk,x...)

    Get observations derived from function `funk`
    y = funk(x...)
    Turns out to not be useful so far.
"""
predictobs(funk,x...) = funk(x...)

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
