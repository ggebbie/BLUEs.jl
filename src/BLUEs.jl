module BLUEs

using LinearAlgebra, Statistics, Unitful, Measurements
# using UnitfulLinearAlgebra
# using DimensionalData
# using DimensionalData:AbstractDimArray
# using DimensionalData:AbstractDimMatrix
# using DimensionalData:AbstractDimVector
# using DimensionalData:@dim
# using AlgebraicArrays           # 

export Estimate, OverdeterminedProblem, UnderdeterminedProblem
export combine
export solve, show, cost, datacost, controlcost
export rmserror, rmscontrol
export expectedunits, impulseresponse, convolve
export addcontrol, addcontrol!, flipped_mult

import Base: show, getproperty, propertynames, *, +, -, \, sum
import LinearAlgebra: pinv, transpose

"""
struct Estimate{Tv <: Number, TP <: Number, Nv, NP}

a structure with some vector of values x and associated uncertainty matrix P

# Fields
-   `v :: AbstractArray{T, Nv}`
-   `P :: AbstractArray{T, NP}`
"""
struct Estimate{Tv <: Number, Ta <: Number, V <: AbstractArray{Tv,1}, A <: AbstractArray{Ta,2}} 
    v :: V
    P :: A
end

# if two vectors are provided, assume it is the standard error 
Estimate(v::AbstractVector, sigma::AbstractVector) = Estimate(v, Diagonal(sigma.^2))

# allow scalar estimates and scalar standard errors to be `Estimate`s
Estimate(v::T, sigma::T) where T <: Number = Estimate([v], [sigma^2;;])

# translate Vector{Measurement} to Estimate
# let it error rather than restricting types at compile-time
function Estimate(v::AbstractVector{T}) where T <: Union{<:Measurement, Quantity{<:Measurement}} 
        vval = Measurements.value.(v)
        verr = Measurements.uncertainty.(v);
        return Estimate(vval, verr) # just provide standard error
end 

include("base.jl")
include("unitful.jl")
# include("algebraic_arrays.jl")
# include("unitful_algebraic_arrays.jl")
# include("dimensional_data.jl")
# include("blockdim.jl")
include("overdetermined_problem.jl")
include("underdetermined_problem.jl")
include("named_tuple.jl")
# include("deprecated.jl")

function show(io::IO, mime::MIME{Symbol("text/plain")}, x::Estimate)
    #summary(io, x); println(io) # too long and noisy although informative
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
        return standard_error(x.P) # .√diag(x.P)
    elseif d === :x
        return measurement.(x.v,x.σ)
    else
        return getfield(x, d)
    end
end

Base.propertynames(x::Estimate) = (:x, :σ, fieldnames(typeof(x))...)

function combine(x0::Estimate, x1::Estimate; alg=:underdetermined)

    if alg == :underdetermined

        n = x1.v - x0.v # n = x1 - x0, difference
        v = x0.v + x0.P * ( (x1.P + x0.P) \ n )
        P = x0.P - x0.P * ( (x1.P + x0.P) \ x0.P )
        
    elseif alg == :overdetermined

        iP0 = inv(x0.P)
        iP1 = inv(x1.P)
        sumP = iP0 + iP1
        P = inv(sumP)
        v = P * (iP0 * x0.v + iP1 * x1.v) 
    else
        error("combine method not implemented")
    end
    return Estimate(v, P)
end

function combine(x0::Estimate, y::Estimate, E::AbstractMatrix)

    #if alg == :underdetermined
    if size(E,2) ≥ size(E,1) 
        n = y.v - E * x0.v
        sumP = y.P + E * x0.P * transpose(E) 
        v = x0.v + x0.P * transpose(E) *
            (  sumP \ n )
        P = x0.P - x0.P * transpose(E) *
            ( sumP \ (E * x0.P) )
        
    #elseif alg == :overdetermined
    elseif size(E,1) > size(E,2) 
    
        iPx = inv(x0.P)
        iPy = inv(y.P)
        sumP = iPx + transpose(E)*iPy*E 
        P = inv(sumP)
        v = P * (iPx * x0.v + transpose(E) * iPy * y.v) 
    else
        error("combine method not implemented")
    end
    return Estimate(v, P)
end

"""
    combine(x0::Estimate,y::Estimate,f::Function)

# Arguments
- `x0::Estimate`: estimate 1
- `y::Estimate`: estimate 2
- `f::Function`: function that relates f(x0) = y
# Returns
- `xtilde::Estimate`: combined estimate

```math
{\\bf E}_i (\\tau)  = 
\\frac{1}{N} \\int_{t - \\tau}^{t} {\\bf G}'^{\\dagger} (t^* + \\tau - t) ~ {\\bf D}_i  ~ {\\bf G}' (t - t^*) ~ d t ^* , 
```
"""
function combine(x0::Estimate,y1::Estimate,E1::Function)
    # written for efficiency with underdetermined problems
    Pyx = E1(x0.P) 
    Pxy = transpose(Pyx)
    EPxy = E1(Pxy)
    Py = EPxy + y1.P
    y0 = E1(x0.v)
    n1 = y1.v - y0
    tmp = Py \ n1
    v = Pxy * tmp
    dP = Pxy * (Py \ Pyx)
    P = x0.P - dP
    return Estimate(v,P)
end

symmetric_innerproduct(E::Union{AbstractVector,AbstractMatrix}) = transpose(E)*E
symmetric_innerproduct(E::AbstractMatrix,Cnn⁻¹) = transpose(E)*(Cnn⁻¹*E)
symmetric_innerproduct(n::AbstractVector,Cnn⁻¹) = n ⋅ (Cnn⁻¹*n)
"""
    for NamedTuple, add up each Hessian contribution
"""
symmetric_innerproduct(E::NamedTuple) = sum(transpose(E)*E)
symmetric_innerproduct(E::NamedTuple,Cnn⁻¹::NamedTuple) = sum(transpose(E)*(Cnn⁻¹*E))

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
    n = p.y - p.E*x̃.v
    return sqrt(n ⋅ n)
end

"""
    function rmscontrol

    compute (ũ-u₀)ᵀ(ũ-u₀), how much are we adjusting the control?
    unweighted `controlcost`

    # Args 
    -`x̃`: Estimate
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
    Ep = zeros(length(y₀), length(x))
    for rr in eachindex(x)
        if length(x) > 1
            u = Quantity.(zeros(length(x)), unit.(vec(x)))
        else
            u = Quantity(0.0,unit(x))
        end
        Δu = Quantity(1.0,unit.(x)[rr])
        u[rr] += Δu
        x₁ = addcontrol(x,u)
        y = funk(x₁,args...)
        # if y isa AbstractDimArray
        #     Ep[:, rr] .= ustrip.(vec(parent((y - y₀)/Δu)))
        # else
            tmp = ustrip.((y - y₀)/Δu)
            # getting ugly around here
            if tmp isa Number
                Ep[:,rr] .= tmp
            else
                Ep[:,rr] .= vec(tmp)
            end
        # end
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
function flipped_mult

    multiply in opposite order given, needs to be defined for impulseresponse
"""
flipped_mult(a,b) = b*a

function convolve end

end # module
