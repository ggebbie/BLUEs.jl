"""
struct DimEstimate 
    x :: DimArray
    P :: DimArray
end

# Fields
-   `x :: DimArray`: central estimate
-   `P :: DimArray`: uncertainty of estimate
"""
struct DimEstimate{T,N,D<:Tuple,R<:Tuple,A<:AbstractArray{T,N},Na,Me,DA<:AbstractDimArray{T,N,D,A}}
    x :: DA
    P :: DA
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
Base.propertynames(x::DimEstimate) = (:x, :σ, fieldnames(typeof(x))...)

"""
    function solve

Solve underdetermined problem

this version requires the following to be trun
 up.x₀ isa DimArray
 up.y isa DimArray

"""
function solve(up::UnderdeterminedProblem)
    y = UnitfulMatrix(ustrip.(vec(up.y)), unit.(vec(up.y)))
   
    if ismissing(up.x₀)
        n = y
    else
        x₀ = UnitfulMatrix(ustrip.(vec(up.x₀)), unit.(vec(up.x₀)))
        n = y - up.E*x₀
    end
    Cxy = up.Cxx*transpose(up.E)
    Cyy = up.E*Cxy + up.Cnn
    v = Cxy*(Cyy \ n)
    (~ismissing(up.x₀)) && (v += x₀)
    P = up.Cxx - Cxy*(Cyy\(transpose(Cxy)))
    return DimEstimate(v,P,dims(up.x₀))
end

*(F::UnitfulDimMatrix,x::DimEstimate) = DimEstimate(F*x.v,F*x.C*transpose(F),x.dims)
*(F::UnitfulMatrix,x::DimEstimate) = Estimate(F*x.v,F*x.C*transpose(F))
+(x::DimEstimate,y::DimEstimate) = error("not implemented yet")

function cost(x̃::Union{Estimate,DimEstimate},up::UnderdeterminedProblem)
#function cost(x̃::Union{Estimate,DimEstimate},up::UnderdeterminedProblem)
   Jdata = datacost(x̃,up)
   Jcontrol = controlcost(x̃,up) 
   J = Jdata + Jcontrol
   return J
end


"""
Cost function contribution from observations

requires  p.y isa DimArray
"""
function datacost( x̃::DimEstimate, p::Union{OverdeterminedProblem,UnderdeterminedProblem})
    y = UnitfulMatrix(ustrip.(vec(p.y)), unit.(vec(p.y)))
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
function rmserror(x̃::DimEstimate, p::Union{OverdeterminedProblem, UnderdeterminedProblem})
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
function rmscontrol(x̃::Union{Estimate, DimEstimate}, p::Union{OverdeterminedProblem, UnderdeterminedProblem}, dim3 = nothing)
    
    if p.x₀ isa DimArray
        if isnothing(dim3) #no state var, assume consistent units, 2D
            Δx = x̃.v - UnitfulMatrix(vec(p.x₀))
        else #there is a statevariable and we need to index 
            Δx = UnitfulMatrix(Measurements.value.(vec(x̃.x[:, :, At(dim3)] - p.x₀[:, :, At(dim3)])))
        end
        
    else
        Δx = x̃.v - p.x₀
    end
    return sqrt(Δx ⋅ Δx)
end

"""
    Cost function contribution from control vector
"""
function controlcost( x̃::Union{Estimate,DimEstimate}, p::Union{OverdeterminedProblem,UnderdeterminedProblem})
    if p.x₀ isa DimArray
        Δx = x̃.v - UnitfulMatrix(ustrip.(vec(p.x₀)), unit.(vec(p.x₀)))
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
