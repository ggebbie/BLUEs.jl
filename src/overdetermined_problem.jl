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
    if ismissing(op.Cxx⁻¹)
        iECE = inv(transpose(op.E)*CE)
        return Estimate( iECE * (transpose(CE)*op.y), iECE)
    else
        # prior information available
        iECE = inv(transpose(op.E)*CE + op.Cxx⁻¹)
        rhs = transpose(CE)*op.y
        (~ismissing(op.x₀)) && (rhs += op.Cxx⁻¹*op.x₀)
        return Estimate( iECE * rhs, iECE)
    end
end

"""
function solve_hessian

    Solving y = Ex

H = Eᵀ(Cnn⁻¹E) + Cxx⁻¹
∂J/∂x =  -E^T ∂J/∂n
∂J/∂n = 2n
n = y - Ex₀
x̃ = -1/2 H⁻¹ ∂J/∂x
"""
function solve_hessian(op::OverdeterminedProblem)
    #the two following functions will iterate over NamedTuples
    ∂J∂x = gradient(op) #-(Eᵀ∂J∂n) 
    H⁻¹ = inv(hessian(op)) #hessian = Eᵀ(Cnn⁻¹E) or Eᵀ(Cnn⁻¹E) + Cxx⁻¹
    x = -(1//2)*H⁻¹*∂J∂x
    #H = hessian(op) #hessian = Eᵀ(Cnn⁻¹E) or Eᵀ(Cnn⁻¹E) + Cxx⁻¹
    #x = -(1//2)*(H\∂J∂x)
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

