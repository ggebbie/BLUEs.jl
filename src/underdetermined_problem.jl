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
    Cxx :: Union{AbstractMatrix, Missing}
    x₀ :: Union{AbstractVector, Missing}
    # x₀ :: Union{AbstractVector, AbstractDimArray, Missing}
end

"""
    function UnderdeterminedProblem

    generates UnderdeterminedProblem structure with x₀ = missing, still requires Cxx 
"""
UnderdeterminedProblem(y::AbstractVector,E::AbstractMatrix,Cnn::AbstractMatrix,Cxx::AbstractMatrix) = OverdeterminedProblem(y,E,Cnn,Cxx,missing)

"""
    function solve

        Solve underdetermined problem
"""
function solve(up::UnderdeterminedProblem)
    y = up.y
    if ismissing(up.x₀)
        n = y
    else
        x₀ = up.x₀
        n = y - up.E*x₀
    end
    Cxy = up.Cxx*transpose(up.E)
    Cyy = up.E*Cxy + up.Cnn
    v = Cxy*(Cyy \ n)
    (~ismissing(up.x₀)) && (v += x₀)
    P = up.Cxx - Cxy*(Cyy\(transpose(Cxy)))
    return Estimate(v,P)
end

function cost(x̃::Estimate,up::UnderdeterminedProblem)
   Jdata = datacost(x̃,up)
   Jcontrol = controlcost(x̃,up) 
   J = Jdata + Jcontrol
   return J
end
