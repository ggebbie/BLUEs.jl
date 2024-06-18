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
    x₀ :: Union{AbstractVector, AbstractDimArray, Missing}
end

"""
    function UnderdeterminedProblem

    generates UnderdeterminedProblem structure with x₀ = missing, still requires Cxx 
"""
UnderdeterminedProblem(y::AbstractVector,E::AbstractMatrix,Cnn::AbstractMatrix,Cxx::AbstractMatrix) = OverdeterminedProblem(y,E,Cnn,Cxx,missing)

