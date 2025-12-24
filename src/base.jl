"""    
    Matrix multiplication for Estimate includes
    error propagation.
"""
*(F::AbstractMatrix,x::Estimate) = Estimate(F*x.v,F*(x.P*transpose(F)))
# *(F::AbstractMatrix,x::Estimate) = Estimate(F*x.v,F*transpose(F*x.P))

"""    
    Matrix left divide for Estimate includes
    error propagation.
"""
function \(F::AbstractMatrix,x::Estimate)
    v = F\x.v
    b = F\x.P
    P = transpose(F\transpose(b)) 
    return Estimate(v,P)
end

"""    
    Matrix addition for Estimate includes
    error propagation. Follow pp. 95, Sec. 2.5.5,
    Recursive Least Squares, "Dynamical Insights from Data" class notes
"""
+(x::Estimate,y::Estimate) = error("not implemented yet")
