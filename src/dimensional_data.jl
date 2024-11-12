
ext = Base.get_extension(AlgebraicArrays, :AlgebraicArraysDimensionalDataExt)
if !isnothing(ext)
    RowVector = ext.RowVector
end

function show(io::IO, mime::MIME{Symbol("text/plain")}, x::DimArray{T, 3}) where T <: Number 
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

function standard_error(P::DimArray)
    #sigma = similar(parent(P))
    if dimensionless(first(first(P))) # assumes first element gives right answer
        inside_type = typeof(√P[1][1])
        sigma = Array{inside_type}(undef,size(P))
    else
        sigma = Array{Quantity}(undef,size(P))
    end
    
    for i in eachindex(P)
        sigma[i] = √P[i][i]
    end
    return DimArray(sigma,dims(P))
end
# function standard_error(P::DimArray)
#     #sigma = similar(parent(P))
#     sigma = Array{eltype(eltype(P))}(undef,size(P))
#     for i in eachindex(P)
#         sigma[i] = √P[i][i]
#     end
#     return DimArray(sigma,dims(P))
# end
# Failed to add dispatch to handle when eltype is Quantity
# function standard_error(P::DimArray{DimArray{T}}) where T <: Quantity 
#     #sigma = similar(parent(P))
#     sigma = Array{eltype(eltype(P))}(undef,size(P))
#     for i in eachindex(P)
#         sigma[i] = √P[i][i]
#     end
#     return DimArray(sigma,dims(P))
# end

#standard_error(P::AbstractDimArray{T,2}) where T <: Number = DimArray(.√diag(P),first(dims(P)))

# function uncertainty_units(x::DimArray{T}) where T<: Number

#     unitlist = unit.(x)
#     U = Array{typeof(unitlist)}(undef,size(unitlist))

#     for i in eachindex(U)
#         U[i] = similar(unitlist)# Array{typeof(unitlist)}(undef,size(unitlist))
#         for j in eachindex(U)
#             U[i][j] = unitlist[i]*unitlist[j]
#         end
#     end
#     return DimArray(U,dims(x))
# end

# function diagonalmatrix_with_units(x::DimArray{T}) where T<: Number

#     unitlist = 1.0.*unit.(x)
#     U = Array{typeof(unitlist)}(undef,size(unitlist))

#     for i in eachindex(U)
#         U[i] = similar(unitlist)
#         #U[i] = Array{Quantity}(undef,size(unitlist))
#         for j in eachindex(U)
#             if i == j 
#                 U[i][j] = unitlist[i]*unitlist[j]
#             else
#                 U[i][j] = 0.0.*unitlist[i]*unitlist[j]
#             end
#         end
#     end
#     return DimArray(U,dims(x))
# end

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

include("convolutions.jl")
