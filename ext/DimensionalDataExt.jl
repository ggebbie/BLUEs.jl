module DimensionalDataExt

using BLUEs
using DimensionalData
using DimensionalData:AbstractDimArray
using DimensionalData:AbstractDimMatrix
using DimensionalData:AbstractDimVector
using DimensionalData:@dim
using AlgebraicArrays
using LinearAlgebra

# ext = Base.get_extension(AlgebraicArrays, :AlgebraicArraysExt)
# # ext = Base.get_extension(AlgebraicArrays, :AlgebraicArraysDimensionalDataExt)
# if !isnothing(ext)
#     RowVector = ext.RowVector
# end

# @dim RowVector "singular dimension"

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

struct BlockDimArray{T <: Number, DA <: AbstractDimArray{T}} 
    da :: DA
    blockdims :: Tuple
end

# helper routine for impulse response
BLUEs.response(y::AbstractDimArray,y₀,Δu) = ustrip.(vec(parent((y - y₀)/Δu)))

end 
