module DimensionalDataExt

using BLUEs
using DimensionalData
using DimensionalData:AbstractDimArray
using DimensionalData:AbstractDimMatrix
using DimensionalData:AbstractDimVector
using DimensionalData:@dim
using AlgebraicArrays
using LinearAlgebra

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

"""
function BLUEs.convolve(x::DimArray{T},E::AbstractDimArray) where T <: Number

Take the convolution of E and x
Account for proper overlap of dimension.
Sum and take into account units.
Return an `AbstractDimArray`
"""
function BLUEs.convolve(x::VectorArray,E::AbstractDimArray) 
    tnow = last(first(rangedims(x)))
    lags = first(dims(E))
    vals = sum([E[ii,:] ⋅ x[Near(tnow-ll),:] for (ii,ll) in enumerate(lags)])
    (vals isa Number) ? (return VectorArray(DimArray([vals],first(rangedims(x))))) : (return VectorArray(AlgebraicArray(vals,first(rangedims(x)))))
end

function BLUEs.convolve(x::VectorArray, M::AbstractDimArray, t::Number)
    lags = first(dims(M))
    return sum([M[ii,:] ⋅ x[Near(t-ll),:] for (ii,ll) in enumerate(lags)])
end

#function convolve(x::AbstractDimArray,M::AbstractDimArray,Tx::Union{Ti,Vector})
function BLUEs.convolve(x::VectorArray,M::AbstractDimArray,Tx::Union{Ti,Vector})
    if ndims(M) == 2 
        return VectorArray(DimArray([convolve(x,M,Tx[tt]) for (tt,yy) in enumerate(Tx)],Tx))
    elseif ndims(M) == 3

        # do a sample calculation to get units.
        Msmall = M[:,:,1]
        yunit = unit.(vec(convolve(x,Msmall,Tx))[1]) # assume everything has the same units

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
# basically repeats previous function: any way to simplify?
function BLUEs.convolve(P::MatrixArray, M::AbstractDimArray, Tx::Union{Ti,Vector}) 
    T2 = typeof(parent(convolve(first(P),M,Tx)))
    Pyx = Array{T2}(undef,size(P))
    for i in eachindex(P)
        #Pyx[i] = parent(parent(convolve(P[i],M,Tx,coeffs)))
        Pyx[i] = parent(convolve(P[i],M,Tx))
    end
    return MatrixArray(DimArray(Pyx,domaindims(P)))
end

function BLUEs.convolve(P::MatrixArray,M::AbstractDimArray) 
    #function convolve(P::DimArray{T},M) where T<: AbstractDimArray
    # became more complicated when returning a scalar was not allowed
    T2 = typeof(first(parent(convolve(first(P),M))))
    println(T2)
    Pyx = Array{T2}(undef,size(P))
    for i in eachindex(P)
        #Pyx[i] = convolve(P[i],M)
        Pyx[i] = first(parent(convolve(P[i],M)))
    end
    return AlgebraicArray(Pyx,RowVector(["1"]),rangedims(P))
end

# basically repeats previous function: any way to simplify?
function BLUEs.convolve(P::MatrixArray,M::AbstractDimArray,coeffs::DimVector)
    T2 = typeof(first(parent(convolve(first(P),M,coeffs))))
    #T2 = typeof(convolve(first(P),M,coeffs))
    Pyx = Array{T2}(undef,size(P))
    for i in eachindex(P)
        #        Pyx[i] = convolve(P[i],M,coeffs)
        Pyx[i] = first(parent(convolve(P[i],M,coeffs)))
    end
    #return DimArray(Pyx,dims(P))
    return transpose(VectorArray(DimArray(Pyx,rangedims(P))))
    #return AlgebraicArray(Pyx,RowVector(["1"]),rangedims(P))
    #return MatrixArray(DimArray(Pyx,(RowVector(["1"]),rangedims(P))))
end

function BLUEs.convolve(x::VectorArray, M::AbstractDimArray, coeffs::DimVector) 
    statevars = dims(x,3) # equal to rangedims(x)[3]
    vals = sum([convolve(x[:,:,At(s)], M)  * coeffs[At(s)] for s in statevars])
    #return sum([convolve(x[:,:,At(s)], M)  * coeffs[At(s)] for s in statevars])
    (vals isa Number) ? (return VectorArray(DimArray([vals],first(rangedims(x))))) : (return VectorArray(AlgebraicArray(vals,first(rangedims(x)))))
end

function BLUEs.convolve(x::VectorArray, M::AbstractDimArray, Tx::Ti, coeffs::DimVector) # where T <: Number
    if ndims(M) == 2
        return VectorArray(DimArray([convolve(x, M, Tx[tt], coeffs) for tt in eachindex(Tx)], Tx))
    elseif ndims(M) == 3
        error("some code should go here")
    else
        error("M has wrong number of dims") 
    end
end
# basically repeats previous function: any way to simplify?
function BLUEs.convolve(P::MatrixArray, M::AbstractDimArray, Tx::Ti, coeffs::DimVector) 
    T2 = typeof(parent(convolve(first(P),M,Tx,coeffs)))
    Pyx = Array{T2}(undef,size(P))
    for i in eachindex(P)
        Pyx[i] = parent(convolve(P[i],M,Tx,coeffs))
    end
    return MatrixArray(DimArray(Pyx,domaindims(P)))
end

function BLUEs.convolve(x::VectorArray, M::AbstractDimArray, t::Number, coeffs::DimVector) 
    statevars = dims(x,3)
    return sum([convolve(x[:,:,At(s)], M, t)  * coeffs[At(s)] for s in statevars])
end

end 
