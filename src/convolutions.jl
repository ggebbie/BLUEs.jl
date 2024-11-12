
"""
function convolve(x::DimArray{T},E::AbstractDimArray) where T <: Number

Take the convolution of E and x
Account for proper overlap of dimension.
Sum and take into account units.
Return an `AbstractDimArray`
"""
function convolve(x::VectorArray,E::AbstractDimArray) 
    tnow = last(first(rangedims(x)))
    lags = first(dims(E))
    vals = sum([E[ii,:] ⋅ x[Near(tnow-ll),:] for (ii,ll) in enumerate(lags)])
    #(vals isa Number) ? (return DimArray([vals],first(dims(x)))) : (return DimArray(vals,first(dims(x))))
    #(vals isa Number) ? (return vals) : (return DimArray(vals,first(dims(x))))
    (vals isa Number) ? (return VectorArray(DimArray([vals],first(rangedims(x))))) : (return VectorArray(AlgebraicArray(vals,first(rangedims(x)))))
end
"""
function convolve(x::AbstractDimArray, E::AbstractDimArray, coeffs::UnitfulMatrix}
    the `coeffs` argument signifies that x is a 3D array (i.e. >1 state variables)

    this function both convolves, and linearly combines the propagated state variables
"""
# function convolve(x::AbstractDimArray,E::AbstractDimArray, coeffs::UnitfulMatrix)
#     statevars = x.dims[3]
#     mat = UnitfulMatrix(transpose([convolve(x[:,:,At(s)], E) for s in statevars])) * coeffs
#     return getindexqty(mat, 1,1)
# end

function convolve(x::VectorArray, M::AbstractDimArray, t::Number)
    lags = first(dims(M))
    return sum([M[ii,:] ⋅ x[Near(t-ll),:] for (ii,ll) in enumerate(lags)])
end

#coeffs signifies that x is 3D 
function convolve(x::AbstractDimArray, E::AbstractDimArray, t::Number, coeffs::UnitfulMatrix)
    statevars = x.dims[3]
    mat = UnitfulMatrix(transpose([convolve(x[:,:,At(s)], E, t) for s in statevars]))*coeffs
    return getindexqty(mat, 1,1) 
end

#don't handle the ndims(M) == 3 case here but I'll get back to it
function convolve(x::VectorArray, M::AbstractDimArray, Tx::Union{Ti, Vector}, coeffs::UnitfulMatrix)
    if ndims(M) == 2
        return DimArray([convolve(x,M,Tx[tt], coeffs) for (tt, yy) in enumerate(Tx)], Tx)
    elseif ndims(M) == 3
        error("some code should go here")
    else
        error("M has wrong number of dims") 
    end
end

#function convolve(x::AbstractDimArray,M::AbstractDimArray,Tx::Union{Ti,Vector})
function convolve(x::VectorArray,M::AbstractDimArray,Tx::Union{Ti,Vector})
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
function convolve(P::MatrixArray, M::AbstractDimArray, Tx::Union{Ti,Vector}) 
    T2 = typeof(parent(convolve(first(P),M,Tx)))
    Pyx = Array{T2}(undef,size(P))
    for i in eachindex(P)
        #Pyx[i] = parent(parent(convolve(P[i],M,Tx,coeffs)))
        Pyx[i] = parent(convolve(P[i],M,Tx))
    end
    return MatrixArray(DimArray(Pyx,domaindims(P)))
end

function convolve(P::MatrixArray,M) 
    #function convolve(P::DimArray{T},M) where T<: AbstractDimArray
    # became more complicated when returning a scalar was not allowed
    T2 = typeof(first(parent(convolve(first(P),M))))
    Pyx = Array{T2}(undef,size(P))
    for i in eachindex(P)
        #Pyx[i] = convolve(P[i],M)
        Pyx[i] = first(parent(convolve(P[i],M)))
    end
    return AlgebraicArray(Pyx,RowVector(["1"]),rangedims(P))
end

# basically repeats previous function: any way to simplify?
function convolve(P::MatrixArray,M::AbstractDimArray,coeffs::DimVector)
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

function convolve(x::VectorArray, M::AbstractDimArray, coeffs::DimVector) 
    statevars = dims(x,3) # equal to rangedims(x)[3]
    vals = sum([convolve(x[:,:,At(s)], M)  * coeffs[At(s)] for s in statevars])
    #return sum([convolve(x[:,:,At(s)], M)  * coeffs[At(s)] for s in statevars])
    (vals isa Number) ? (return VectorArray(DimArray([vals],first(rangedims(x))))) : (return VectorArray(AlgebraicArray(vals,first(rangedims(x)))))
end

function convolve(x::VectorArray, M::AbstractDimArray, Tx::Ti, coeffs::DimVector) # where T <: Number
    if ndims(M) == 2
        return VectorArray(DimArray([convolve(x, M, Tx[tt], coeffs) for tt in eachindex(Tx)], Tx))
    elseif ndims(M) == 3
        error("some code should go here")
    else
        error("M has wrong number of dims") 
    end
end
# basically repeats previous function: any way to simplify?
function convolve(P::MatrixArray, M::AbstractDimArray, Tx::Ti, coeffs::DimVector) 
    #T2 = typeof(convolve(first(P),M,Tx,coeffs))
    #T2 = typeof(first(parent(convolve(first(P),M,Tx,coeffs))))
    #T2 = typeof(parent(parent(convolve(first(P),M,Tx,coeffs))))
    T2 = typeof(parent(convolve(first(P),M,Tx,coeffs)))
    Pyx = Array{T2}(undef,size(P))
    for i in eachindex(P)
        #Pyx[i] = parent(parent(convolve(P[i],M,Tx,coeffs)))
        Pyx[i] = parent(convolve(P[i],M,Tx,coeffs))
    end
    return MatrixArray(DimArray(Pyx,domaindims(P)))
end

function convolve(x::VectorArray, M::AbstractDimArray, t::Number, coeffs::DimVector)
    statevars = dims(x,3)
    return sum([convolve(x[:,:,At(s)], M, t)  * coeffs[At(s)] for s in statevars])
end
