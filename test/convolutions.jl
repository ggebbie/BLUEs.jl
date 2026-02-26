function firstcol(P::MatrixArray)
    # number of column dimensions
    ncoldims = length(last(Px0.dims))
    nrowdims = length(first(Px0.dims))
    icol = Tuple(fill(1,ncoldims))
    irow = Tuple(fill(Colon(), nrowdims))
    return P[irow,icol]
end

returncol(P::MatrixDimArray, colno) = VectorArray(DimArray(reshape(P[:,colno],size(rangedims(P))),rangedims(P)))

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
    println("tnow ",tnow)
    println("lags ",lags)
    println("vals ", vals)
    println(" vals isa Number ",vals isa Number)
    println("first(rangedims(x)) ",first(rangedims(x)))

    # (tnow isa Number) ? (tdim = Ti([tnow])) : (tdim = Ti(tnow))
        # (vals isa Number) ? (return VectorArray(DimArray([vals], tdim))) :
    # (return VectorArray(AlgebraicArray(vals, tdim)))

    # works with all previous cases, but not DD timeseries
    (vals isa Number) ? (return VectorArray(DimArray([vals],first(rangedims(x))))) : (return VectorArray(AlgebraicArray(vals,first(rangedims(x)))))

    # Tdims = Ti((ustrip(tnow):ustrip(tnow))yr)
    # if tnow isa Number
    #     Tdims = Ti([tnow])
    # else
    #     Tdims = Ti(tnow)
    # end

    # if vals isa Number
    #     da = DimArray([vals],Tdims)
    #     println(da)
    #     return VectorArray(da)
    # else
    #     # return VectorArray(AlgebraicArray(vals,first(rangedims(x))))
    #     return VectorArray(AlgebraicArray(vals,Tdims))
    # end
end

function convolve(x::VectorArray, M::AbstractDimArray, t::Number)
    lags = first(dims(M))
    return sum([M[ii,:] ⋅ x[Near(t-ll),:] for (ii,ll) in enumerate(lags)])
end

function convolve(x::VectorArray, M::AbstractDimArray, Tx::Union{Ti,Vector})
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
        Pyx[i] = parent(convolve(P[i],M,Tx))
    end
    return MatrixArray(DimArray(Pyx,domaindims(P)))
end

function convolve(P::MatrixDimArray{T},M::AbstractDimArray) where T
    outputdims = first(rangedims(P))
    Pyx = Array{T}(undef,length(outputdims),size(P,2))
    for j in 1:size(P,2)
        Pyx[:,j] = parent(convolve(returncol(P,j), M))
    end
    arr = reshape(Pyx, length(outputdims), size(domaindims(P))...)
    da = DimArray(arr, (outputdims, domaindims(P)...))
    return AlgebraicArray(da, ((length(outputdims),),size(domaindims(P))))
end

function convolve(x::VectorArray, M::AbstractDimArray, coeffs::DimVector) 
    statevars = dims(x,3) # equal to rangedims(x)[3]
    vals = sum([convolve(x[:,:,At(s)], M)  * coeffs[At(s)] for s in statevars])
    da = 
        (vals isa Number) ?
        (return VectorArray(DimArray([vals],first(rangedims(x))))) :
        (return VectorArray(DimArray(vals,first(rangedims(x)))))
end

# basically repeats previous function: any way to simplify?
function convolve(P::MatrixDimArray{T}, M::AbstractDimArray, coeffs::DimVector) where T
    # T2 = typeof(first(parent(convolve(first(P),M,coeffs)))
    outputdims = first(rangedims(P))
    Pyx = Array{T}(undef,length(outputdims),size(P,2))

    # Pyx = Array{T}(undef,size(P))
    for j in 1:size(P,2)
        Pyx[:,j] = parent(convolve(returncol(P,j), M, coeffs))
    end
    arr = reshape(Pyx, length(outputdims), size(domaindims(P))...)
    da = DimArray(arr, (outputdims, domaindims(P)...))
    return AlgebraicArray(da, ((length(outputdims),),size(domaindims(P))))
end

function convolve(x::VectorDimArray, M::AbstractDimArray, Tx::Ti, coeffs::DimVector) # where T <: Number
    if ndims(M) == 2
        return VectorArray(DimArray([convolve(x, M, Tx[tt], coeffs) for tt in eachindex(Tx)], Tx))
    elseif ndims(M) == 3
        error("some code should go here")
    else
        error("M has wrong number of dims") 
    end
end
# basically repeats previous function: any way to simplify?
function convolve(P::MatrixDimArray{T}, M::AbstractDimArray, Tx::Ti, coeffs::DimVector) where T
    
    # T2 = typeof(parent(convolve(first(P),M,Tx,coeffs)))
    outputdims = first(rangedims(P))
    Pyx = Array{T}(undef,length(outputdims),size(P,2))

    for j in 1:size(P,2)
        Pyx[:,j] = parent(convolve(returncol(P,j), M, Tx, coeffs))
    end
    arr = reshape(Pyx, length(outputdims), size(domaindims(P))...)
    da = DimArray(arr, (outputdims, domaindims(P)...))
    return AlgebraicArray(da, ((length(outputdims),),size(domaindims(P))))

    
    # Pyx = Array{T}(undef,size(P))
    # for i in eachindex(P)
    #     Pyx[i] = parent(convolve(P[i],M,Tx,coeffs))
    # end
    # return MatrixArray(DimArray(Pyx,domaindims(P)))
end

function convolve(x::VectorArray, M::AbstractDimArray, t::Number, coeffs::DimVector) 
    statevars = dims(x,3)
    return sum([convolve(x[:,:,At(s)], M, t)  * coeffs[At(s)] for s in statevars])
end

