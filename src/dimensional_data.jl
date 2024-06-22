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
    inside_type = typeof(√P[1][1])
    sigma = Array{inside_type}(undef,size(P))
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

"""
     function left divide

     Left divide of Multipliable Matrix.
     Reverse mapping from unitdomain to range.
     Is `exact` if input is exact.
"""
function Base.:\(A::AbstractDimMatrix,b::AbstractDimVector)
    DimensionalData.comparedims(first(dims(A)), first(dims(b)); val=true)
    return rebuild(A,parent(A)\parent(b),(last(dims(A)),)) 
end

"""
function convolve(x::DimArray{T},E::AbstractDimArray) where T <: Number

Take the convolution of E and x
Account for proper overlap of dimension.
Sum and take into account units.
Return an `AbstractDimArray`
"""
function convolve(x::DimArray{T},E::AbstractDimArray) where T <: Number
    tnow = last(first(dims(x)))
    lags = first(dims(E))
    vals = sum([E[ii,:] ⋅ x[Near(tnow-ll),:] for (ii,ll) in enumerate(lags)])
    (vals isa Number) ? (return DimArray([vals],first(dims(x)))) : (return DimArray(vals,first(dims(x))))
end
"""
function convolve(x::AbstractDimArray, E::AbstractDimArray, coeffs::UnitfulMatrix}
    the `coeffs` argument signifies that x is a 3D array (i.e. >1 state variables)

    this function both convolves, and linearly combines the propagated state variables
"""
function convolve(x::AbstractDimArray,E::AbstractDimArray, coeffs::UnitfulMatrix)
    statevars = x.dims[3]
    mat = UnitfulMatrix(transpose([convolve(x[:,:,At(s)], E) for s in statevars])) * coeffs
    return getindexqty(mat, 1,1)
end

function convolve(x::AbstractDimArray,E::AbstractDimArray,t::Number)
    lags = first(dims(E))
    return sum([E[ii,:] ⋅ x[Near(t-ll),:] for (ii,ll) in enumerate(lags)])
end

#coeffs signifies that x is 3D 
function convolve(x::AbstractDimArray, E::AbstractDimArray, t::Number, coeffs::UnitfulMatrix)
    statevars = x.dims[3]
    mat = UnitfulMatrix(transpose([convolve(x[:,:,At(s)], E, t) for s in statevars]))*coeffs
    return getindexqty(mat, 1,1) 
end

#don't handle the ndims(M) == 3 case here but I'll get back to it
function convolve(x::AbstractDimArray, M::AbstractDimArray, Tx::Union{Ti, Vector}, coeffs::UnitfulMatrix)
    if ndims(M) == 2
        return DimArray([convolve(x,M,Tx[tt], coeffs) for (tt, yy) in enumerate(Tx)], Tx)
    elseif ndims(M) == 3
        error("some code should go here")
    else
        error("M has wrong number of dims") 
    end
end

function convolve(x::AbstractDimArray,M::AbstractDimArray,Tx::Union{Ti,Vector})
    if ndims(M) == 2 
        return DimArray([convolve(x,M,Tx[tt]) for (tt,yy) in enumerate(Tx)],Tx)
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

"""
function observe(P::AbstractDimArray)

functional form of E*P
P only stores the diagonals as could be
identified from its type.
"""
function observematrix(P::DimArray,M)

    #out = zeros(dims(P))
    #typeout = typeof(out)
    in = convolve(first(P),M)
    typein = typeof(in)
    Pyx = Array{typein}(undef,size(P))
    for i in eachindex(P)
#        Pyx[i] = observe(P[i])
        Pyx[i] = convolve(P[i],M)
    end
    return DimArray(Pyx,dims(P))
end
function convolve(P::DimArray{T},M) where T<: AbstractDimArray
    T2 = typeof(convolve(first(P),M))
    Pyx = Array{T2}(undef,size(P))
    for i in eachindex(P)
#        Pyx[i] = observe(P[i])
        Pyx[i] = convolve(P[i],M)
    end
    return DimArray(Pyx,dims(P))
end

function diagonalmatrix(Pdims::Tuple)

    tmp = zeros(Pdims)
    typetmp = typeof(tmp)

    P = Array{typetmp}(undef,size(tmp))

    for i in eachindex(P)
        P[i] = zeros(Pdims)
        P[i][i] += 1.0
    end
    return DimArray(P,Pdims)
end

"""
function algebraic_object(P::DimArray{T}) where T <: AbstractDimArray
"""
function algebraic_object(P::DimArray{T}) where T <: AbstractDimArray
    # number of columns/ outer dims
    N = length(P)
    # number of rows, take first inner element as example
    M = length(first(P))
    A = Matrix{eltype(first(P))}(undef,M,N)
    if N > 1  
        for j in eachindex(P)
            A[:,j] = P[j][:]
        end
    elseif N == 1
        for i in eachindex(first(P))
            A[i,1] = first(P)[i]
        end
    end
    return A 
end

"""
function algebraic_object(P::DimArray{Number})
"""
function algebraic_object(P::DimArray{T}) where T <: Number
    M = length(P)
    A = Vector{T}(undef,M)
    for i in eachindex(P)
        A[i] = P[i]
    end
    return A 
end

"""
function matrix_to_dimarray(A,rangedims,domaindims)
"""
function matrix_to_dimarray(A,rangedims,domaindims)

    # extra step for type stability
    Q1 = reshape(A[:,1],size(rangedims))
    P1 = DimArray(Q1, rangedims)

    P = Array{typeof(P1)}(undef,size(domaindims))
    for j in eachindex(P)
        Q = reshape(A[:,j],size(rangedims))
        P[j] = DimArray(Q, rangedims)
    end
    return DimArray(P, domaindims)
end

"""
function vector_to_dimarray(A,rangedims)
"""
function vector_to_dimarray(A,rangedims)
    Q1 = reshape(A,size(rangedims))
    return DimArray(Q1, rangedims)
end

function algebraic_transpose(P::DimArray)
    ddims = dims(P)
    rdims = dims(first(P))
    A = algebraic_object(P)
    return matrix_to_dimarray( transpose(A), ddims, rdims)
end

function ldiv(A::DimArray, b::DimArray) 
    Amat = algebraic_object(A) \ algebraic_object(b)
    (Amat isa Number) && (Amat = [Amat])
    ddims = dims(b)
    rdims = dims(A)
    return matrix_to_dimarray(Amat, rdims, ddims)
#    return DimArray(reshape(Amat, size(dims(A))), dims(A))
end
function ldiv(A::DimArray{T1}, b::DimArray{T2}) where T1<: AbstractDimArray where T2 <: Number
    Amat = algebraic_object(A) \ algebraic_object(b)
    (Amat isa Number) && (Amat = [Amat])
    return DimArray(reshape(Amat, size(dims(A))), dims(A))
end

function matmul(A::DimArray, b::DimArray{T}) where T <: Number
    Amat = algebraic_object(A) * algebraic_object(b)
    (Amat isa Number) && (Amat = [Amat])
    rdims = dims(first(A))
    return vector_to_dimarray(Amat, rdims)
  # return DimArray( reshape(Amat, size(rdims)), rdims)
end
function matmul(A::DimArray, b::DimArray)
    Amat = algebraic_object(A) * algebraic_object(b)
    (Amat isa Number) && (Amat = [Amat])
#    rdims = dims(first(A))
 #   return DimArray( reshape(Amat, size(rdims)), rdims)
    ddims = dims(b)
    rdims = dims(first(A))
    return matrix_to_dimarray(Amat, rdims, ddims)

end

"""
function combine(x0::Estimate,y::Estimate,fmat::Function,fvec::Function,farg)
"""
function combine(x0::Estimate,y::Estimate,fmat::Function,fvec::Function,farg)
            Cyx = fmat(x0.P,farg) 
            Cxy = algebraic_transpose(Cyx)
            ECxy = fmat(Cxy,farg)
            Cyy = ECxy + y.P
            y0 = fvec(x0.v,farg)
            n = y.v - y0
            tmp = ldiv(Cyy, n)
            v = matmul(Cxy, tmp)
            Pdecrease = matmul(Cxy,BLUEs.ldiv(Cyy,Cyx))
            P = x0.P - Pdecrease
            return Estimate(v,P)
end
function combine(x0::Estimate,y::Estimate,f::Function)
            Cyx = f(x0.P) 
            Cxy = algebraic_transpose(Cyx)
            ECxy = f(Cxy)
            Cyy = ECxy + y.P
            y0 = f(x0.v)
            n = y.v - y0
            tmp = ldiv(Cyy, n)
            v = matmul(Cxy, tmp)
            Pdecrease = matmul(Cxy,BLUEs.ldiv(Cyy,Cyx))
            P = x0.P - Pdecrease
            return Estimate(v,P)
end
