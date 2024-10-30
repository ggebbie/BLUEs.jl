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

"""
function convolve(x::DimArray{T},E::AbstractDimArray) where T <: Number

Take the convolution of E and x
Account for proper overlap of dimension.
Sum and take into account units.
Return an `AbstractDimArray`
"""
#function convolve(x::DimArray{T},E::AbstractDimArray) where T <: Number
function convolve(x::VectorArray,E::AbstractDimArray) 
    tnow = last(first(rangedims(x)))
    lags = first(dims(E))
    vals = sum([E[ii,:] ⋅ x[Near(tnow-ll),:] for (ii,ll) in enumerate(lags)])
    #(vals isa Number) ? (return DimArray([vals],first(dims(x)))) : (return DimArray(vals,first(dims(x))))
    (vals isa Number) ? (return vals) : (return DimArray(vals,first(dims(x))))
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

function convolve(P::MatrixArray,M) 
#function convolve(P::DimArray{T},M) where T<: AbstractDimArray
    T2 = typeof(convolve(first(P),M))
    Pyx = Array{T2}(undef,size(P))
    for i in eachindex(P)
        Pyx[i] = convolve(P[i],M)
    end
    return AlgebraicArray(Pyx,RowVector(["1"]),rangedims(P))
end
# basically repeats previous function: any way to simplify?
function convolve(P::DimArray{T},M::AbstractDimArray,coeffs::DimVector) where T<: AbstractDimArray
    T2 = typeof(convolve(first(P),M,coeffs))
    Pyx = Array{T2}(undef,size(P))
    for i in eachindex(P)
#        Pyx[i] = observe(P[i])
        Pyx[i] = convolve(P[i],M,coeffs)
    end
    return DimArray(Pyx,dims(P))
end

function convolve(x::DimArray{T}, M::AbstractDimArray, coeffs::DimVector) where T <: Number
    statevars = dims(x,3)
    return sum([convolve(x[:,:,At(s)], M)  * coeffs[At(s)] for s in statevars])
end

function convolve(x::DimArray{T}, M::AbstractDimArray, Tx::Ti, coeffs::DimVector) where T <: Number
    if ndims(M) == 2
        return DimArray([convolve(x, M, Tx[tt], coeffs) for tt in eachindex(Tx)], Tx)
    elseif ndims(M) == 3
        error("some code should go here")
    else
        error("M has wrong number of dims") 
    end
end
# basically repeats previous function: any way to simplify?
function convolve(P::DimArray{T},M::AbstractDimArray,Tx::Ti, coeffs::DimVector) where T<: AbstractDimArray
    T2 = typeof(convolve(first(P),M,Tx,coeffs))
    Pyx = Array{T2}(undef,size(P))
    for i in eachindex(P)
        Pyx[i] = convolve(P[i],M,Tx,coeffs)
    end
    return DimArray(Pyx,dims(P))
end

function convolve(x::AbstractDimArray, E::AbstractDimArray, t::Number, coeffs::DimVector)
    statevars = dims(x,3)
    return sum([convolve(x[:,:,At(s)], E, t) * coeffs[At(s)] for s in statevars])
end

# function convolve(x::AbstractDimArray,E::AbstractDimArray,t::Number)
#     lags = first(dims(E))
#     return sum([E[ii,:] ⋅ x[Near(t-ll),:] for (ii,ll) in enumerate(lags)])
# end

function uncertainty_units(x::DimArray{T}) where T<: Number

    unitlist = unit.(x)
    U = Array{typeof(unitlist)}(undef,size(unitlist))

    for i in eachindex(U)
        U[i] = similar(unitlist)# Array{typeof(unitlist)}(undef,size(unitlist))
        for j in eachindex(U)
            U[i][j] = unitlist[i]*unitlist[j]
        end
    end
    return DimArray(U,dims(x))
end

function diagonalmatrix_with_units(x::DimArray{T}) where T<: Number

    unitlist = 1.0.*unit.(x)
    U = Array{typeof(unitlist)}(undef,size(unitlist))

    for i in eachindex(U)
        U[i] = similar(unitlist)
        #U[i] = Array{Quantity}(undef,size(unitlist))
        for j in eachindex(U)
            if i == j 
                U[i][j] = unitlist[i]*unitlist[j]
            else
                U[i][j] = 0.0.*unitlist[i]*unitlist[j]
            end
        end
    end
    return DimArray(U,dims(x))
end

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
    EPxy isa Number ? Pyy = [EPxy;;] + y1.P : Pyy = EPxy + y1.P
    #Pyy = EPxy + y1.P
    y0 = E1(x0.v)
    y0 isa Number ? n1 = y1.v - [y0] : n1 = y1.v - y0
    tmp = Pyy \ n1
    v = Pxy * tmp
    dP = Pxy * (Pyy \ Pyx)
    P = x0.P - dP
    return Estimate(v,P)
end
