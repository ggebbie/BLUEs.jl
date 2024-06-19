function show(io::IO, mime::MIME{Symbol("text/plain")}, x::DimArray{Quantity{Float64}, 3})
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

standard_error(P::AbstractDimArray{T,2}) where T <: Number = DimArray(.√diag(P),first(dims(P)))

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
