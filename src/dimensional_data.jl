
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

standard_error(P::AbstractDimArray{T,2}) where T <: Number = DimArray(.√diag(x.P),first(dims(x.P)))
