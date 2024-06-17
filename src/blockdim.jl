struct BlockDimArray{T <: Number, DA <: AbstractDimArray{T}} 
    da :: DA
    blockdims :: Tuple
end

#groupby(A, dims(A, Ti))
