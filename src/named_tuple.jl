"""
    multiplication for `NamedTuple`s

    Matrix multiply, M*x
"""
function *(A::NamedTuple, b::AbstractVector) 
    # Update to use parametric type to set type of Vector
    c = Vector(undef, length(A))
    for (i, V) in enumerate(A)
        c[i] = V*b 
    end
    return NamedTuple{keys(A)}(c)
end
function *(A::NamedTuple, b::NamedTuple)
    ~(keys(A) == keys(b)) && error("named tuples don't have consistent fields")
    
    # Update to use parametric type to set type of Vector
    c = Vector(undef, length(A))
    for (i, V) in enumerate(A)
        c[i] = V*b[i] # b index safe?
    end
    return NamedTuple{keys(A)}(c)
end
function *(b::T,A::NamedTuple) where T<:Number
    
    # Update to use parametric type to set type of Vector
    # Overwriting A would be more efficient
    c = Vector(undef, length(A))
    #c = similar(A)
    for (i, V) in enumerate(A)
        c[i] = V*b # b index safe?
    end
    return NamedTuple{keys(A)}(c)
end
function -(A::NamedTuple) 
    
    # Update to use parametric type to set type of Vector
    # Overwriting A would be more efficient
    c = Vector(undef, length(A))
    for (i, V) in enumerate(A)
        c[i] = -V # b index safe?
    end
    return NamedTuple{keys(A)}(c)
end
function -(A::NamedTuple,B::NamedTuple) 
    
    # Update to use parametric type to set type of Vector
    # Overwriting A would be more efficient
    c = Vector(undef, length(A))
    for (i, V) in enumerate(A)
        c[i] = A[i]-B[i] # b index safe?
    end
    return NamedTuple{keys(A)}(c)
end

function +(A::NamedTuple,B::NamedTuple) 
    
    # Update to use parametric type to set type of Vector
    # Overwriting A would be more efficient
    c = Vector(undef, length(A))
    for (i, V) in enumerate(A)
        c[i] = A[i]+B[i] # b index safe?
    end
    return NamedTuple{keys(A)}(c)
end

function sum(A::NamedTuple) 
    # Update to use parametric type to initialize output
    Asum = 0 * A[1] # a kludge
    for (i, V) in enumerate(A)
        Asum += V 
    end
    return Asum
end

function LinearAlgebra.transpose(A::NamedTuple) 
    
    # Update to use parametric type to set type of Vector
    c = Vector(undef, length(A))
    for (i, V) in enumerate(A)
        c[i] = transpose(V) # b index safe?
    end
    return NamedTuple{keys(A)}(c)
end
