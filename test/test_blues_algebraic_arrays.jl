@testset "algebraic arrays" begin
    using Measurements
    using AlgebraicArrays 

    include("function_algebraic_arrays.jl")
    
    # A scalar like N = 1 is not relevant here, would not need VectorArray 
    Nlist = [(1,1,1),(2,3)] # grid of obs
    for N in Nlist
        a = randn(N) .± rand(N)
        func_algebraic_arrays(a)
    end
end
