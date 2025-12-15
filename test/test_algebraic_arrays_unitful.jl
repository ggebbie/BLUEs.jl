@testset "algebraic arrays + unitful" begin
    using Measurements
    using AlgebraicArrays 

    # A scalar like N = 1 is not relevant here, would not need VectorArray 
    Nlist = [(1,1,1),(2,3)] # grid of obs
    for N in Nlist
        a = randn(N)u"K" .± rand(N)u"K"
	func_algebraic_arrays(a)
    end
end
