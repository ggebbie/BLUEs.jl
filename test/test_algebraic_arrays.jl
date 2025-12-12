## with units, doesn't construct 10-31
x = Estimate(a)
        
 aval = VectorArray(Measurements.value.(a))
 aerr = VectorArray(Measurements.uncertainty.(a))
 x1 = Estimate(aval,aerr)
 x2 = Estimate(aval, Diagonal(aerr.^2))

 @test x.v == x1.v
 @test x1.v == x2.v
 @test x.P == x1.P
 @test x1.P == x2.P
 
 M = (1,4) # grid of estimated values
 # force E to be a Matrix
 E = randn(M, N, :MatrixArray)
 
 # need to define vector times vector
 # or force E to be a matrix
 @test Measurements.value.(E*VectorArray(a)) ≈ (E*x).v
 @test Measurements.uncertainty.(E*VectorArray(a)) ≈ (E*x).σ

 # combine two estimates
 xplus = combine(x,x,alg=:underdetermined)
 # error should decrease by 70%
 @test sum( xplus.σ./x.σ .< 0.8) == prod(N)
 # central estimate should not change
 @test isapprox( xplus.v, x.v )

 # combine two estimates another way
 xplus2 = combine(x,x,alg=:underdetermined)
 # error should decrease by 70%
 @test sum( xplus2.σ./x.σ .< 0.8) == prod(N)
 # central estimate should not change
 @test isapprox( xplus2.v, x.v )
