@testset "algebraic arrays constructors" begin
    using Measurements

    # A scalar like N = 1 is not relevant here, would not need VectorArray 
    Nlist = [(1,1,1),(2,3)] # grid of obs
    for N in Nlist
        if use_units
            a = randn(N)u"K" .± rand(N)u"K"
        else
            a = randn(N) .± rand(N)
        end

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
    end
end

@testset "objective mapping, problem 4.3" begin

    M = 11
    if use_units
        yr = u"yr"; cm = u"cm"
        τ = range(0.0yr,5.0yr,step=0.1yr)
        ρ = exp.(-τ.^2/(1yr)^2)
        n = length(ρ)
        Px = UnitfulMatrix(SymmetricToeplitz(ρ),fill(cm,n),fill(cm^-1,n),exact=true) +
            Diagonal(fill(1e-6,n),fill(cm,n),fill(cm^-1,n))
        σy = 0.1cm
        Py = Diagonal(fill(ustrip(σy),M),fill(cm,M),fill(cm^-1,M))
        Enm = sparse(1:M,1:5:n,fill(1.0,M))
        E = UnitfulMatrix(Enm,fill(cm,M),fill(cm,n))
        x₀ = zeros(n)cm
    else
        τ = range(0.0,5.0,step=0.1)
        ρ = exp.(-τ.^2/(1)^2)
        n = length(ρ)
        Px = SymmetricToeplitz(ρ) + Diagonal(fill(1e-6,n))
        σy = 0.1
        Py = Diagonal(fill(σy,M))
        E = sparse(1:M,1:5:n,fill(1.0,M))
        x₀ = zeros(n)
    end
    
    Px¹² = cholesky(Px)
    x = Px¹².L*randn(n)
    y = E*x
    x0 = Estimate(x₀, Px)
    y_estimate = Estimate(y, Py)
    x1 = combine(x0, y_estimate, E) # a BLUE
    
    uproblem = UnderdeterminedProblem(y,E,Py,Px,x₀)
    x̃ = solve(uproblem)
    @test cost(x̃,uproblem) < 5M 

    if use_units
        @test isapprox(ustrip.(x1.v), ustrip.(x̃.v))
        @test isapprox(ustrip.(x1.σ), ustrip.(x̃.σ))
    else
        @test isapprox(x1.v, x̃.v)
        @test isapprox(x1.σ, x̃.σ)
    end

end
