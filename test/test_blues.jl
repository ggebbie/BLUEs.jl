@testset "error propagation" begin
    using Measurements

    # check what happens for a scalar
    Mlist = (1,5)
    for M in Mlist
        a = randn(M) .± rand(M)
                E = randn(M,M)
        aval = Measurements.value.(a)
        aerr = Measurements.uncertainty.(a);

        # allow scalar input to Estimate constructor 
        x9 = Estimate(first(aval), first(aerr))

        x0 = Estimate(Measurements.value.(a),
            Diagonal(aerr.^2))
        x  = Estimate(aval, aerr) # just provide standard error
        x1  = Estimate(a) # just provide Vector{Measurement}

        @test   isequal(x.v, x0.v)
        @test   isequal(x.P, x0.P)
        @test   isequal(x1.v, x0.v)
        @test   isequal(x1.P, x0.P)
        @test Measurements.value.(E*a) ≈ (E*x).v
        @test Measurements.uncertainty.(E*a) ≈ (E*x).σ

        # combine two estimates
        xplus = combine(x,x,alg=:underdetermined)
        # error should decrease by 70%
        @test sum( xplus.σ./x.σ .< 0.8) == M
        # central estimate should not change
        @test isapprox( xplus.v, x.v )

        # combine two estimates another way
        xplus2 = combine(x,x,alg=:overdetermined)
        # error should decrease by 70%
        @test sum( xplus2.σ./x.σ .< 0.8) == M
        # central estimate should not change
        @test isapprox( xplus2.v, x.v )
    end
end

@testset "mixed signals: dimensionless matrix" begin
    N = 2
    for M in 1:4
        σₓ = rand()
        E = randn(M,N)
        Px⁻¹ = Diagonal(fill(σₓ^-1,N))
        v = randn(N)
        x = Estimate(v,inv(Px⁻¹))
        y = E*x

        if M == N
            xtilde1 = inv(E)*y # use inv for this special repeat, repeat with left divide below
            @test sum(isapprox.(x.v, xtilde1.v, atol=1e-8)) == N  
            @test sum(isapprox.(x.P, xtilde1.P, atol=1e-8)) == N^2
        end
        
        xtilde2 = E\y
        if  N ≤ M
            @test sum(isapprox.(x.v, xtilde2.v, atol=1e-8)) == N  
            @test sum(isapprox.(x.P, xtilde2.P, atol=1e-8)) == N^2
        end
    end
end

@testset "trend analysis: left-uniform matrix" begin

    M = 10  # number of obs
    t = collect(0:1:M-1)
    a = randn() # intercept
    b = randn() # slope
    ytrue = a .+ b.*t 
    ỹ = a .+ b.*t .+ randn(M)
    E = hcat(ones(M),t)
    Py⁻¹ = Diagonal(fill(1.0,M))

    y_estimate = Estimate(ytrue,inv(Py⁻¹))
    x = E\y_estimate # invert the observations to obtain solution
    @test isapprox(x.v[1],a)
    @test isapprox(x.v[2],b)
    
    # least squares solution method  
    problem = OverdeterminedProblem(ỹ,E,Py⁻¹)
    x̃ = solve(problem,alg=:textbook)
    x̃1 = solve(problem,alg=:hessian)
    @test cost(x̃,problem) < 5M # rough guide, could get unlucky and not pass
    @test cost(x̃1,problem) < 5M # rough guide, could get unlucky and not pass
end

@testset "left-uniform problem with prior info" begin
    y = [-1.9]
    σn = 0.2
    a = -0.24
    γδ = 1.0 
    γT = 1.0
    E = [1 a] 
    x₀ = [-1.0, 4.0]
        
    # "overdetermined, problem 1.4" 
    Px⁻¹ = Diagonal([γδ,γT])
    Py⁻¹  = Diagonal([σn.^-2])
            
    # "underdetermined, problem 2.1" 
    Py  = Diagonal([σn.^2])
    Px = Diagonal(ustrip.([γδ,γT].^-1))
    x0 = Estimate(x₀, Px)
    y_estimate = Estimate(y, Py)
    x1 = combine(x0, y_estimate, E) # a BLUE
    
    # least squares methods
    oproblem = OverdeterminedProblem(y,E,Py⁻¹,Px⁻¹,x₀)
    x̃1 = solve(oproblem)
    @test cost(x̃1,oproblem) < 1 # rough guide, coul

    uproblem = UnderdeterminedProblem(y,E,Py,Px,x₀)
    x̃2 = solve(uproblem)
    @test cost(x̃2,uproblem) < 1 # rough guide, could ge

    # same answer both least-squares ways?
    @test cost(x̃2,uproblem) ≈ cost(x̃1,oproblem)

    if use_units
        @test isapprox(ustrip.(x1.v), ustrip.(x̃1.v))
        @test isapprox(ustrip.(x1.σ), ustrip.(x̃1.σ))
    else
        @test isapprox(x1.v, x̃1.v)
        @test isapprox(x1.σ, x̃1.σ)
    end
end

@testset "polynomial fitting, problem 2.3" begin

    M = 50
    N = 4 
    t = (1:M)
    E =hcat(t.^0, t, t.^2, t.^3)
    σy = 0.1
    Py = Diagonal(fill(ustrip(σy^2),M))
    Py⁻¹ = Diagonal(fill(ustrip(σy^-2),M))
    γ = [1.0e1, 1.0e2, 1.0e3, 1.0e4]
    Px⁻¹ = Diagonal(γ)
    x₀ = zeros(N)
    
    Py¹² = cholesky(Py)
    Py⁻¹² = cholesky(Py⁻¹)
    Px = inv(Px⁻¹)
    Px¹² = cholesky(Px)
    x = Px¹².L*randn(N)
    y = E*x

    x0 = Estimate(x₀, Px)
    y_estimate = Estimate(y, Py)
    x1 = combine(x0, y_estimate, E) # a BLUE

    oproblem = OverdeterminedProblem(y,E,Py⁻¹,Px⁻¹,x₀)
    
    # not perfect data fit b.c. of prior info
    x̃ = solve(oproblem,alg=:hessian)
    @test cost(x̃,oproblem) < 5M
    x̃ = solve(oproblem,alg=:textbook)
    @test cost(x̃,oproblem) < 5M
    @test isapprox(x1.v, x̃.v)
    @test isapprox(x1.σ, x̃.σ)
end

@testset "overdetermined problem for mean with autocovariance, problem 4.1" begin

end

# additional problem: 5.1 model of exponential decay
# Not implemented/tested for `combine` function of two Estimates.
@testset "overdetermined named tuple E,y" begin
    N = 2
    M = 1
    σₓ = rand()
    E1 = randn(M,N)
    E2 = randn(M,N)
    E = (one=E1,two=E2)
    Py⁻¹1 = Diagonal(fill(σₓ^-1,M))
    x = randn(N)
    Py⁻¹ = (one=Py⁻¹1, two =Py⁻¹1)
    # create perfect data
    y = E*x
    problem = OverdeterminedProblem(y,E,Py⁻¹)
    x̃ = solve(problem,alg=:hessian)
    @test x ≈ x̃.v # no noise in obs
    @test cost(x̃,problem) < 1e-5 # no noise in obs
end
